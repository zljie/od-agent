"""
采购意图路由器
负责意图识别、槽位提取和执行
"""

import json
import re
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .intents import PROCUREMENT_INTENTS, Intent, get_intent_by_id
from .connector import ProcurementConnector, get_procurement_connector, ConnectorResponse


@dataclass
class IntentMatch:
    intent: Intent
    confidence: float
    extracted_slots: Dict[str, Any]
    missing_slots: List[str] = field(default_factory=list)
    message: str = ""
    llm_inferred: bool = False


@dataclass
class ExecutionResult:
    success: bool
    intent: Intent
    connector_response: Optional[ConnectorResponse] = None
    response_text: str = ""
    error: Optional[str] = None


class IntentRouter:
    """采购意图路由器"""

    def __init__(self, connector: Optional[ProcurementConnector] = None):
        self.intents = PROCUREMENT_INTENTS
        self.connector = connector or get_procurement_connector()
        self._build_keyword_index()

    def _build_keyword_index(self):
        """构建关键词索引"""
        self.keyword_index: Dict[str, List[tuple]] = {}
        for intent in self.intents:
            for keyword in intent.trigger_keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append((intent.id, len(keyword)))

    def route(self, message: str) -> Optional[IntentMatch]:
        """路由用户消息到对应意图 - 优先使用LLM语义分析，fallback到关键词匹配"""
        # 优先使用 LLM 进行语义意图识别
        llm_match = self._llm_route(message)
        if llm_match:
            return llm_match

        # Fallback 到关键词匹配
        return self._keyword_route(message)

    def _llm_route(self, message: str) -> Optional[IntentMatch]:
        """使用 LLM 进行语义意图识别"""
        try:
            from ..models import get_default_model
            model = get_default_model()

            # 构建意图列表
            intent_list = "\n".join([
                f'- "{i.id}": {i.name} - {i.description}\n  关键词: {", ".join(i.trigger_keywords[:5])}'
                for i in self.intents
            ])

            prompt = f"""你是采购助手。请根据用户消息判断采购意图。

## 可用意图
{intent_list}

## 用户消息
"{message}"

## 要求
1. 仔细分析用户消息的语义，不要简单匹配关键词
2. 如果消息模糊，选择最可能的意图
3. 尝试从消息中提取关键参数（ID、编号、供应商名等）
4. 如果不属于任何已知意图，返回 "unknown"

请以JSON格式返回（不要有其他内容）：
{{"intent_id": "...", "confidence": 0.0-1.0, "extracted_params": {{"pr_id": "...", "po_id": "...", ...}}, "reasoning": "..."}}
"""
            print(f"[IntentRouter] LLM调用开始 | input_len={len(message)} | thinking_budget=500")
            llm_start = _time.time()
            response = model.generate(prompt, thinking_budget=500)
            llm_elapsed = (_time.time() - llm_start) * 1000
            if not response:
                print(f"[IntentRouter] LLM调用完成 | elapsed_ms={llm_elapsed:.1f} | response为空")
                return None
            print(f"[IntentRouter] LLM调用完成 | elapsed_ms={llm_elapsed:.1f} | response_len={len(response)}")
            print(f"[IntentRouter] LLM原始响应: {response[:300]}...")

            # 解析 JSON 响应
            result = json.loads(response.strip())
            intent_id = result.get("intent_id", "")

            if intent_id == "unknown" or not intent_id:
                return None

            intent = get_intent_by_id(intent_id)
            if not intent:
                return None

            confidence = float(result.get("confidence", 0.5))
            extracted_params = result.get("extracted_params", {})

            # 合并 LLM 提取的参数和正则提取的参数
            regex_params = self._extract_slots(intent, message)
            merged_params = {**extracted_params, **regex_params}

            # 检查缺失的必需槽位
            missing_slots = []
            for slot in intent.required_slots:
                if slot not in merged_params or not merged_params[slot]:
                    missing_slots.append(slot)

            print(f"[IntentRouter] LLM识别: {intent.name}, 置信度: {confidence:.2f}, 理由: {result.get('reasoning', '')}")

            return IntentMatch(
                intent=intent,
                confidence=confidence,
                extracted_slots=merged_params,
                missing_slots=missing_slots,
                message=message,
                llm_inferred=True
            )

        except json.JSONDecodeError as e:
            print(f"[IntentRouter] LLM响应JSON解析失败: {e}")
            return None
        except Exception as e:
            print(f"[IntentRouter] LLM意图识别失败: {e}")
            return None

    def _keyword_route(self, message: str) -> Optional[IntentMatch]:
        """基于关键词匹配的意图路由（fallback方案）"""
        message_lower = message.lower()

        # 记录匹配的关键词及其长度(优先匹配更长的关键词)
        matches: List[tuple] = []
        for keyword, intent_list in self.keyword_index.items():
            if keyword.lower() in message_lower:
                for intent_id, kw_len in intent_list:
                    matches.append((intent_id, kw_len))

        if not matches:
            return None

        # 按关键词长度降序排序,优先匹配更长的关键词
        matches.sort(key=lambda x: x[1], reverse=True)

        # 返回得分最高的意图ID
        top_intent_id = matches[0][0]
        intent = get_intent_by_id(top_intent_id)

        if not intent:
            return None

        # 计算置信度 (基于匹配关键词长度)
        max_kw_len = max(kw_len for _, kw_len in matches)
        confidence = max_kw_len / len(message) * 100 / 100.0  # 归一化到 0-1

        # 提取槽位
        extracted_slots = self._extract_slots(intent, message)

        # 检查缺失的必需槽位
        missing_slots = []
        for slot in intent.required_slots:
            if slot not in extracted_slots or not extracted_slots[slot]:
                missing_slots.append(slot)

        print(f"[IntentRouter] 关键词匹配: {intent.name}, 置信度: {confidence:.2f}")

        return IntentMatch(
            intent=intent,
            confidence=confidence,
            extracted_slots=extracted_slots,
            missing_slots=missing_slots,
            message=message,
            llm_inferred=False
        )

    def _extract_slots(self, intent: Intent, message: str) -> Dict[str, Any]:
        """从消息中提取槽位值"""
        slots = {}

        # 提取采购需求ID (PR-XXXXXX)
        pr_pattern = r'PR[_-]?\d{8}[_-]?\d{3,}'
        pr_matches = re.findall(pr_pattern, message, re.IGNORECASE)
        if pr_matches:
            slots['pr_id'] = pr_matches[0].upper().replace('-', '').replace('_', '')

        # 提取行项目
        pr_item_pattern = r'PR[_-]?\d{8}[_-]?\d{3,}[/-](\d{4})'
        pr_item_matches = re.findall(pr_item_pattern, message, re.IGNORECASE)
        if pr_item_matches:
            slots['pr_item'] = pr_item_matches[0]

        # 提取询价单号 (RFQ-XXXXXX / INQ-XXXXXX)
        inquiry_pattern = r'(?:RFQ|INQ)[_-]?\d{8}[_-]?\d{3,}'
        inquiry_matches = re.findall(inquiry_pattern, message, re.IGNORECASE)
        if inquiry_matches:
            slots['inquiry_id'] = inquiry_matches[0].upper().replace('-', '').replace('_', '')

        # 提取报价单号 (QUO-XXXXXX)
        quotation_pattern = r'QUO[_-]?\d{3,}'
        quotation_matches = re.findall(quotation_pattern, message, re.IGNORECASE)
        if quotation_matches:
            slots['quotation_id'] = quotation_matches[0].upper().replace('-', '').replace('_', '')

        # 提取行项目
        quotation_item_pattern = r'QUO[_-]?\d{3,}[/-](\d{4})'
        quotation_item_matches = re.findall(quotation_item_pattern, message, re.IGNORECASE)
        if quotation_item_matches:
            slots['quotation_item'] = quotation_item_matches[0]

        # 提取采购订单号 (PO-XXXXXX)
        po_pattern = r'PO[_-]?\d{8}[_-]?\d{3,}'
        po_matches = re.findall(po_pattern, message, re.IGNORECASE)
        if po_matches:
            slots['po_id'] = po_matches[0].upper().replace('-', '').replace('_', '')

        # 提取决策 (通过/批准 vs 驳回)
        if any(word in message for word in ['通过', '批准', '确认', '同意']):
            slots['decision'] = 'approve'
        elif any(word in message for word in ['驳回', '拒绝', '不同意']):
            slots['decision'] = 'reject'

        # 提取采购类型
        pr_type_match = re.search(r'NB|标准采购|紧急采购|生产采购', message)
        if pr_type_match:
            slots['pr_type'] = pr_type_match.group()

        # 提取部门
        dep_pattern = r'研发部|市场部|销售部|采购部|财务部|人事部|生产部|质量管理部'
        dep_match = re.search(dep_pattern, message)
        if dep_match:
            slots['apply_dep'] = dep_match.group()

        # 提取物料描述
        material_pattern = r'物料[：:]\s*([^，,。\n]+)'
        material_match = re.search(material_pattern, message)
        if material_match:
            slots['material_keyword'] = material_match.group(1).strip()

        return slots

    def execute(self, match: IntentMatch) -> ExecutionResult:
        """执行意图"""
        if not match.missing_slots:
            # 所有必需槽位都已提取,执行action
            connector_response = self.connector.call(
                match.intent.action,
                match.extracted_slots
            )

            response_text = self._format_response(match.intent, connector_response)

            return ExecutionResult(
                success=connector_response.success,
                intent=match.intent,
                connector_response=connector_response,
                response_text=response_text,
                error=connector_response.error
            )
        else:
            # 缺少必需槽位,返回槽位收集提示
            return ExecutionResult(
                success=False,
                intent=match.intent,
                error="missing_slots",
                response_text=self._generate_slot_prompt(match.intent, match.missing_slots)
            )

    def _format_response(self, intent: Intent, response: ConnectorResponse) -> str:
        """格式化响应文本"""
        if not response.success:
            return f"操作失败：{response.error}"

        data = response.data

        if intent.id == "procurement/query_open_pr":
            items = data.get('items', [])
            total = data.get('total', 0)
            system_count = len([i for i in items if i.get('claimer') != '手工'])
            manual_count = len(items) - system_count
            return f"好的，已为您检索到 {total} 条未执行的采购需求。其中{system_count}条来源于系统接口集成，{manual_count}条为手工创建。需要我为您展示详细列表吗？"

        elif intent.id == "procurement/create_inquiry":
            return f"已基于采购需求 {data.get('pr_id')} 自动生成询价单 {data.get('inquiry_id')}，关联采购需求：{data.get('pr_id')}，物料：{data.get('material_d')}，数量：{data.get('quantity')}。是否需要发布给供应商询价？"

        elif intent.id == "procurement/compare_quotations":
            recommended = data.get('recommended', {})
            history_avg = data.get('history_avg_price', 0)
            market_idx = data.get('market_index', 0)
            suggested_min = int(recommended.get('net_price', 0) * 0.96)
            suggested_max = int(recommended.get('net_price', 0) * 1.0)
            return f"已完成自动比价分析：\n推荐供应商：{recommended.get('vendor_d')}（{recommended.get('quotation_id')}）\n推荐理由：单价最低，综合成本最优\n参考依据：历史均价 {history_avg} 元、市场指数 {market_idx} 元\n议价建议区间：{suggested_min}–{suggested_max} 元\n是否确认选择 {recommended.get('vendor_d')}（{recommended.get('quotation_id')}）并提交审批？"

        elif intent.id == "procurement/approve_quotation":
            decision_text = "通过" if data.get('status') == '已审批' else "驳回"
            return f"报价单 {data.get('quotation_id')} 审批已{decision_text}，价格正式生效，可用于创建采购订单。"

        elif intent.id == "procurement/create_purchase_order":
            item = data.get('item', {})
            return f"采购订单已创建：\n订单号：{data.get('po_id')}\n关联计划：{item.get('pr_id')}\n关联报价：{item.get('quotation_id')}\n供应商：{data.get('vendor_d')}｜物料：{item.get('material_d')}｜数量：{item.get('quantity')}｜单价：{item.get('net_price')}\n状态：{data.get('status')}\n是否需要提交审批？"

        elif intent.id == "procurement/query_order_status":
            details = data.get('status_details', {})
            summary = data.get('summary', {})
            return f"采购订单 {data.get('po_id')} 执行状态如下：\n1. 审批状态：{details.get('审批状态', '未知')}\n2. 发货状态：{details.get('发货状态', '未知')}\n3. 收货状态：{details.get('收货状态', '未知')}\n4. 发票状态：{details.get('发票状态', '未知')}\n收货进度：{summary.get('receipt_progress', 'N/A')}\n后续状态变更将自动为您推送提醒。"

        elif intent.id == "procurement/approve_pr":
            decision_text = "通过" if data.get('flow_status') == 'APPROVED' else "驳回"
            return f"采购需求 {data.get('pr_id')}/{data.get('pr_item')} 已{decision_text}。"

        else:
            # 默认响应
            return response.message or f"操作成功"

    def _generate_slot_prompt(self, intent: Intent, missing_slots: List[str]) -> str:
        """生成槽位收集提示"""
        prompts = {
            'pr_id': '请提供采购需求编号（如：PR-20260528-001）',
            'pr_item': '请提供行项目号',
            'quotation_id': '请提供报价单号（如：QUO-001）',
            'quotation_item': '请提供报价单行项目号',
            'po_id': '请提供采购订单号（如：PO-20260528-001）',
            'inquiry_id': '请提供询价单号（如：RFQ-20260528-001）',
            'decision': '请说明您的决定（通过/批准 或 驳回）',
        }

        slot_prompts = [prompts.get(s, s) for s in missing_slots]

        if len(slot_prompts) == 1:
            return f"好的，需要您提供：{slot_prompts[0]}"
        else:
            return f"好的，需要您提供以下信息：\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(slot_prompts))

    def handle_message(self, message: str) -> str:
        """处理用户消息的完整流程"""
        match = self.route(message)

        if not match:
            return "抱歉，我暂时无法理解您的意图。您可以：\n1. 查询采购需求\n2. 创建询价单\n3. 比价分析\n4. 创建采购订单\n5. 查询订单执行情况\n请告诉我您需要什么帮助？"

        result = self.execute(match)
        return result.response_text


# 全局路由器实例
_router_instance: Optional[IntentRouter] = None


def get_intent_router() -> IntentRouter:
    """获取意图路由器单例"""
    global _router_instance
    if _router_instance is None:
        _router_instance = IntentRouter()
    return _router_instance
