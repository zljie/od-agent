"""
采购场景意图定义
基于 procurement_s2a_semantic_model.yaml 本体定义的意图
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Intent:
    id: str
    name: str
    description: str
    trigger_keywords: List[str]
    required_slots: List[str]
    optional_slots: List[str]
    action: str  # 本体定义的 action ID
    filters: Optional[Dict[str, Any]] = None
    examples: Optional[List[str]] = None
    response_template: Optional[str] = None


PROCUREMENT_INTENTS = [
    Intent(
        id="procurement/query_unexecuted_pr",
        name="查询未执行采购需求",
        description="识别已审批但尚未生成采购订单的采购需求行项目",
        trigger_keywords=[
            "采购需求", "未执行", "待处理", "待审批", "待执行",
            "采购计划", "未完成的采购", "待处理的需求", "哪些采购需求没有执行"
        ],
        required_slots=[],
        optional_slots=["company_id", "factory_id", "apply_dep", "as_of_date"],
        action="analytics/find_unexecuted_purchase_requests",
        examples=[
            "帮我查一下目前还有哪些采购需求没有执行",
            "查一下所有未执行的采购计划",
            "看看有哪些待审批的采购需求"
        ],
        response_template="查询到 {total} 条未执行的采购需求。其中系统接口集成 {system_count} 条，手工创建 {manual_count} 条。需要展示详细列表吗？"
    ),

    Intent(
        id="procurement/create_inquiry",
        name="根据采购需求创建询价单",
        description="基于采购需求行项目创建询价单，并继承物料、数量、交期等信息",
        trigger_keywords=[
            "询价", "发起询价", "创建询价单", "生成询价单",
            "向供应商询价", "发起询价流程", "根据采购需求询价"
        ],
        required_slots=["pr_id"],
        optional_slots=["pr_item", "supplier_scope", "deadline"],
        action="procurement/create_inquiry_from_pr",
        examples=[
            "先处理采购类型为NB的采购需求，创建询价单",
            "帮我创建一个询价单",
            "基于采购需求创建询价单"
        ],
        response_template="已基于采购需求 {pr_id} 创建询价单 {inquiry_id}，物料：{material_d}，数量：{quantity}。是否需要发布给供应商询价？"
    ),

    Intent(
        id="rfq/collect_quotations",
        name="回收报价",
        description="汇总某个询价单下供应商报价进度和报价明细",
        trigger_keywords=[
            "报价回收", "报价跟踪", "询价单报价情况", 
            "有哪些供应商报价", "报价进度", "回收报价"
        ],
        required_slots=["inquiry_id"],
        optional_slots=[],
        action="rfq/collect_quotations",
        examples=[
            "询价单现在有几家供应商报价了",
            "查看询价单报价情况",
            "回收报价"
        ]
    ),

    Intent(
        id="analytics/generate_price_comparison",
        name="生成比价分析",
        description="按同一询价单和物料维度，对多家供应商报价进行比价并给出推荐",
        trigger_keywords=[
            "比价", "开始比价", "推荐供应商", "比较报价",
            "哪家便宜", "最优价格", "价格分析", "自动比价"
        ],
        required_slots=["inquiry_id"],
        optional_slots=["material_id", "min_supplier_count"],
        action="analytics/generate_price_comparison",
        examples=[
            "开始比价",
            "帮我比较一下这几家供应商的报价",
            "哪家供应商的价格最优"
        ],
        response_template="已完成比价分析：\n推荐供应商：{vendor_d}（{quotation_id}）\n推荐理由：{reason}\n参与报价供应商：{supplier_count} 家\n是否确认选择该供应商并提交审批？"
    ),

    Intent(
        id="procurement/submit_award_approval",
        name="提交中标报价审批",
        description="将推荐报价提交审批流程，审批通过后价格生效",
        trigger_keywords=[
            "提交审批", "中标审批", "定标审批", "提交中标",
            "报价审批", "确认报价"
        ],
        required_slots=["quotation_id", "quotation_item"],
        optional_slots=["reason"],
        action="procurement/submit_award_approval",
        examples=[
            "确认报价单QUO-002并提交审批",
            "批准这个报价",
            "提交中标审批"
        ],
        response_template="报价单 {quotation_id} 已提交审批，审批流程ID: {workflow_id}。"
    ),

    Intent(
        id="procurement/create_purchase_order",
        name="创建采购订单",
        description="基于审批通过的采购需求和有效报价创建采购订单",
        trigger_keywords=[
            "生成订单", "创建采购订单", "下达采购订单",
            "创建PO", "生成采购单", "生成订单", "根据报价创建PO"
        ],
        required_slots=["pr_id", "pr_item", "quotation_id", "quotation_item"],
        optional_slots=["submit_approval"],
        action="procurement/create_purchase_order_from_pr_and_quotation",
        examples=[
            "根据采购需求生成采购订单",
            "帮我创建PO",
            "基于审批通过的报价创建采购订单"
        ],
        response_template="采购订单已创建：\n订单号：{po_id}\n供应商：{vendor_d}｜物料：{material_d}｜数量：{quantity}｜单价：{unit_price}\n状态：{status}\n是否需要提交审批？"
    ),

    Intent(
        id="analytics/get_purchase_order_execution_status",
        name="查询采购订单执行情况",
        description="查询采购订单审批、发货、收货、发票等执行状态",
        trigger_keywords=[
            "执行情况", "订单状态", "收货进度", "订单进度",
            "发货状态", "入库情况", "PO执行", "采购进度"
        ],
        required_slots=["po_id"],
        optional_slots=[],
        action="analytics/get_purchase_order_execution_status",
        examples=[
            "帮我查一下采购订单的执行情况",
            "看看这个订单的收货进度",
            "订单执行到哪里了"
        ],
        response_template="采购订单 {po_id} 执行状态如下：\n1. 审批状态：{approval_status}\n2. 发货状态：{shipment_status}\n3. 收货状态：{receipt_status}\n4. 发票状态：{invoice_status}\n后续状态变更将自动为您推送提醒。"
    ),

    Intent(
        id="procurement/approve_pr",
        name="审批采购需求",
        description="审批采购需求的通过或驳回",
        trigger_keywords=[
            "审批采购需求", "批准采购", "驳回采购",
            "通过需求", "需求审批"
        ],
        required_slots=["pr_id"],
        optional_slots=["pr_item", "decision", "comment"],
        action="purchase_request/approve",
        examples=[
            "审批采购需求",
            "批准这个采购需求",
            "通过采购申请"
        ],
        response_template="采购需求 {pr_id}/{pr_item} 已{decision_text}。"
    ),

    Intent(
        id="procurement/query_inquiry",
        name="查询询价单",
        description="查询询价单列表或详情",
        trigger_keywords=[
            "询价单", "RFQ", "询价进度", "询价状态"
        ],
        required_slots=[],
        optional_slots=["inquiry_id"],
        action="purchase_inquiries/list",
        examples=[
            "查一下询价单列表",
            "看看有哪些询价单",
            "RFQ的状态"
        ]
    ),

    Intent(
        id="procurement/query_quotation",
        name="查询报价单",
        description="查询报价单列表或详情",
        trigger_keywords=[
            "报价单", "报价", "供应商报价", "QUO"
        ],
        required_slots=[],
        optional_slots=["quotation_id", "inquiry_id", "vendor_id"],
        action="purchase_quotations/list",
        examples=[
            "查一下报价单列表",
            "看看有哪些报价",
            "QUO的详细信息"
        ]
    ),

    Intent(
        id="procurement/query_purchase_order",
        name="查询采购订单",
        description="查询采购订单抬头列表或详情",
        trigger_keywords=[
            "采购订单", "PO", "订单列表"
        ],
        required_slots=[],
        optional_slots=["po_id"],
        action="purchase_order_heads/list",
        examples=[
            "查一下采购订单列表",
            "看看有哪些订单"
        ]
    ),
]


def get_intent_by_id(intent_id: str) -> Optional[Intent]:
    """根据ID获取意图"""
    for intent in PROCUREMENT_INTENTS:
        if intent.id == intent_id:
            return intent
    return None


def get_intents_by_action(action: str) -> List[Intent]:
    """根据action获取相关意图"""
    return [i for i in PROCUREMENT_INTENTS if i.action == action]


def get_all_trigger_keywords() -> Dict[str, List[str]]:
    """获取所有触发关键词及其对应的意图ID"""
    result = {}
    for intent in PROCUREMENT_INTENTS:
        for keyword in intent.trigger_keywords:
            if keyword not in result:
                result[keyword] = []
            result[keyword].append(intent.id)
    return result


class ProcurementIntentRouter:
    """采购意图路由器 - 基于关键词匹配"""

    def __init__(self):
        self.intents = PROCUREMENT_INTENTS
        self._build_keyword_index()

    def _build_keyword_index(self):
        """构建关键词索引"""
        self.keyword_index: Dict[str, List[tuple]] = {}
        for intent in self.intents:
            for keyword in intent.trigger_keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append((intent.id, len(keyword)))

    def route(self, message: str) -> Optional[Intent]:
        """路由用户消息到对应意图"""
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
        return get_intent_by_id(top_intent_id)

    def extract_slots(self, intent: Intent, message: str) -> Dict[str, Any]:
        """从消息中提取槽位值"""
        slots = {}

        # 提取采购需求ID (PR-XXXXXX)
        import re
        pr_pattern = r'PR[_-]?\d{8}[_-]?\d{3,}'
        pr_matches = re.findall(pr_pattern, message, re.IGNORECASE)
        if pr_matches:
            slots['pr_id'] = pr_matches[0].upper().replace('-', '').replace('_', '')

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

        # 提取采购订单号 (PO-XXXXXX)
        po_pattern = r'PO[_-]?\d{8}[_-]?\d{3,}'
        po_matches = re.findall(po_pattern, message, re.IGNORECASE)
        if po_matches:
            slots['po_id'] = po_matches[0].upper().replace('-', '').replace('_', '')

        # 提取决策 (通过/批准 vs 驳回)
        if '通过' in message or '批准' in message or '确认' in message:
            slots['decision'] = 'approve'
        elif '驳回' in message or '拒绝' in message:
            slots['decision'] = 'reject'

        # 提取采购类型
        pr_type_match = re.search(r'NB|标准采购|紧急采购', message)
        if pr_type_match:
            slots['pr_type'] = pr_type_match.group()

        # 提取部门
        dep_pattern = r'研发部|市场部|销售部|采购部|财务部|人事部'
        dep_match = re.search(dep_pattern, message)
        if dep_match:
            slots['apply_dep'] = dep_match.group()

        return slots
