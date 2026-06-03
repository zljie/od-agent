"""用户确认机制模块 - 处理中高风险动作的确认流程

本模块实现文档定义的确认态管理：
- 识别需要确认的动作（创建询价单、发布、提交审批、删除、撤回、创建采购订单）
- 生成确认提示信息
- 处理用户确认/拒绝响应
- 状态持久化以支持多轮对话
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class ConfirmationAction(str, Enum):
    """需要确认的动作类型"""
    QUERY = "query"                    # 查询操作，通常不需要确认
    CREATE_INQUIRY = "create_inquiry"    # 创建询价单
    PUBLISH_INQUIRY = "publish_inquiry"   # 发布询价单
    SUBMIT_APPROVAL = "submit_approval"   # 提交审批
    APPROVE = "approve"                # 审批通过
    REJECT = "reject"                  # 审批驳回
    DELETE = "delete"                  # 删除操作
    WITHDRAW = "withdraw"              # 撤回操作
    CREATE_PO = "create_po"           # 创建采购订单
    OTHER = "other"                   # 其他操作


class ConfirmationRisk(str, Enum):
    """风险等级"""
    LOW = "low"        # 低风险，如查询
    MEDIUM = "medium"  # 中风险，如创建
    HIGH = "high"     # 高风险，如删除、撤回


@dataclass
class ConfirmationContext:
    """确认上下文 - 记录需要用户确认的操作详情"""
    action: str                              # 动作ID
    action_type: ConfirmationAction          # 动作类型
    risk_level: ConfirmationRisk             # 风险等级
    description: str                          # 操作描述
    details: Dict[str, Any] = field(default_factory=dict)  # 操作详情
    affected_records: List[Dict] = field(default_factory=list)  # 影响的记录
    rules_triggered: List[str] = field(default_factory=list)  # 触发的规则
    warning_messages: List[str] = field(default_factory=list)  # 警告信息
    timestamp: str = ""                       # 确认请求时间
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ConfirmationResponse:
    """用户确认响应"""
    confirmed: bool              # 用户是否确认执行
    action: str                  # 对应的动作ID
    user_input: str              # 用户的原始输入
    timestamp: str = ""          # 响应时间
    modified_params: Optional[Dict[str, Any]] = None  # 用户修改的参数
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ConfirmationPolicy:
    """确认策略 - 定义哪些动作需要确认"""
    
    # 动作与风险等级映射
    RISK_MAPPING: Dict[str, ConfirmationRisk] = {
        "purchase_requests/list": ConfirmationRisk.LOW,
        "purchase_requests/get_by_id": ConfirmationRisk.LOW,
        "purchase_inquiries/list": ConfirmationRisk.LOW,
        "purchase_inquiries/get_by_id": ConfirmationRisk.LOW,
        "purchase_quotations/list": ConfirmationRisk.LOW,
        "purchase_quotations/get_by_id": ConfirmationRisk.LOW,
        "purchase_order_heads/list": ConfirmationRisk.LOW,
        "purchase_order_heads/get_by_id": ConfirmationRisk.LOW,
        "analytics/find_unexecuted_purchase_requests": ConfirmationRisk.LOW,
        "analytics/generate_price_comparison": ConfirmationRisk.LOW,
        "analytics/get_purchase_order_execution_status": ConfirmationRisk.LOW,
        "rfq/collect_quotations": ConfirmationRisk.LOW,
        # 中风险
        "procurement/create_inquiry_from_pr": ConfirmationRisk.MEDIUM,
        "procurement/submit_award_approval": ConfirmationRisk.MEDIUM,
        "approval/submit_for_approval": ConfirmationRisk.MEDIUM,
        "approval/approve_workflow": ConfirmationRisk.MEDIUM,
        "inquiry/publish_inquiry": ConfirmationRisk.MEDIUM,
        "inquiry/close_inquiry": ConfirmationRisk.MEDIUM,
        # 高风险
        "procurement/create_purchase_order_from_pr_and_quotation": ConfirmationRisk.HIGH,
        "approval/reject_workflow": ConfirmationRisk.HIGH,
        "approval/withdraw_workflow": ConfirmationRisk.HIGH,
        "inquiry/withdraw_inquiry": ConfirmationRisk.HIGH,
    }
    
    # 需要确认的kind类型
    COMMAND_KIND_CONFIRM: bool = True  # 所有command类型都需要确认
    DELETE_OPERATIONS: List[str] = ["delete", "remove"]  # 删除操作
    
    @classmethod
    def get_risk_level(cls, action_id: str, operation: str = "") -> ConfirmationRisk:
        """获取动作的风险等级"""
        # 直接查找
        if action_id in cls.RISK_MAPPING:
            return cls.RISK_MAPPING[action_id]
        
        # 检查是否包含删除操作
        if operation in cls.DELETE_OPERATIONS:
            return ConfirmationRisk.HIGH
        
        # 检查是否包含command
        if "command" in action_id.lower() or operation in ["create", "update"]:
            return ConfirmationRisk.MEDIUM
        
        return ConfirmationRisk.LOW
    
    @classmethod
    def needs_confirmation(cls, action_id: str, kind: str = "") -> bool:
        """判断动作是否需要确认"""
        # 直接检查风险映射
        risk = cls.get_risk_level(action_id)
        if risk != ConfirmationRisk.LOW:
            return True
        
        # 检查kind类型
        if cls.COMMAND_KIND_CONFIRM and kind == "command":
            return True
        
        return False


class ConfirmationManager:
    """确认管理器 - 管理会话中的确认状态"""
    
    def __init__(self):
        self._pending_confirmations: Dict[str, ConfirmationContext] = {}  # 待确认的上下文
        self._confirmation_history: List[ConfirmationResponse] = []  # 确认历史
    
    def create_confirmation(
        self,
        session_id: str,
        action: str,
        action_type: ConfirmationAction,
        risk_level: ConfirmationRisk,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        affected_records: Optional[List[Dict]] = None,
        rules_triggered: Optional[List[str]] = None,
        warning_messages: Optional[List[str]] = None
    ) -> ConfirmationContext:
        """创建确认上下文"""
        context = ConfirmationContext(
            action=action,
            action_type=action_type,
            risk_level=risk_level,
            description=description,
            details=details or {},
            affected_records=affected_records or [],
            rules_triggered=rules_triggered or [],
            warning_messages=warning_messages or []
        )
        self._pending_confirmations[session_id] = context
        return context
    
    def get_pending_confirmation(self, session_id: str) -> Optional[ConfirmationContext]:
        """获取待确认的上下文"""
        return self._pending_confirmations.get(session_id)
    
    def process_confirmation(
        self,
        session_id: str,
        user_input: str,
        confirmed: bool,
        modified_params: Optional[Dict[str, Any]] = None
    ) -> Optional[ConfirmationResponse]:
        """处理用户确认响应"""
        context = self._pending_confirmations.pop(session_id, None)
        if not context:
            return None
        
        response = ConfirmationResponse(
            confirmed=confirmed,
            action=context.action,
            user_input=user_input,
            modified_params=modified_params
        )
        self._confirmation_history.append(response)
        return response
    
    def clear_confirmation(self, session_id: str) -> bool:
        """清除待确认状态"""
        if session_id in self._pending_confirmations:
            del self._pending_confirmations[session_id]
            return True
        return False
    
    def get_confirmation_history(self, session_id: str = "") -> List[ConfirmationResponse]:
        """获取确认历史"""
        if session_id:
            return [r for r in self._confirmation_history if r.action.startswith(session_id)]
        return self._confirmation_history


class ConfirmationMessageBuilder:
    """确认消息构建器 - 生成用户友好的确认提示"""
    
    @staticmethod
    def build_confirmation_message(context: ConfirmationContext) -> str:
        """构建确认提示消息"""
        parts = []
        
        # 风险等级提示
        risk_emoji = {
            ConfirmationRisk.LOW: "📋",
            ConfirmationRisk.MEDIUM: "⚠️",
            ConfirmationRisk.HIGH: "🔴"
        }
        risk_text = {
            ConfirmationRisk.LOW: "低风险操作",
            ConfirmationRisk.MEDIUM: "中风险操作",
            ConfirmationRisk.HIGH: "高风险操作"
        }
        
        parts.append(f"{risk_emoji.get(context.risk_level, '📋')} {risk_text.get(context.risk_level, '')}")
        parts.append("")
        
        # 操作描述
        parts.append(f"**操作**: {context.description}")
        parts.append("")
        
        # 操作详情
        if context.details:
            parts.append("**操作详情**:")
            for key, value in context.details.items():
                if value is not None:
                    parts.append(f"  - {key}: {value}")
            parts.append("")
        
        # 影响的记录
        if context.affected_records:
            count = len(context.affected_records)
            parts.append(f"**将影响 {count} 条记录**")
            # 只显示前3条
            for record in context.affected_records[:3]:
                record_desc = " / ".join(f"{k}={v}" for k, v in list(record.items())[:3])
                parts.append(f"  - {record_desc}")
            if count > 3:
                parts.append(f"  - ... 还有 {count - 3} 条")
            parts.append("")
        
        # 触发的规则
        if context.rules_triggered:
            parts.append("**触发的规则**:")
            for rule in context.rules_triggered:
                parts.append(f"  - {rule}")
            parts.append("")
        
        # 警告信息
        if context.warning_messages:
            parts.append("**⚠️ 注意事项**:")
            for warning in context.warning_messages:
                parts.append(f"  - {warning}")
            parts.append("")
        
        # 确认提示
        if context.risk_level == ConfirmationRisk.HIGH:
            parts.append("**此操作风险较高，请确认是否继续？**")
        elif context.risk_level == ConfirmationRisk.MEDIUM:
            parts.append("**请确认是否执行此操作？**")
        else:
            parts.append("**请确认**")
        
        parts.append("")
        parts.append("回复 \"确认\" 或 \"是\" 执行，回复 \"取消\" 或 \"否\" 放弃。")
        
        return "\n".join(parts)
    
    @staticmethod
    def build_success_message(action: str, result: Dict[str, Any]) -> str:
        """构建操作成功消息"""
        parts = ["✅ **操作已执行成功**"]
        
        if "inquiry_id" in result:
            parts.append(f"询价单号: {result['inquiry_id']}")
        if "po_id" in result:
            parts.append(f"采购订单号: {result['po_id']}")
        if "workflow_id" in result:
            parts.append(f"审批流程号: {result['workflow_id']}")
        if "message" in result:
            parts.append(result["message"])
        
        return "\n".join(parts)
    
    @staticmethod
    def build_cancelled_message(action: str) -> str:
        """构建操作取消消息"""
        return "❌ **操作已取消**\n\n您可以继续其他操作。"
    
    @staticmethod
    def build_rule_blocked_message(
        rule_name: str,
        rule_message: str,
        current_status: Dict[str, Any],
        remediation: str
    ) -> str:
        """构建规则阻断消息"""
        parts = [
            "🚫 **操作无法执行**",
            "",
            "**命中的规则**:",
            f"- {rule_name}",
            "",
            "**规则说明**:",
            f"- {rule_message}",
            "",
            "**当前状态**:"
        ]
        
        for key, value in current_status.items():
            parts.append(f"- {key}: {value}")
        
        parts.extend([
            "",
            "**建议处理**:",
            f"- {remediation}"
        ])
        
        return "\n".join(parts)


# 全局确认管理器实例
_confirmation_manager: Optional[ConfirmationManager] = None


def get_confirmation_manager() -> ConfirmationManager:
    """获取确认管理器单例"""
    global _confirmation_manager
    if _confirmation_manager is None:
        _confirmation_manager = ConfirmationManager()
    return _confirmation_manager


def is_confirmation_input(user_input: str) -> bool:
    """判断用户输入是否为确认响应"""
    confirm_keywords = ["确认", "是", "好的", "行", "执行", "确定", "yes", "y", "confirm"]
    cancel_keywords = ["取消", "否", "不", "算了", "不用", "no", "n", "cancel"]
    
    user_input_lower = user_input.lower().strip()
    
    for keyword in confirm_keywords:
        if keyword in user_input_lower:
            return True
    
    for keyword in cancel_keywords:
        if keyword in user_input_lower:
            return False
    
    return False
