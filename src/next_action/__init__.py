"""下一步动作推荐引擎 - Next Action Engine

本模块实现文档定义的下一步动作推荐功能：
- 查询采购需求后: 展示明细/按部门汇总/生成询价单/导出清单
- 创建询价单后: 发布供应商/修改询价单/删除未发布询价单
- 报价回收后: 开始比价/查看报价明细/催办未报价供应商
- 比价完成后: 提交审批/发起议价/查看历史价格依据
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


class ActionCategory(str, Enum):
    """动作类别"""
    QUERY = "query"           # 查询类
    CREATE = "create"         # 创建类
    UPDATE = "update"         # 更新类
    DELETE = "delete"         # 删除类
    APPROVAL = "approval"     # 审批类
    PUBLISH = "publish"       # 发布类
    WITHDRAW = "withdraw"     # 撤回类
    COMPARE = "compare"       # 比价类
    EXPORT = "export"         # 导出类
    NAVIGATE = "navigate"     # 导航类
    REMIND = "remind"         # 提醒类
    BARGAIN = "bargain"       # 议价类


class ActionPriority(str, Enum):
    """动作优先级"""
    HIGH = "high"      # 高优先级
    MEDIUM = "medium"  # 中优先级
    LOW = "low"       # 低优先级


@dataclass
class SuggestedAction:
    """建议动作"""
    id: str                         # 动作ID
    label: str                      # 显示标签
    description: str                # 动作描述
    category: ActionCategory        # 动作类别
    priority: ActionPriority        # 优先级
    enabled: bool = True            # 是否启用
    icon: str = ""                 # 图标
    confirm_required: bool = False # 是否需要确认
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class ActionContext:
    """动作上下文 - 用于生成下一步建议"""
    current_intent: str                         # 当前意图
    current_action: str                         # 当前动作
    current_state: str                           # 当前状态
    previous_actions: List[str] = field(default_factory=list)  # 历史动作
    result_data: Optional[Dict[str, Any]] = None  # 结果数据
    session_context: Dict[str, Any] = field(default_factory=dict)  # 会话上下文


class NextActionEngine:
    """下一步动作推荐引擎"""
    
    def __init__(self):
        self._action_templates: Dict[str, List[SuggestedAction]] = {}
        self._context_hooks: List[Callable[[ActionContext], List[SuggestedAction]]] = []
        self._initialize_default_actions()
    
    def _initialize_default_actions(self):
        """初始化默认动作模板"""
        
        # 查询采购需求后的下一步
        self._action_templates["procurement/query_unexecuted_pr"] = [
            SuggestedAction(
                id="show_detail",
                label="展示明细",
                description="查看采购需求详细列表",
                category=ActionCategory.NAVIGATE,
                priority=ActionPriority.HIGH,
                icon="📋"
            ),
            SuggestedAction(
                id="group_by_dept",
                label="按部门汇总",
                description="按申请部门统计采购需求",
                category=ActionCategory.QUERY,
                priority=ActionPriority.MEDIUM,
                icon="📊"
            ),
            SuggestedAction(
                id="create_inquiry",
                label="生成询价单",
                description="基于采购需求创建询价单",
                category=ActionCategory.CREATE,
                priority=ActionPriority.HIGH,
                icon="📝",
                confirm_required=True
            ),
            SuggestedAction(
                id="export_list",
                label="导出清单",
                description="导出采购需求清单",
                category=ActionCategory.EXPORT,
                priority=ActionPriority.LOW,
                icon="📥"
            ),
        ]
        
        # 查询采购需求列表后的下一步
        self._action_templates["procurement/query_with_conditions"] = [
            SuggestedAction(
                id="show_detail",
                label="展示明细",
                description="查看采购需求详细列表",
                category=ActionCategory.NAVIGATE,
                priority=ActionPriority.HIGH,
                icon="📋"
            ),
            SuggestedAction(
                id="group_by_dept",
                label="按部门汇总",
                description="按申请部门统计采购需求",
                category=ActionCategory.QUERY,
                priority=ActionPriority.MEDIUM,
                icon="📊"
            ),
            SuggestedAction(
                id="group_by_type",
                label="按类型汇总",
                description="按采购类型统计采购需求",
                category=ActionCategory.QUERY,
                priority=ActionPriority.MEDIUM,
                icon="📊"
            ),
            SuggestedAction(
                id="create_inquiry",
                label="生成询价单",
                description="基于采购需求创建询价单",
                category=ActionCategory.CREATE,
                priority=ActionPriority.HIGH,
                icon="📝",
                confirm_required=True
            ),
            SuggestedAction(
                id="export_list",
                label="导出清单",
                description="导出采购需求清单",
                category=ActionCategory.EXPORT,
                priority=ActionPriority.LOW,
                icon="📥"
            ),
        ]
        
        # 创建询价单后的下一步
        self._action_templates["procurement/create_inquiry"] = [
            SuggestedAction(
                id="publish_inquiry",
                label="发布供应商",
                description="将询价单发布给供应商",
                category=ActionCategory.PUBLISH,
                priority=ActionPriority.HIGH,
                icon="📤",
                confirm_required=True
            ),
            SuggestedAction(
                id="modify_inquiry",
                label="修改询价单",
                description="修改询价单内容",
                category=ActionCategory.UPDATE,
                priority=ActionPriority.MEDIUM,
                icon="✏️"
            ),
            SuggestedAction(
                id="delete_inquiry",
                label="删除询价单",
                description="删除未发布的询价单",
                category=ActionCategory.DELETE,
                priority=ActionPriority.LOW,
                icon="🗑️",
                confirm_required=True
            ),
        ]
        
        # 询价单发布后的下一步
        self._action_templates["inquiry/published"] = [
            SuggestedAction(
                id="collect_quotations",
                label="回收报价",
                description="查看供应商报价进度",
                category=ActionCategory.QUERY,
                priority=ActionPriority.HIGH,
                icon="📥"
            ),
            SuggestedAction(
                id="remind_supplier",
                label="催办供应商",
                description="向未报价供应商发送提醒",
                category=ActionCategory.REMIND,
                priority=ActionPriority.MEDIUM,
                icon="⏰"
            ),
            SuggestedAction(
                id="view_inquiry_detail",
                label="查看询价单",
                description="查看询价单详细信息",
                category=ActionCategory.NAVIGATE,
                priority=ActionPriority.LOW,
                icon="👁️"
            ),
        ]
        
        # 报价回收后的下一步
        self._action_templates["rfq/collect_quotations"] = [
            SuggestedAction(
                id="start_comparison",
                label="开始比价",
                description="对供应商报价进行比价分析",
                category=ActionCategory.COMPARE,
                priority=ActionPriority.HIGH,
                icon="⚖️"
            ),
            SuggestedAction(
                id="view_quotation_detail",
                label="查看报价明细",
                description="查看各供应商报价详情",
                category=ActionCategory.NAVIGATE,
                priority=ActionPriority.MEDIUM,
                icon="📄"
            ),
            SuggestedAction(
                id="remind_pending",
                label="催办未报价",
                description="向尚未报价的供应商发送提醒",
                category=ActionCategory.REMIND,
                priority=ActionPriority.MEDIUM,
                icon="⏰"
            ),
        ]
        
        # 比价完成后的下一步
        self._action_templates["analytics/generate_price_comparison"] = [
            SuggestedAction(
                id="submit_approval",
                label="提交审批",
                description="将推荐报价提交审批流程",
                category=ActionCategory.APPROVAL,
                priority=ActionPriority.HIGH,
                icon="✅",
                confirm_required=True
            ),
            SuggestedAction(
                id="start_bargain",
                label="发起议价",
                description="与推荐供应商议价",
                category=ActionCategory.BARGAIN,
                priority=ActionPriority.MEDIUM,
                icon="💬"
            ),
            SuggestedAction(
                id="view_history",
                label="查看历史价格",
                description="查看该物料的历史成交价格",
                category=ActionCategory.QUERY,
                priority=ActionPriority.LOW,
                icon="📜"
            ),
            SuggestedAction(
                id="view_comparison_detail",
                label="查看比价详情",
                description="查看完整的比价分析报告",
                category=ActionCategory.NAVIGATE,
                priority=ActionPriority.MEDIUM,
                icon="📊"
            ),
        ]
        
        # 提交审批后的下一步
        self._action_templates["procurement/submit_award_approval"] = [
            SuggestedAction(
                id="check_approval_status",
                label="查看审批状态",
                description="查看审批流程进度",
                category=ActionCategory.QUERY,
                priority=ActionPriority.HIGH,
                icon="🔍"
            ),
            SuggestedAction(
                id="withdraw_approval",
                label="撤回审批",
                description="撤回待审批的申请",
                category=ActionCategory.WITHDRAW,
                priority=ActionPriority.LOW,
                icon="↩️",
                confirm_required=True
            ),
        ]
        
        # 创建采购订单后的下一步
        self._action_templates["procurement/create_purchase_order"] = [
            SuggestedAction(
                id="track_receipt",
                label="跟踪收货",
                description="跟踪采购订单收货进度",
                category=ActionCategory.NAVIGATE,
                priority=ActionPriority.HIGH,
                icon="📦"
            ),
            SuggestedAction(
                id="check_order_status",
                label="查看订单状态",
                description="查看采购订单详细信息",
                category=ActionCategory.QUERY,
                priority=ActionPriority.MEDIUM,
                icon="📋"
            ),
        ]
        
        # 查询订单执行后的下一步
        self._action_templates["analytics/get_purchase_order_execution_status"] = [
            SuggestedAction(
                id="track_receipt",
                label="跟踪收货进度",
                description="持续跟踪订单收货状态",
                category=ActionCategory.NAVIGATE,
                priority=ActionPriority.HIGH,
                icon="📦"
            ),
            SuggestedAction(
                id="view_receipt_history",
                label="查看收货历史",
                description="查看订单收货记录",
                category=ActionCategory.QUERY,
                priority=ActionPriority.MEDIUM,
                icon="📋"
            ),
            SuggestedAction(
                id="export_receipt",
                label="导出收货单",
                description="导出收货记录",
                category=ActionCategory.EXPORT,
                priority=ActionPriority.LOW,
                icon="📥"
            ),
        ]
        
        # 审批采购需求后的下一步
        self._action_templates["procurement/approve_pr"] = [
            SuggestedAction(
                id="create_inquiry_from_approved",
                label="创建询价单",
                description="基于审批通过的采购需求创建询价单",
                category=ActionCategory.CREATE,
                priority=ActionPriority.HIGH,
                icon="📝",
                confirm_required=True
            ),
            SuggestedAction(
                id="view_approved_prs",
                label="查看已审批列表",
                description="查看所有已审批通过的采购需求",
                category=ActionCategory.QUERY,
                priority=ActionPriority.MEDIUM,
                icon="📋"
            ),
        ]
        
        # 通用下一步动作
        self._action_templates["__generic__"] = [
            SuggestedAction(
                id="new_query",
                label="继续查询",
                description="进行新的采购查询",
                category=ActionCategory.QUERY,
                priority=ActionPriority.MEDIUM,
                icon="🔍"
            ),
            SuggestedAction(
                id="get_help",
                label="获取帮助",
                description="获取采购助手使用帮助",
                category=ActionCategory.NAVIGATE,
                priority=ActionPriority.LOW,
                icon="❓"
            ),
        ]
    
    def register_context_hook(
        self, 
        hook: Callable[[ActionContext], List[SuggestedAction]]
    ) -> None:
        """注册上下文钩子，用于动态生成动作"""
        self._context_hooks.append(hook)
    
    def get_next_actions(
        self,
        context: ActionContext,
        max_actions: int = 4
    ) -> List[SuggestedAction]:
        """获取下一步动作建议
        
        Args:
            context: 动作上下文
            max_actions: 最大返回动作数
            
        Returns:
            建议动作列表，按优先级排序
        """
        actions: List[SuggestedAction] = []
        
        # 1. 首先尝试使用当前意图对应的模板
        if context.current_intent in self._action_templates:
            actions.extend(self._action_templates[context.current_intent])
        
        # 2. 如果没有找到，尝试当前动作对应的模板
        if not actions and context.current_action in self._action_templates:
            actions.extend(self._action_templates[context.current_action])
        
        # 3. 如果还是没有，尝试基于当前状态
        if not actions and context.current_state in self._action_templates:
            actions.extend(self._action_templates[context.current_state])
        
        # 4. 使用通用动作
        if not actions:
            actions.extend(self._action_templates.get("__generic__", []))
        
        # 5. 调用注册的钩子进行上下文调整
        for hook in self._context_hooks:
            try:
                hook_actions = hook(context)
                # 合并钩子返回的动作
                existing_ids = {a.id for a in actions}
                for action in hook_actions:
                    if action.id not in existing_ids:
                        actions.append(action)
            except Exception:
                pass
        
        # 6. 根据上下文过滤和调整动作
        actions = self._filter_actions(context, actions)
        
        # 7. 按优先级排序
        actions = self._sort_by_priority(actions)
        
        # 8. 限制数量
        return actions[:max_actions]
    
    def _filter_actions(
        self,
        context: ActionContext,
        actions: List[SuggestedAction]
    ) -> List[SuggestedAction]:
        """根据上下文过滤动作"""
        filtered = []
        
        for action in actions:
            # 过滤未启用的动作
            if not action.enabled:
                continue
            
            # 根据上下文条件过滤
            if not self._check_action_conditions(context, action):
                continue
            
            filtered.append(action)
        
        return filtered
    
    def _check_action_conditions(
        self,
        context: ActionContext,
        action: SuggestedAction
    ) -> bool:
        """检查动作是否满足执行条件"""
        # 检查结果数据条件
        if action.metadata.get("requires_result") and not context.result_data:
            return False
        
        # 检查历史动作条件
        required_history = action.metadata.get("requires_previous_action")
        if required_history and required_history not in context.previous_actions:
            return False
        
        # 检查会话上下文条件
        required_context = action.metadata.get("requires_context")
        if required_context:
            for key, expected_value in required_context.items():
                actual_value = context.session_context.get(key)
                if actual_value != expected_value:
                    return False
        
        return True
    
    def _sort_by_priority(self, actions: List[SuggestedAction]) -> List[SuggestedAction]:
        """按优先级排序"""
        priority_order = {
            ActionPriority.HIGH: 0,
            ActionPriority.MEDIUM: 1,
            ActionPriority.LOW: 2,
        }
        return sorted(actions, key=lambda a: priority_order.get(a.priority, 99))
    
    def add_action_template(
        self,
        intent: str,
        actions: List[SuggestedAction]
    ) -> None:
        """添加动作模板"""
        self._action_templates[intent] = actions
    
    def remove_action_template(self, intent: str) -> bool:
        """移除动作模板"""
        if intent in self._action_templates:
            del self._action_templates[intent]
            return True
        return False
    
    def get_action_by_id(
        self,
        action_id: str,
        context: ActionContext
    ) -> Optional[SuggestedAction]:
        """根据ID获取动作"""
        actions = self.get_next_actions(context, max_actions=20)
        for action in actions:
            if action.id == action_id:
                return action
        return None
    
    def build_action_buttons(
        self,
        actions: List[SuggestedAction]
    ) -> List[Dict[str, str]]:
        """构建前端按钮格式"""
        buttons = []
        for action in actions:
            buttons.append({
                "id": action.id,
                "label": action.label,
                "description": action.description,
                "icon": action.icon,
                "confirm_required": action.confirm_required,
            })
        return buttons
    
    def build_natural_language_suggestion(
        self,
        actions: List[SuggestedAction]
    ) -> str:
        """构建自然语言建议"""
        if not actions:
            return "您可以继续查询或进行其他操作。"
        
        action_labels = [f"「{a.label}」" for a in actions[:3]]
        
        if len(actions) == 1:
            return f"您可以{action_labels[0]}。"
        elif len(actions) == 2:
            return f"您可以{action_labels[0]}或{action_labels[1]}。"
        else:
            return f"您可以{action_labels[0]}、{action_labels[1]}或{action_labels[2]}。"


# 全局引擎实例
_next_action_engine: Optional[NextActionEngine] = None


def get_next_action_engine() -> NextActionEngine:
    """获取下一步动作引擎单例"""
    global _next_action_engine
    if _next_action_engine is None:
        _next_action_engine = NextActionEngine()
    return _next_action_engine
