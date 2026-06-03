"""对话状态管理模块 - 管理多轮对话中的状态流转

本模块实现文档定义的6种对话状态：
- 建议态 (suggestion): 显示分析结果和建议
- 确认态 (confirmation): 等待用户确认执行
- 执行态 (executing): 正在执行操作
- 完成态 (completed): 操作成功完成
- 阻断态 (blocked): 规则阻断
- 异常态 (error): 系统异常
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class DialogState(str, Enum):
    """对话状态枚举"""
    SUGGESTION = "suggestion"      # 建议态 - 显示分析结果和建议
    CONFIRMATION = "confirmation"  # 确认态 - 等待用户确认执行
    EXECUTING = "executing"       # 执行态 - 正在执行操作
    COMPLETED = "completed"       # 完成态 - 操作成功完成
    BLOCKED = "blocked"           # 阻断态 - 规则阻断
    ERROR = "error"              # 异常态 - 系统异常


@dataclass
class StateContext:
    """状态上下文 - 记录当前状态的详细信息"""
    state: DialogState                           # 当前状态
    intent: str = ""                            # 识别的意图
    action: str = ""                            # 当前动作
    message: str = ""                           # 状态消息
    data: Optional[Dict[str, Any]] = None       # 状态相关数据
    rules_triggered: List[str] = field(default_factory=list)   # 触发的规则
    connectors_called: List[str] = field(default_factory=list)  # 调用的连接器
    timestamp: str = ""                         # 状态时间
    session_id: str = ""                         # 会话ID
    user_id: str = ""                           # 用户ID
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.data is None:
            self.data = {}


@dataclass
class StateTransition:
    """状态转换记录"""
    from_state: DialogState          # 原状态
    to_state: DialogState           # 目标状态
    trigger: str                    # 触发原因
    timestamp: str = ""             # 转换时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DialogSession:
    """对话会话 - 管理整个对话的生命周期"""
    session_id: str
    user_id: str = ""
    current_state: DialogState = DialogState.SUGGESTION
    state_context: Optional[StateContext] = None
    state_history: List[StateTransition] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)  # 对话历史
    context_data: Dict[str, Any] = field(default_factory=dict)   # 跨状态共享数据
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "current_state": self.current_state.value,
            "state_context": {
                "state": self.state_context.state.value if self.state_context else None,
                "intent": self.state_context.intent if self.state_context else None,
                "action": self.state_context.action if self.state_context else None,
                "message": self.state_context.message if self.state_context else None,
                "rules_triggered": self.state_context.rules_triggered if self.state_context else [],
                "timestamp": self.state_context.timestamp if self.state_context else None,
            },
            "state_history": [
                {
                    "from_state": t.from_state.value,
                    "to_state": t.to_state.value,
                    "trigger": t.trigger,
                    "timestamp": t.timestamp,
                }
                for t in self.state_history
            ],
            "context_data_keys": list(self.context_data.keys()),
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class DialogStateMachine:
    """对话状态机 - 管理状态转换逻辑"""
    
    # 允许的状态转换映射
    ALLOWED_TRANSITIONS: Dict[DialogState, List[DialogState]] = {
        DialogState.SUGGESTION: [
            DialogState.CONFIRMATION,  # 需要确认
            DialogState.EXECUTING,     # 直接执行
            DialogState.COMPLETED,     # 查询类操作直接完成
            DialogState.BLOCKED,       # 规则阻断
            DialogState.ERROR,         # 系统异常
        ],
        DialogState.CONFIRMATION: [
            DialogState.EXECUTING,     # 用户确认，开始执行
            DialogState.SUGGESTION,    # 用户取消，返回建议态
            DialogState.CANCELLED,     # 显式取消
        ],
        DialogState.EXECUTING: [
            DialogState.COMPLETED,     # 执行成功
            DialogState.BLOCKED,      # 执行时被阻断
            DialogState.ERROR,         # 执行异常
        ],
        DialogState.COMPLETED: [
            DialogState.SUGGESTION,    # 开始新操作
        ],
        DialogState.BLOCKED: [
            DialogState.SUGGESTION,    # 解决阻断或开始新操作
        ],
        DialogState.ERROR: [
            DialogState.SUGGESTION,    # 处理错误或开始新操作
        ],
    }
    
    @classmethod
    def can_transition(cls, from_state: DialogState, to_state: DialogState) -> bool:
        """检查状态转换是否允许"""
        if from_state == to_state:
            return True
        allowed = cls.ALLOWED_TRANSITIONS.get(from_state, [])
        return to_state in allowed
    
    @classmethod
    def get_allowed_transitions(cls, from_state: DialogState) -> List[DialogState]:
        """获取允许的目标状态列表"""
        return cls.ALLOWED_TRANSITIONS.get(from_state, [])


class DialogStateManager:
    """对话状态管理器 - 管理会话状态"""
    
    def __init__(self):
        self._sessions: Dict[str, DialogSession] = {}
        self._default_ttl: int = 3600  # 默认会话过期时间(秒)
    
    def create_session(self, session_id: str, user_id: str = "") -> DialogSession:
        """创建新的对话会话"""
        session = DialogSession(
            session_id=session_id,
            user_id=user_id,
            current_state=DialogState.SUGGESTION
        )
        self._sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[DialogSession]:
        """获取会话"""
        return self._sessions.get(session_id)
    
    def get_or_create_session(self, session_id: str, user_id: str = "") -> DialogSession:
        """获取或创建会话"""
        if session_id not in self._sessions:
            return self.create_session(session_id, user_id)
        return self._sessions[session_id]
    
    def update_state(
        self,
        session_id: str,
        new_state: DialogState,
        context: Optional[StateContext] = None,
        trigger: str = ""
    ) -> Optional[StateTransition]:
        """更新会话状态"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        # 检查转换是否允许
        if not DialogStateMachine.can_transition(session.current_state, new_state):
            # 允许的转换之外的处理：如果是同一个状态，只更新上下文
            if session.current_state != new_state:
                return None
        
        # 记录状态转换
        transition = StateTransition(
            from_state=session.current_state,
            to_state=new_state,
            trigger=trigger
        )
        session.state_history.append(transition)
        
        # 更新状态
        session.current_state = new_state
        session.updated_at = datetime.now().isoformat()
        
        if context:
            session.state_context = context
        
        return transition
    
    def set_context_data(self, session_id: str, key: str, value: Any) -> bool:
        """设置会话上下文数据"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.context_data[key] = value
        session.updated_at = datetime.now().isoformat()
        return True
    
    def get_context_data(self, session_id: str, key: str, default: Any = None) -> Any:
        """获取会话上下文数据"""
        session = self._sessions.get(session_id)
        if not session:
            return default
        return session.context_data.get(key, default)
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """添加消息到会话历史"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            message["metadata"] = metadata
        
        session.messages.append(message)
        session.updated_at = datetime.now().isoformat()
        return True
    
    def get_state_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取状态摘要"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "current_state": session.current_state.value,
            "state_message": session.state_context.message if session.state_context else "",
            "intent": session.state_context.intent if session.state_context else "",
            "action": session.state_context.action if session.state_context else "",
            "rules_triggered": session.state_context.rules_triggered if session.state_context else [],
            "message_count": len(session.messages),
            "last_updated": session.updated_at,
        }
    
    def reset_session(self, session_id: str) -> bool:
        """重置会话状态"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.current_state = DialogState.SUGGESTION
        session.state_context = None
        session.state_history.clear()
        session.context_data.clear()
        session.updated_at = datetime.now().isoformat()
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话"""
        return [s.to_dict() for s in self._sessions.values()]


class DialogStateResponseBuilder:
    """对话状态响应构建器 - 根据状态生成相应的响应"""
    
    @staticmethod
    def build_state_message(state: DialogState, context: Optional[StateContext] = None) -> str:
        """根据状态构建消息"""
        if state == DialogState.SUGGESTION:
            return DialogStateResponseBuilder._build_suggestion_message(context)
        elif state == DialogState.CONFIRMATION:
            return DialogStateResponseBuilder._build_confirmation_message(context)
        elif state == DialogState.EXECUTING:
            return DialogStateResponseBuilder._build_executing_message(context)
        elif state == DialogState.COMPLETED:
            return DialogStateResponseBuilder._build_completed_message(context)
        elif state == DialogState.BLOCKED:
            return DialogStateResponseBuilder._build_blocked_message(context)
        elif state == DialogState.ERROR:
            return DialogStateResponseBuilder._build_error_message(context)
        return ""
    
    @staticmethod
    def _build_suggestion_message(context: Optional[StateContext]) -> str:
        """构建建议态消息"""
        if not context:
            return ""
        
        parts = []
        
        if context.data:
            if "results" in context.data:
                results = context.data["results"]
                parts.append(f"查询到 **{len(results)}** 条记录")
            if "summary" in context.data:
                parts.append(context.data["summary"])
        
        return "\n".join(parts)
    
    @staticmethod
    def _build_confirmation_message(context: Optional[StateContext]) -> str:
        """构建确认态消息"""
        if not context:
            return "请确认是否继续执行"
        
        parts = [f"**{context.message}**"]
        
        if context.data:
            if "details" in context.data:
                parts.append("\n操作详情:")
                for key, value in context.data["details"].items():
                    parts.append(f"- {key}: {value}")
        
        if context.rules_triggered:
            parts.append("\n⚠️ 触发的规则:")
            for rule in context.rules_triggered:
                parts.append(f"- {rule}")
        
        parts.append("\n\n回复 \"确认\" 执行，回复 \"取消\" 放弃。")
        
        return "\n".join(parts)
    
    @staticmethod
    def _build_executing_message(context: Optional[StateContext]) -> str:
        """构建执行态消息"""
        if not context:
            return "正在执行操作..."
        
        parts = ["🔄 正在执行"]
        
        if context.action:
            parts.append(f"动作: {context.action}")
        
        if context.connectors_called:
            parts.append("调用系统:")
            for connector in context.connectors_called:
                parts.append(f"- {connector}")
        
        return "\n".join(parts)
    
    @staticmethod
    def _build_completed_message(context: Optional[StateContext]) -> str:
        """构建完成态消息"""
        if not context:
            return "操作已完成"
        
        parts = ["✅ 操作成功完成"]
        
        if context.data:
            if "result" in context.data:
                parts.append(f"\n结果: {context.data['result']}")
            if "next_action" in context.data:
                parts.append(f"\n下一步: {context.data['next_action']}")
        
        return "\n".join(parts)
    
    @staticmethod
    def _build_blocked_message(context: Optional[StateContext]) -> str:
        """构建阻断态消息"""
        if not context:
            return "操作被阻断，无法继续"
        
        parts = ["🚫 操作无法执行"]
        
        if context.message:
            parts.append(f"\n原因: {context.message}")
        
        if context.rules_triggered:
            parts.append("\n触发的规则:")
            for rule in context.rules_triggered:
                parts.append(f"- {rule}")
        
        if context.data and "remediation" in context.data:
            parts.append(f"\n建议: {context.data['remediation']}")
        
        return "\n".join(parts)
    
    @staticmethod
    def _build_error_message(context: Optional[StateContext]) -> str:
        """构建异常态消息"""
        if not context:
            return "发生错误，请稍后重试"
        
        parts = ["❌ 操作异常"]
        
        if context.message:
            parts.append(f"\n错误信息: {context.message}")
        
        if context.data and "error_type" in context.data:
            parts.append(f"\n错误类型: {context.data['error_type']}")
        
        parts.append("\n请稍后重试，或联系管理员。")
        
        return "\n".join(parts)


# 全局状态管理器实例
_state_manager: Optional[DialogStateManager] = None


def get_state_manager() -> DialogStateManager:
    """获取状态管理器单例"""
    global _state_manager
    if _state_manager is None:
        _state_manager = DialogStateManager()
    return _state_manager
