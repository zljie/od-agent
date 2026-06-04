"""
Action Intent Normalizer
=======================
Normalizes user action expressions to standard Action Intent Catalog.

Based on Section 5.4 and 10.1 of BeBIOS_Intent_Processing_Architecture_v1.0.md

Action Intent Catalog:
- Query, Analyze, Create, Update, Delete
- Approve, Reject, Compare, Recommend
- Notify, ExecuteWorkflow, Explain, Summarize, Predict, Diagnose
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .semantic_frame import SemanticFrame, RiskLevel


# ---------------------------------------------------------------------------
# Action Intent Catalog
# ---------------------------------------------------------------------------

# Standard action intents from the BeBIOS architecture
STANDARD_ACTION_INTENTS: Set[str] = {
    "Query", "Analyze", "Create", "Update", "Delete",
    "Approve", "Reject", "Compare", "Recommend",
    "Notify", "ExecuteWorkflow", "Explain", "Summarize",
    "Predict", "Diagnose",
}

# Default risk levels for actions
DEFAULT_RISK_LEVELS: Dict[str, RiskLevel] = {
    "Query": RiskLevel.LOW,
    "Analyze": RiskLevel.LOW,
    "Explain": RiskLevel.LOW,
    "Summarize": RiskLevel.LOW,
    "Recommend": RiskLevel.LOW,
    "Notify": RiskLevel.LOW,
    "Compare": RiskLevel.LOW,
    "Predict": RiskLevel.LOW,
    "Diagnose": RiskLevel.LOW,
    "Update": RiskLevel.MEDIUM,
    "Delete": RiskLevel.HIGH,
    "Create": RiskLevel.HIGH,
    "Approve": RiskLevel.MEDIUM,
    "Reject": RiskLevel.MEDIUM,
    "ExecuteWorkflow": RiskLevel.HIGH,
}

# Actions that typically require HITL
DEFAULT_HITL_REQUIRED: Set[str] = {
    "Create", "Delete", "Approve", "Reject", "ExecuteWorkflow",
}


# ---------------------------------------------------------------------------
# Action Mapping Tables
# ---------------------------------------------------------------------------

# Chinese expressions to standard actions
CN_ACTION_MAPPINGS: Dict[str, str] = {
    # Query actions
    "查": "Query",
    "查询": "Query",
    "查一下": "Query",
    "查下": "Query",
    "看下": "Query",
    "看看": "Query",
    "看一下": "Query",
    "看一下": "Query",
    "搜": "Query",
    "搜索": "Query",
    "找": "Query",
    "找一下": "Query",
    "看看有没有": "Query",
    "看看有没有": "Query",
    "有没有": "Query",
    "多少": "Query",
    "统计": "Query",

    # Analyze actions
    "分析": "Analyze",
    "分析一下": "Analyze",
    "剖析": "Analyze",
    "分析下": "Analyze",
    "解读": "Analyze",

    # Create actions
    "创建": "Create",
    "新建": "Create",
    "新增": "Create",
    "添加": "Create",
    "增加": "Create",
    "发起": "Create",
    "生成": "Create",
    "生成一个": "Create",
    "生成一条": "Create",

    # Update actions
    "更新": "Update",
    "修改": "Update",
    "编辑": "Update",
    "调整": "Update",
    "变更": "Update",
    "改": "Update",
    "改一下": "Update",

    # Delete actions
    "删除": "Delete",
    "移除": "Delete",
    "取消": "Delete",
    "作废": "Delete",

    # Approve actions
    "审批": "Approve",
    "批准": "Approve",
    "通过": "Approve",
    "同意": "Approve",
    "核准": "Approve",

    # Reject actions
    "驳回": "Reject",
    "拒绝": "Reject",
    "不通过": "Reject",
    "否决": "Reject",

    # Compare actions
    "对比": "Compare",
    "比较": "Compare",
    "比对": "Compare",
    "比一下": "Compare",
    "比价": "Compare",

    # Recommend actions
    "推荐": "Recommend",
    "建议": "Recommend",
    "推荐一下": "Recommend",
    "生成建议": "Recommend",

    # Notify actions
    "通知": "Notify",
    "提醒": "Notify",
    "告知": "Notify",
    "发送": "Notify",

    # ExecuteWorkflow actions
    "执行": "ExecuteWorkflow",
    "执行工作流": "ExecuteWorkflow",
    "运行": "ExecuteWorkflow",
    "触发": "ExecuteWorkflow",

    # Explain actions
    "解释": "Explain",
    "说明": "Explain",
    "讲解": "Explain",
    "阐述": "Explain",

    # Summarize actions
    "总结": "Summarize",
    "汇总": "Summarize",
    "概括": "Summarize",

    # Predict actions
    "预测": "Predict",
    "预估": "Predict",
    "预报": "Predict",

    # Diagnose actions
    "诊断": "Diagnose",
    "检查": "Diagnose",
    "诊断": "Diagnose",
}

# English keywords to standard actions
EN_ACTION_MAPPINGS: Dict[str, str] = {
    "query": "Query",
    "get": "Query",
    "find": "Query",
    "search": "Query",
    "list": "Query",
    "show": "Query",
    "display": "Query",
    "retrieve": "Query",

    "analyze": "Analyze",
    "analysis": "Analyze",
    "examine": "Analyze",

    "create": "Create",
    "add": "Create",
    "new": "Create",
    "insert": "Create",
    "generate": "Create",

    "update": "Update",
    "modify": "Update",
    "edit": "Update",
    "change": "Update",

    "delete": "Delete",
    "remove": "Delete",
    "drop": "Delete",

    "approve": "Approve",
    "accept": "Approve",

    "reject": "Reject",
    "deny": "Reject",
    "refuse": "Reject",

    "compare": "Compare",
    "comparison": "Compare",

    "recommend": "Recommend",
    "suggest": "Recommend",

    "notify": "Notify",
    "send": "Notify",
    "alert": "Notify",

    "execute": "ExecuteWorkflow",
    "run": "ExecuteWorkflow",
    "start": "ExecuteWorkflow",
    "trigger": "ExecuteWorkflow",

    "explain": "Explain",
    "describe": "Explain",

    "summarize": "Summarize",
    "summary": "Summarize",

    "predict": "Predict",
    "forecast": "Predict",

    "diagnose": "Diagnose",
    "check": "Diagnose",
}


# ---------------------------------------------------------------------------
# Result Model
# ---------------------------------------------------------------------------

@dataclass
class NormalizationResult:
    """Result of action intent normalization."""
    action_intent: str           # Normalized action (e.g., "Query")
    action_candidate: str       # Original candidate from LLM
    confidence: float           # Normalization confidence [0.0, 1.0]
    matched_keywords: List[str]  # Keywords that matched
    is_ambiguous: bool = False   # True if multiple actions detected
    alternatives: List[str] = field(default_factory=list)  # Alternative actions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actionIntent": self.action_intent,
            "actionCandidate": self.action_candidate,
            "confidence": self.confidence,
            "matchedKeywords": self.matched_keywords,
            "isAmbiguous": self.is_ambiguous,
            "alternatives": self.alternatives,
        }


# ---------------------------------------------------------------------------
# Action Intent Normalizer
# ---------------------------------------------------------------------------

class ActionIntentNormalizer:
    """Normalizes user action expressions to standard Action Intent Catalog.

    This implements Section 5.4 of the BeBIOS architecture.

    Example:
        normalizer = ActionIntentNormalizer()
        result = normalizer.normalize("帮我查一下采购需求")
        print(result.action_intent)  # "Query"
    """

    def __init__(
        self,
        custom_mappings: Optional[Dict[str, str]] = None,
        custom_risk_levels: Optional[Dict[str, RiskLevel]] = None,
        custom_hitl_required: Optional[Set[str]] = None,
    ):
        """Initialize the normalizer.

        Args:
            custom_mappings: Additional action mappings (CN or EN keywords)
            custom_risk_levels: Override risk levels for actions
            custom_hitl_required: Set of actions requiring HITL
        """
        self.cn_mappings = {**CN_ACTION_MAPPINGS}
        self.en_mappings = {**EN_ACTION_MAPPINGS}

        if custom_mappings:
            for keyword, action in custom_mappings.items():
                if any('\u4e00' <= c <= '\u9fff' for c in keyword):
                    self.cn_mappings[keyword.lower()] = action
                else:
                    self.en_mappings[keyword.lower()] = action

        self.risk_levels = {**DEFAULT_RISK_LEVELS}
        if custom_risk_levels:
            self.risk_levels.update(custom_risk_levels)

        self.hitl_required = DEFAULT_HITL_REQUIRED.copy()
        if custom_hitl_required:
            self.hitl_required = custom_hitl_required

    def normalize(
        self,
        text: str,
        candidate_action: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ) -> NormalizationResult:
        """Normalize user action expression to standard intent.

        Args:
            text: The user input text or action expression to normalize
            candidate_action: Optional action candidate from LLM parser
            confidence_threshold: Minimum confidence to accept match

        Returns:
            NormalizationResult with normalized action and confidence
        """
        text_lower = text.lower().strip()

        # First, try to match from LLM candidate
        if candidate_action:
            candidate_normalized = self._normalize_candidate(candidate_action)
            if candidate_normalized:
                return NormalizationResult(
                    action_intent=candidate_normalized,
                    action_candidate=candidate_action,
                    confidence=0.95,
                    matched_keywords=[candidate_action],
                )

        # Try to match keywords in text
        matched_actions: Dict[str, List[str]] = {}

        # Check CN mappings
        for keyword, action in self.cn_mappings.items():
            if keyword in text_lower:
                if action not in matched_actions:
                    matched_actions[action] = []
                matched_actions[action].append(keyword)

        # Check EN mappings
        for keyword, action in self.en_mappings.items():
            if keyword in text_lower:
                if action not in matched_actions:
                    matched_actions[action] = []
                matched_actions[action].append(keyword)

        if not matched_actions:
            # No match found - default to Query as fallback
            return NormalizationResult(
                action_intent="Query",
                action_candidate=text[:50] if len(text) > 50 else text,
                confidence=0.5,
                matched_keywords=[],
                alternatives=[],
            )

        # Find the action with most keyword matches
        best_action = max(matched_actions.keys(), key=lambda a: len(matched_actions[a]))
        best_keywords = matched_actions[best_action]

        # Check for ambiguity (multiple actions with similar confidence)
        alternatives = [
            action for action, keywords in matched_actions.items()
            if action != best_action and len(keywords) >= len(best_keywords) * 0.8
        ]

        is_ambiguous = len(alternatives) > 0

        # Calculate confidence based on keyword coverage
        keyword_count = len(best_keywords)
        confidence = min(0.5 + (keyword_count * 0.15), 1.0)

        return NormalizationResult(
            action_intent=best_action,
            action_candidate=text[:100] if len(text) > 100 else text,
            confidence=confidence,
            matched_keywords=best_keywords,
            is_ambiguous=is_ambiguous,
            alternatives=alternatives[:3],
        )

    def _normalize_candidate(self, candidate: str) -> Optional[str]:
        """Normalize an LLM action candidate to standard intent."""
        if not candidate:
            return None

        candidate_lower = candidate.lower().strip()

        # Check if already a standard intent
        if candidate in STANDARD_ACTION_INTENTS:
            return candidate

        # Check direct mapping
        if candidate_lower in self.en_mappings:
            return self.en_mappings[candidate_lower]

        # Check if candidate contains standard intent
        for intent in STANDARD_ACTION_INTENTS:
            if intent.lower() in candidate_lower:
                return intent

        return None

    def get_risk_level(self, action: str) -> RiskLevel:
        """Get risk level for an action."""
        return self.risk_levels.get(action, RiskLevel.MEDIUM)

    def requires_hitl(self, action: str) -> bool:
        """Check if an action requires HITL."""
        return action in self.hitl_required

    def get_supported_objects(self, action: str) -> List[str]:
        """Get list of objects that support this action.

        This should be loaded from configuration in production.
        """
        # Default supported objects (should be configured via ActionIntentCatalog)
        supported = {
            "Query": ["PurchaseRequirement", "Inquiry", "Quotation", "PurchaseOrder",
                      "Contract", "Supplier", "Material"],
            "Create": ["PurchaseRequirement", "Inquiry", "Quotation", "PurchaseOrder", "Contract"],
            "Update": ["PurchaseRequirement", "PurchaseOrder", "Supplier"],
            "Delete": ["PurchaseRequirement", "Inquiry"],
            "Approve": ["PurchaseRequirement", "PurchaseOrder"],
            "Reject": ["PurchaseRequirement", "PurchaseOrder"],
            "Compare": ["Quotation", "Supplier", "Material"],
            "Analyze": ["PurchaseRequirement", "PurchaseOrder", "Supplier"],
        }
        return supported.get(action, [])


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_default_normalizer: Optional[ActionIntentNormalizer] = None


def get_default_normalizer() -> ActionIntentNormalizer:
    """Get the default normalizer instance."""
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = ActionIntentNormalizer()
    return _default_normalizer


def normalize_action(
    text: str,
    candidate: Optional[str] = None,
) -> NormalizationResult:
    """Convenience function to normalize an action expression."""
    return get_default_normalizer().normalize(text, candidate)
