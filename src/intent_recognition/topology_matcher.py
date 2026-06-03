"""
Layer 2: Ontology Topology Matcher
===================================
Matches LLM-extracted intent against the intent topology graph.

Responsibilities:
1. Object matching: verify object_candidate exists in topology
2. Action mapping: map action_candidate to topology action IDs
3. Dimension matching: align filters with dataset fields
4. Path selection: find matching topology paths by object + template
5. Gap detection: identify missing mappings
6. B_score calculation: weighted confidence from all matches

Key design:
- Uses IntentTopology from Phase 1 (intent_topology module)
- Computes per-component scores and combined B_score
- Detects gaps for downstream HITL or deep reasoning
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    ActionMatch,
    DimensionMatch,
    ObjectMatch,
    OntologyMatchResult,
)
from .light_reasoner import LLMLightResult
from .preprocessor import PreprocessingResult


# ---------------------------------------------------------------------------
# Known ontology objects and their aliases
# ---------------------------------------------------------------------------

_OBJECT_ALIAS_MAP: Dict[str, List[str]] = {
    "purchase_requests": [
        "采购需求",
        "PR",
        "需求",
        "需求单",
        "pr",
        "purchase_request",
        "requisition",
    ],
    "inquiries": [
        "询价单",
        "RFQ",
        "INQ",
        "询价",
        "rfq",
        "inq",
        "inquiry",
        "请求报价",
    ],
    "quotations": [
        "报价单",
        "QUO",
        "报价",
        "quo",
        "quotation",
        "报价书",
        "报价单据",
    ],
    "purchase_orders": [
        "采购订单",
        "PO",
        "订单",
        "po",
        "purchase_order",
        "采购单",
    ],
    "contracts": [
        "合同",
        "contract",
        "合约",
        "协议",
    ],
    "suppliers": [
        "供应商",
        "vendor",
        "Vendor",
        "厂商",
        "供货商",
    ],
    "materials": [
        "物料",
        "material",
        "材料",
        "商品",
        "产品",
    ],
    "approval_flows": [
        "审批流",
        "审批",
        "approval",
        "审批流程",
    ],
}


# ---------------------------------------------------------------------------
# Action verb to ontology action mapping
# ---------------------------------------------------------------------------

_ACTION_VERB_MAP: Dict[str, List[str]] = {
    # Query/list actions
    "list": ["列出", "列表", "查询", "查看", "检索", "搜索", "浏览", "获取", "找到", "list", "query", "get", "show"],
    "get": ["获取", "查看", "详情", "get", "view", "detail"],
    # Create actions
    "create": ["创建", "新建", "生成", "新增", "添加", "录入", "create", "add", "new"],
    # Update actions
    "update": ["修改", "更新", "变更", "编辑", "调整", "update", "edit", "modify"],
    # Submit/approve actions
    "submit": ["提交", "上报", "发送", "发布", "submit", "send"],
    "approve": ["批准", "通过", "确认", "同意", "approve", "pass", "confirm"],
    "reject": ["驳回", "拒绝", "不同意", "reject", "deny", "refuse"],
    # Comparison actions
    "compare": ["比较", "对比", "比价", "compare", "contrast"],
    "recommend": ["推荐", "建议", "recommend", "suggest"],
    # Generation actions
    "generate": ["生成", "创建", "生成订单", "generate", "create"],
}


def _fuzzy_match(text: str, candidates: List[str], threshold: float = 0.7) -> Tuple[Optional[str], float]:
    """Simple fuzzy matching between text and candidate strings.

    Uses substring matching and character overlap ratio.
    Returns (matched_candidate, score) or (None, 0.0) if no match.
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return None, 0.0

    best_match: Optional[str] = None
    best_score = 0.0

    for candidate in candidates:
        candidate_lower = candidate.lower().strip()

        # Exact match
        if text_lower == candidate_lower:
            return candidate, 1.0

        # Substring match (text in candidate or candidate in text)
        if text_lower in candidate_lower:
            score = len(text_lower) / len(candidate_lower)
            if score > best_score:
                best_match = candidate
                best_score = score

        if candidate_lower in text_lower:
            score = len(candidate_lower) / len(text_lower)
            if score > best_score:
                best_match = candidate
                best_score = score

    if best_score >= threshold:
        return best_match, best_score

    return None, 0.0


def _match_object(object_candidate: str, object_label: str) -> ObjectMatch:
    """Match object_candidate against known ontology objects.

    Uses both the raw candidate and its label for matching.
    """
    # Build all searchable strings for each object
    search_candidates: Dict[str, List[str]] = {}
    for obj, aliases in _OBJECT_ALIAS_MAP.items():
        search_candidates[obj] = [obj] + aliases

    all_candidates: List[Tuple[str, List[str]]] = list(search_candidates.items())

    # Try matching with object_candidate first
    matched, score = _fuzzy_match(object_candidate, [obj for obj, _ in all_candidates])

    if matched is None and score == 0.0:
        # Try matching with object_label
        matched, score = _fuzzy_match(object_label, [obj for obj, _ in all_candidates])

    if matched is None:
        # Try against aliases
        for obj, aliases in all_candidates:
            m, s = _fuzzy_match(object_candidate, aliases)
            if m is not None and s > score:
                matched = obj
                score = s
                break

            m, s = _fuzzy_match(object_label, aliases)
            if m is not None and s > score:
                matched = obj
                score = s
                break

    if matched is None:
        return ObjectMatch.from_topology_matcher(object="", aliases=[], score=0.0)

    return ObjectMatch.from_topology_matcher(
        object=matched,
        aliases=_OBJECT_ALIAS_MAP.get(matched, []),
        score=score,
    )


def _match_action(action_candidate: str) -> ActionMatch:
    """Map action_candidate to ontology action IDs.

    Uses verb-based mapping to find corresponding ontology actions.
    """
    if not action_candidate:
        return ActionMatch.from_topology_matcher(raw_action="", mapped_action="", mapped_to=[], score=0.0)

    action_lower = action_candidate.lower().strip()

    # Find matching action verbs
    matched_verbs: List[str] = []
    for verb, aliases in _ACTION_VERB_MAP.items():
        for alias in aliases:
            if action_lower == alias.lower():
                matched_verbs.append(verb)
                break
            if action_lower in alias.lower() or alias.lower() in action_lower:
                matched_verbs.append(verb)
                break

    if not matched_verbs:
        # Default to "list" for unknown actions
        return ActionMatch.from_topology_matcher(
            raw_action=action_candidate,
            mapped_action="list",
            mapped_to=["list"],
            score=0.3,
        )

    # Use the most confident verb (first in list is best match)
    primary_verb = matched_verbs[0]
    score = min(0.9, 0.5 + 0.1 * len(matched_verbs))  # Higher score for multiple matches

    return ActionMatch.from_topology_matcher(
        raw_action=action_candidate,
        mapped_action=primary_verb,
        mapped_to=matched_verbs,
        score=score,
    )


def _match_dimensions(
    filters: Dict[str, Any],
    object_match: ObjectMatch,
) -> List[DimensionMatch]:
    """Match filter keys against dataset fields for the matched object.

    Determines which filters need dynamic lookup based on their values.
    """
    dimension_matches: List[DimensionMatch] = []

    # Known field patterns for each object type
    OBJECT_FIELDS: Dict[str, Dict[str, List[str]]] = {
        "purchase_requests": {
            "pr_id": ["pr_id", "需求编号", "PR编号", "PR号"],
            "pr_item": ["pr_item", "行项目", "行号", "item"],
            "apply_dep": ["apply_dep", "申请部门", "部门"],
            "purchase_type": ["purchase_type", "采购类型", "类型"],
            "flow_status": ["flow_status", "状态", "流程状态"],
            "requester": ["requester", "申请人", "请求人"],
            "material_keyword": ["material_keyword", "物料", "物料描述"],
        },
        "inquiries": {
            "inquiry_id": ["inquiry_id", "询价单号", "RFQ号", "INQ号"],
            "pr_id": ["pr_id", "关联需求", "需求编号"],
            "vendor_id": ["vendor_id", "供应商编号"],
            "status": ["status", "状态"],
        },
        "quotations": {
            "quotation_id": ["quotation_id", "报价单号", "QUO号"],
            "inquiry_id": ["inquiry_id", "询价单号"],
            "vendor_id": ["vendor_id", "供应商"],
            "net_price": ["net_price", "单价", "价格"],
            "status": ["status", "状态"],
        },
        "purchase_orders": {
            "po_id": ["po_id", "订单号", "PO号"],
            "pr_id": ["pr_id", "关联需求"],
            "quotation_id": ["quotation_id", "关联报价"],
            "vendor_id": ["vendor_id", "供应商"],
            "quantity": ["quantity", "数量"],
            "status": ["status", "状态"],
        },
        "suppliers": {
            "vendor_id": ["vendor_id", "供应商编号"],
            "vendor_name": ["vendor_name", "供应商名称", "厂商"],
        },
        "materials": {
            "material_id": ["material_id", "物料编号"],
            "material_name": ["material_name", "物料名称", "物料描述"],
        },
    }

    object_fields = OBJECT_FIELDS.get(object_match.object, {})

    for filter_key, filter_value in filters.items():
        if filter_value is None or filter_value == "":
            continue

        # Try to match filter_key to a known field
        matched_field: Optional[str] = None
        for field_name, aliases in object_fields.items():
            if filter_key.lower() in [a.lower() for a in aliases]:
                matched_field = field_name
                break

        if matched_field is None:
            # Check if filter_key matches any known alias
            for field_name, aliases in object_fields.items():
                for alias in aliases:
                    if alias.lower() in filter_key.lower() or filter_key.lower() in alias.lower():
                        matched_field = field_name
                        break
                if matched_field:
                    break

        # Determine if this needs dynamic lookup
        # Values that look like entity IDs need lookup
        need_lookup = False
        value_str = str(filter_value)

        # Check for ID patterns
        id_patterns = ["PR-", "PO-", "RFQ-", "INQ-", "QUO-", "V-"]
        if any(value_str.upper().startswith(p.upper()) for p in id_patterns):
            need_lookup = False  # ID pattern is self-resolving
        # Names that are too short or too common need lookup
        elif len(value_str) <= 3 and not value_str.isdigit():
            need_lookup = True
        # Department names, person names typically need lookup
        elif any(word in value_str for word in ["部", "部门", "科", "公司", "经理", "总"]):
            need_lookup = True

        dimension_matches.append(
            DimensionMatch.from_topology_matcher(
                raw_value=value_str,
                field=matched_field or filter_key,
                need_dynamic_lookup=need_lookup,
                score=0.8 if matched_field else 0.4,
            )
        )

    return dimension_matches


def _find_topology_paths(
    object_match: ObjectMatch,
    action_match: ActionMatch,
    intent_template: str,
) -> List[str]:
    """Find matching topology paths from object + action + template.

    Returns path IDs that match the criteria.
    """
    # This would normally query the IntentTopology
    # For now, return a structured path ID based on matching
    if not object_match.object:
        return []

    # Build path ID pattern: {object}.{action}.{template}
    path_id = f"{object_match.object}.{action_match.mapped_action}.{intent_template}"

    # Also add variations
    paths = [path_id]

    # Add list variant if action is ambiguous
    if action_match.mapped_action in ["list", "get"]:
        paths.append(f"{object_match.object}.list.{intent_template}")
        paths.append(f"{object_match.object}.get.{intent_template}")

    return paths


def _detect_gaps(
    object_match: ObjectMatch,
    action_match: ActionMatch,
    intent_template: str,
    dimension_matches: List[DimensionMatch],
) -> List[str]:
    """Detect gaps in the ontology matching.

    Returns a list of gap descriptions.
    """
    gaps: List[str] = []

    if object_match.confidence < 0.5:
        gaps.append(f"对象匹配置信度过低 (score={object_match.confidence}): {object_match.object or '未匹配'}")

    if action_match.confidence < 0.5:
        gaps.append(f"动作匹配置信度过低 (score={action_match.confidence}): {action_match.raw_action}")

    if not intent_template:
        gaps.append("意图模板未识别")

    # Check for low-scoring dimensions
    low_dims = [d for d in dimension_matches if d.confidence < 0.5]
    if low_dims:
        gaps.append(f"以下过滤字段匹配度低: {[d.raw_value for d in low_dims]}")

    # Check for dimensions needing lookup but not yet resolved
    lookup_dims = [d for d in dimension_matches if d.need_dynamic_lookup]
    if lookup_dims:
        gaps.append(f"以下字段需要动态解析: {[d.raw_value for d in lookup_dims]}")

    return gaps


def _calculate_b_score(
    object_match: ObjectMatch,
    action_match: ActionMatch,
    dimension_matches: List[DimensionMatch],
    gaps: List[str],
) -> float:
    """Calculate overall B_score from ontology matching.

    Weighted combination:
    - Object match: 40%
    - Action match: 30%
    - Dimension match: 20%
    - Gap penalty: -10% per gap (up to -30%)
    """
    weights = {
        "object": 0.40,
        "action": 0.30,
        "dimension": 0.20,
    }

    # Component scores
    object_score = object_match.confidence if object_match.object else 0.0
    action_score = action_match.confidence if action_match.mapped_action else 0.0

    # Average dimension score
    if dimension_matches:
        dim_score = sum(d.confidence for d in dimension_matches) / len(dimension_matches)
    else:
        dim_score = 0.0

    # Base score
    base_score = (
        weights["object"] * object_score
        + weights["action"] * action_score
        + weights["dimension"] * dim_score
    )

    # Gap penalty
    gap_penalty = min(0.3, len(gaps) * 0.1)
    final_score = max(0.0, base_score - gap_penalty)

    return round(final_score, 4)


class OntologyTopologyMatcher:
    """Layer 2 ontology topology matcher.

    Aligns LLM-extracted intent with the procurement ontology topology
    to validate object, action, and dimension mappings.

    Attributes
    ----------
    topology:
        IntentTopology instance from Phase 1 (optional, for future use).
    """

    def __init__(self, topology=None):
        """Initialize the topology matcher.

        Parameters
        ----------
        topology:
            Optional IntentTopology instance for advanced path lookup.
        """
        self.topology = topology

    def match(
        self,
        llm_result: LLMLightResult,
        preprocessing: PreprocessingResult,
        topology=None,
    ) -> OntologyMatchResult:
        """Run ontology topology matching on LLM result.

        Parameters
        ----------
        llm_result:
            Result from Layer 1 (LightIntentReasoner).
        preprocessing:
            Result from Layer 0 (InputPreprocessor).
        topology:
            Optional IntentTopology for path lookup.

        Returns
        -------
        OntologyMatchResult
            Complete match result with object, action, dimension matches,
            topology paths, gaps, and B_score.
        """
        # Use provided topology or instance-level
        effective_topology = topology or self.topology

        # Step 1: Object matching
        object_match = _match_object(
            llm_result.object_candidate,
            llm_result.object_label,
        )

        # Step 2: Action mapping
        action_match = _match_action(llm_result.action_candidate)

        # Step 3: Dimension matching
        dimension_matches = _match_dimensions(
            llm_result.filters,
            object_match,
        )

        # Step 4: Find topology paths
        topology_paths = _find_topology_paths(
            object_match,
            action_match,
            llm_result.intent_template,
        )

        # Step 5: Gap detection
        gaps = _detect_gaps(
            object_match,
            action_match,
            llm_result.intent_template,
            dimension_matches,
        )

        # Step 6: Calculate B_score
        b_score = _calculate_b_score(
            object_match,
            action_match,
            dimension_matches,
            gaps,
        )

        return OntologyMatchResult(
            object_match=object_match,
            action_match=action_match,
            dimension_matches=dimension_matches,
            matched_path_ids=topology_paths,
            gaps=gaps,
            confidence_b=b_score,
        )

    def get_ontology_summary(self) -> Dict[str, Any]:
        """Get a summary of available ontology objects and actions.

        Used to inject into Layer 1 prompts.
        """
        return {
            "objects": list(_OBJECT_ALIAS_MAP.keys()),
            "actions": list(_ACTION_VERB_MAP.keys()),
            "dimensions": [
                "pr_id",
                "inquiry_id",
                "quotation_id",
                "po_id",
                "apply_dep",
                "vendor_id",
                "material_id",
                "purchase_type",
                "flow_status",
                "date_range",
            ],
        }
