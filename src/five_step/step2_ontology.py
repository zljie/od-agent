"""Step 2: Ontology Expansion for the 5-step pipeline.

PHASE 4: Ontology Expansion (formerly "Object Resolution")
=========================================================

Step 2's role has been upgraded from "Object Resolution" to "Ontology Expansion".

Before (Phase 3):
    Intent → Object Locator → Re-identify the object again

After (Phase 4):
    Intent → Semantic Contract → Ontology Expansion (supplement, not replace)

Key changes:
1. Semantic Contract from Step 1 is the SINGLE SOURCE OF TRUTH
2. Step 2's job is to EXPAND the contract with ontology knowledge:
   - Supplement relationships (foreign keys, associations)
   - Supplement rules (business rules, defaults)
   - Supplement fields (additional filterable fields)
3. Step 2 CANNOT override the action from the Semantic Contract
4. If Semantic Contract exists, use it directly as the object basis

This prevents "Semantic Collapse" where high-confidence intent recognition
gets overwritten by lower-confidence downstream steps.
"""

from typing import Dict, List, Optional, Any
from .models import OntologyResolveResult, SemanticContract
from ..procurement.ontology_loader import get_procurement_ontology, OntologyQuery
from ..prompt_config import get_field


_STEP2_ONTOLOGY_MATCHING_PROMPT_DEFAULT = """## 任务
基于以下业务本体和用户输入，进行深度语义推理，找出最匹配的业务对象。

## 用户输入
{user_input}

## Step1 意图识别结果
- 意图: {intent_id} ({intent_label})
- 对象术语: {object_term}
- 操作类型: {operation_type}

## 采购业务本体
{ontology_context}

## 推理过程
请按以下步骤思考：

1. **分析用户意图**: 用户说的"对象术语"是什么？采购计划/请购单/PR 这些都指向采购需求
2. **理解业务语义**: 采购计划、请购单、PR 都是 purchase_requests 的不同叫法
3. **检查本体覆盖**: 从本体中找到语义最接近的数据集
4. **输出推理结果**:

请按以下 JSON 格式输出（只输出 JSON，不要其他内容）：
{{
    "matched_name": "purchase_requests",  // 匹配的 dataset name
    "reasoning": "用户说的'采购计划'在本体系中对应'采购需求'（purchase_requests），因为...",  // 推理过程
    "confidence": 0.95,  // 置信度 0-1
    "alternatives": [  // 备选方案
        {{"name": "purchase_inquiries", "reason": "如果用户实际想询价..."}}
    ]
}}

如果无法确定匹配，输出：
{{
    "matched_name": null,
    "reasoning": "无法确定匹配...",
    "confidence": 0.0,
    "alternatives": []
}}

"""


def _step2_ontology_matching_prompt() -> str:
    return get_field(
        "five_step",
        "step2_ontology_matching_prompt",
        _STEP2_ONTOLOGY_MATCHING_PROMPT_DEFAULT,
    )


class Step2OntologyResolver:
    """Step 2: Ontology Object Resolution.

    Resolves the matched ontology object based on intent and user input.
    Uses the YAML-based procurement ontology as the single source of truth.
    """

    def __init__(self):
        self._ontology = get_procurement_ontology()
        self._query: Optional[OntologyQuery] = None

    def _get_query(self) -> OntologyQuery:
        """Get the ontology query interface."""
        if self._query is None:
            self._query = OntologyQuery(self._ontology)
        return self._query

    def resolve(self, user_input: str, intent_result, composite_result=None) -> OntologyResolveResult:
        """Expand ontology knowledge for the given intent using Semantic Contract.

        Args:
            user_input: The original user message
            intent_result: Result from Step 1 (contains Semantic Contract)
            composite_result: Optional CompositeIntentResult from Step1's composite pipeline.

        Returns:
            OntologyResolveResult with expanded ontology knowledge
        """
        # PHASE 4: Check for Semantic Contract from Step 1
        semantic_contract = getattr(intent_result, 'semantic_contract', None)
        if semantic_contract:
            return self._expand_from_semantic_contract(semantic_contract, intent_result, composite_result)

        # Legacy path: no Semantic Contract (for backward compatibility)
        if composite_result is not None:
            return self._build_result_from_composite(composite_result, intent_result)

        import asyncio

        query = self._get_query()

        # Strategy 1: Find object by the object_term from Step1
        matched_object = None
        if intent_result.object_term:
            matched_object = query.find_object_by_term(intent_result.object_term)

        # Strategy 2: Find object by intent action ID
        if not matched_object and intent_result.intent:
            matched_object = query.find_object_by_intent(intent_result.intent)

        # Strategy 3: LLM-powered fallback (for complex cases)
        llm_reasoning = None
        llm_alternatives = []
        if not matched_object:
            try:
                loop = asyncio.get_running_loop()
                matched_object, llm_reasoning, llm_alternatives = self._llm_resolve_fallback_sync(
                    user_input, intent_result, query
                )
            except RuntimeError:
                matched_object, llm_reasoning, llm_alternatives = self._llm_resolve_fallback_sync(
                    user_input, intent_result, query
                )

        if matched_object:
            llm_inferred = llm_reasoning is not None
            result = self._build_success_result(matched_object, intent_result, query)
            result.llm_inferred = llm_inferred
            result.llm_reasoning = llm_reasoning
            result.alternatives = llm_alternatives
            return result

        # Fallback when no ontology match
        return self._fallback_resolution(intent_result)

    def _expand_from_semantic_contract(
        self,
        semantic_contract: SemanticContract,
        intent_result,
        composite_result=None
    ) -> OntologyResolveResult:
        """Expand ontology knowledge FROM the Semantic Contract.

        This is the Phase 4 Ontology Expansion:
        - Use the object from Semantic Contract as the basis
        - Supplement with ontology attributes, relationships, and rules
        - CANNOT override the action (which stays from Semantic Contract)

        Args:
            semantic_contract: The Semantic Contract from Step 1
            intent_result: IntentRecognitionResult from Step 1
            composite_result: Optional CompositeIntentResult

        Returns:
            OntologyResolveResult with ontology expansion applied
        """
        query = self._get_query()

        # 1. Find the ontology object based on Semantic Contract's object
        object_type = semantic_contract.object
        object_label = semantic_contract.object_label

        # Try to find the entity in ontology
        dataset = query.find_object_by_term(object_type)
        if not dataset:
            dataset = query.find_object_by_term(object_label)
        if not dataset:
            # Try to find by the action in topology
            try:
                from ..intent_topology import IntentTopology
                topology = IntentTopology()
                path = topology.get_path_for_intent(semantic_contract.action)
                if path:
                    dataset = query.ontology.get_entity(path.object_id)
                    object_type = path.object_id
                    object_label = path.object_label or object_label
            except Exception:
                pass

        # 2. Get attributes and actions from ontology
        if dataset:
            object_type = dataset.name
            object_label = dataset.label
            attributes = query.get_entity_attributes(object_type)
            actions = query.get_actions_for_entity(object_type)
            action_list = [
                {
                    "id": a.id,
                    "name": a.name,
                    "kind": a.kind,
                    "operation": a.operation,
                    "description": a.description,
                }
                for a in actions
            ]
        else:
            # No ontology match - use Semantic Contract info directly
            attributes = []
            action_list = []

        # 3. Build ontology expansion data
        ontology_expansion = self._build_ontology_expansion(
            object_type, dataset, semantic_contract
        )

        # 4. Extract hit keywords
        hit_keywords = [semantic_contract.object_label]
        if semantic_contract.slots:
            hit_keywords.extend([s.display_value for s in semantic_contract.slots.values()])

        # 5. Assess completeness
        completeness = "full"
        gaps = []
        if not dataset:
            completeness = "partial"
            gaps.append({
                "gap_type": "missing_object",
                "description": f"本体中未找到对象 '{object_type}'",
                "suggestion": "请在本体中补充该业务对象的定义"
            })

        # 6. Build result - NOTE: we pass the Semantic Contract's action as available_action
        # This ensures the correct analytics action is preserved
        available_actions = action_list.copy()
        # Ensure the Semantic Contract's action is in the list
        contract_action = semantic_contract.action
        if contract_action not in [a.get("id") for a in available_actions]:
            available_actions.insert(0, {
                "id": contract_action,
                "name": semantic_contract.action_label,
                "kind": "analytics",
                "operation": "query",
                "description": f"执行 {semantic_contract.action_label}",
            })

        result = OntologyResolveResult(
            object_type=object_type,
            object_label=object_label,
            hit_keywords=hit_keywords,
            attributes=attributes,
            available_actions=available_actions,
            ontology_completeness=completeness,
            gaps=gaps,
            llm_inferred=False,
            llm_reasoning="从语义契约扩展（Phase 4）",
        )

        # Store expansion in result metadata for later use
        result.ontology_expansion = ontology_expansion

        return result

    def _build_ontology_expansion(
        self,
        object_type: str,
        dataset,
        semantic_contract: SemanticContract
    ) -> Dict[str, Any]:
        """Build ontology expansion data for the Semantic Contract.

        Supplements the Semantic Contract with:
        - Entity fields (for parameter mapping)
        - Business rules (for query filtering)
        - Relationships (for joining data)
        - Default values (for missing parameters)
        """
        expansion = {
            "fields": [],
            "rules": [],
            "relationships": [],
            "defaults": {},
        }

        if not dataset:
            return expansion

        # 1. Extract entity fields
        if hasattr(dataset, 'fields'):
            for field in dataset.fields:
                expansion["fields"].append({
                    "name": field.name,
                    "type": getattr(field, 'type', 'string'),
                    "description": getattr(field, 'description', ''),
                    "required": getattr(field, 'required', False),
                })

        # 2. Add business rules from semantics
        # Default rule: exclude deleted records
        expansion["rules"].append({
            "id": "deleted_records_excluded_by_default",
            "description": "默认排除已删除记录",
            "field": "delete_flag",
            "operator": "=",
            "value": "0",
        })

        # 3. Map slot display values to field values
        if semantic_contract.slots:
            field_mapping = {}
            for slot_name, slot in semantic_contract.slots.items():
                # Map common slot names to fields
                field_map = {
                    "department": "apply_dep",
                    "apply_dep": "apply_dep",
                    "date_range": "delivery_date",
                    "time_date_from": "delivery_date",
                    "time_date_to": "delivery_date",
                    "execution_status": "flow_status",
                }
                field_name = field_map.get(slot_name, slot_name)
                field_mapping[slot_name] = {
                    "field_name": field_name,
                    "display_value": slot.display_value,
                    "source": slot.source,
                }
            expansion["field_mapping"] = field_mapping

        # 4. Add default values for missing parameters
        if semantic_contract.action == "analytics/find_unexecuted_purchase_requests":
            # For unexecuted PR query, we need to know the status mapping
            expansion["defaults"]["flow_status"] = "UNEXECUTED"
            expansion["rules"].append({
                "id": "unexecuted_pr_filter",
                "description": "筛选未执行的采购需求",
                "impl": "purchase_requests without corresponding purchase_order_items",
            })

        return expansion

    def _build_success_result(self, dataset, intent_result, query: OntologyQuery) -> OntologyResolveResult:
        """Build successful resolution result."""
        object_type = dataset.name
        object_label = dataset.label

        # Get attributes from the dataset
        attributes = query.get_entity_attributes(object_type)

        # Get available actions for this entity
        actions = query.get_actions_for_entity(object_type)
        action_list = [
            {
                "id": a.id,
                "name": a.name,
                "kind": a.kind,
                "operation": a.operation,
                "description": a.description,
            }
            for a in actions
        ]

        # Extract keywords from user input
        keywords = self._extract_keywords(intent_result.object_term, dataset)

        # Assess completeness
        completeness, gaps = self._assess_completeness(dataset, intent_result)

        return OntologyResolveResult(
            object_type=object_type,
            object_label=object_label,
            hit_keywords=keywords,
            attributes=attributes,
            available_actions=action_list,
            ontology_completeness=completeness,
            gaps=gaps,
        )

    def _llm_resolve_fallback_sync(
        self, 
        user_input: str, 
        intent_result,
        query: OntologyQuery
    ) -> tuple:
        """LLM-powered fallback for complex object resolution.
        
        Uses a thinking-enabled LLM call to reason about which ontology
        object best matches the user's query.
        
        Returns:
            tuple: (matched_object, llm_reasoning, alternatives)
        """
        try:
            from ..models import get_default_model
            
            # Build context with ontology summary
            ontology_context = query.ontology.to_system_prompt_context()
            
            prompt = _step2_ontology_matching_prompt().format(
                user_input=user_input,
                intent_id=intent_result.intent,
                intent_label=intent_result.intent_label,
                object_term=intent_result.object_term,
                operation_type=intent_result.operation_type,
                ontology_context=ontology_context,
            )
            model = get_default_model()
            response = model.generate(prompt, thinking_budget=800)
            
            # Parse the JSON response
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                matched_name = result.get("matched_name")
                reasoning = result.get("reasoning", "")
                confidence = result.get("confidence", 0.5)
                alternatives = result.get("alternatives", [])
                
                if matched_name and confidence >= 0.6:
                    dataset = query.ontology.get_entity(matched_name)
                    if dataset:
                        alt_list = [
                            {"name": a.get("name", ""), "reason": a.get("reason", "")}
                            for a in alternatives
                        ]
                        return dataset, reasoning, alt_list
            
        except Exception as e:
            print(f"[Step2] LLM fallback failed: {e}")
        
        return None, None, []

    def _extract_keywords(self, object_term: str, dataset) -> List[str]:
        """Extract keywords from user input and dataset."""
        keywords = []
        
        # Add dataset keywords
        if hasattr(dataset, 'keywords'):
            keywords.extend(dataset.keywords[:5])
        
        # Add dataset synonyms
        if hasattr(dataset, 'synonyms'):
            keywords.extend(dataset.synonyms[:3])
        
        return list(set(keywords)) if keywords else [object_term or "业务查询"]

    def _assess_completeness(self, dataset, intent_result) -> tuple:
        """Assess ontology completeness and detect gaps."""
        gaps = []
        
        # Check for essential fields
        essential_fields = ['status', 'delete_flag', 'pr_id']
        existing_fields = {f.name for f in dataset.fields}
        
        for field_name in essential_fields:
            if field_name not in existing_fields:
                gaps.append({
                    "gap_type": "missing_attribute",
                    "description": f"本体对象 '{dataset.name}' 缺少属性 '{field_name}'",
                    "suggestion": f"建议在 {dataset.name} 中补充 {field_name} 字段"
                })

        # Assess completeness
        if len(gaps) == 0:
            completeness = "full"
        elif len(gaps) <= 2:
            completeness = "partial"
        else:
            completeness = "insufficient"

        return completeness, gaps

    def _fallback_resolution(self, intent_result) -> OntologyResolveResult:
        """Fallback when no ontology match is found."""
        return OntologyResolveResult(
            object_type="unknown",
            object_label=intent_result.object_term or "未知对象",
            hit_keywords=[],
            attributes=[],
            available_actions=[],
            ontology_completeness="insufficient",
            gaps=[{
                "gap_type": "missing_object",
                "description": f"未找到 '{intent_result.object_term}' 对应的本体对象",
                "suggestion": "请确认本体中已定义该业务对象，或意图识别结果是否准确"
            }],
        )

    def _build_result_from_composite(self, composite_result, intent_result) -> OntologyResolveResult:
        """Build OntologyResolveResult from composite pipeline result.

        Reuses the topology match from Layer 2 (ontology match) of the composite pipeline.

        Args:
            composite_result: CompositeIntentResult from Step1's composite pipeline
            intent_result: IntentRecognitionResult from Step1

        Returns:
            OntologyResolveResult populated from composite layer results
        """
        # Get layer 2 ontology match results
        layer_2_result = composite_result.layer_results.get("layer_2_ontology_match", {})

        # Extract object info from composite result
        top_candidate = composite_result.top_candidate
        object_type = layer_2_result.get("matched_object_type", top_candidate.intent_id if top_candidate else "unknown")
        object_label = layer_2_result.get("matched_object_label", intent_result.object_term or "未知对象")

        # Try to find the IntentPath from topology for attributes and actions
        try:
            from ..intent_topology import IntentTopology
            topology = IntentTopology()
            path = topology.get_path_for_intent(top_candidate.intent_id if top_candidate else intent_result.intent)
            if path:
                object_type = path.object_id
                object_label = path.object_label or object_label
        except Exception:
            pass  # Fall back to values from composite

        query = self._get_query()

        # Get attributes and actions from ontology
        attributes = query.get_entity_attributes(object_type)
        actions = query.get_actions_for_entity(object_type)
        action_list = [
            {
                "id": a.id,
                "name": a.name,
                "kind": a.kind,
                "operation": a.operation,
                "description": a.description,
            }
            for a in actions
        ]

        # Extract hit keywords
        hit_keywords = []
        if layer_2_result.get("hit_keywords"):
            hit_keywords = layer_2_result["hit_keywords"]
        elif top_candidate and top_candidate.params:
            hit_keywords = [top_candidate.params.get("object_term", "")]

        # Determine if LLM inference was used (deep reasoning path)
        llm_inferred = layer_2_result.get("from_deep_reasoning", False) or \
                       composite_result.confidence.confidence_b > 0.9

        # Assess completeness
        completeness, gaps = self._assess_completeness_from_composite(object_type, layer_2_result)

        return OntologyResolveResult(
            object_type=object_type,
            object_label=object_label,
            hit_keywords=hit_keywords,
            attributes=attributes,
            available_actions=action_list,
            ontology_completeness=completeness,
            gaps=gaps,
            llm_inferred=llm_inferred,
            llm_reasoning=layer_2_result.get("reasoning"),
        )

    def _assess_completeness_from_composite(self, object_type: str, layer_2_result: dict) -> tuple:
        """Assess ontology completeness from composite layer 2 results."""
        gaps = []

        # Check what was matched
        if not layer_2_result.get("matched_object_type"):
            gaps.append({
                "gap_type": "missing_object",
                "description": f"复合管道未能匹配对象类型 '{object_type}'",
                "suggestion": "请确认本体中已定义该业务对象"
            })

        # Assess completeness
        if len(gaps) == 0:
            completeness = "full"
        elif len(gaps) <= 2:
            completeness = "partial"
        else:
            completeness = "insufficient"

        return completeness, gaps
