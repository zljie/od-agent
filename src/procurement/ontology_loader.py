"""
采购本体加载器
解析 procurement_s2a_semantic_model.yaml YAML 文件，构建结构化知识图谱
作为采购场景下 Agent 的唯一本体源，在启动时注入为持久化知识图谱
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# 默认本体路径
DEFAULT_ONTOLOGY_PATH = Path(__file__).parent.parent.parent / "config" / "ontology" / "procurement_s2a_semantic_model.yaml"


@dataclass
class Field:
    name: str
    type: str
    description: str
    expression: Optional[str] = None


@dataclass
class DataSet:
    name: str
    code: str
    description: str
    primary_key: List[str]
    label: str = ""  # Display label, derived from name if not specified
    fields: List[Field] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class Relationship:
    name: str
    from_entity: str
    to_entity: str
    from_columns: List[str]
    to_columns: List[str]
    description: str


@dataclass
class Metric:
    name: str
    expression: str
    description: str
    depends_on: List[str] = field(default_factory=list)


@dataclass
class IOSchema:
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


@dataclass
class Action:
    id: str
    name: str
    kind: str
    operation: str
    entity_name: str
    description: str
    input_schema: Optional[IOSchema] = None
    labels: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)


@dataclass
class Rule:
    id: str
    name: str
    severity: str
    when: Dict[str, str]
    message: str
    remediation: str


@dataclass
class Scenario:
    name: str
    mode: Optional[str]
    description: str
    key_functions: Optional[str]
    prompt_example: Optional[str]


@dataclass
class ProcurementOntology:
    version: str
    datasets: Dict[str, DataSet] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    metrics: Dict[str, Metric] = field(default_factory=dict)
    actions: Dict[str, Action] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    scenarios: List[Scenario] = field(default_factory=list)
    ai_context: Dict[str, Any] = field(default_factory=dict)
    instructions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def get_entity(self, name: str) -> Optional[DataSet]:
        return self.datasets.get(name)

    def get_action(self, action_id: str) -> Optional[Action]:
        for action in self.actions:
            if action.id == action_id:
                return action
        return None

    def get_actions_for_entity(self, entity_name: str) -> List[Action]:
        return [a for a in self.actions if a.entity_name == entity_name]

    def get_relationship(self, from_entity: str, to_entity: str) -> Optional[Relationship]:
        for rel in self.relationships:
            if rel.from_entity == from_entity and rel.to_entity == to_entity:
                return rel
        return None

    def get_scenario(self, name: str) -> Optional[Scenario]:
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        return None

    def get_metrics_for_dataset(self, dataset_name: str) -> List[Metric]:
        return [m for m in self.metrics.values() if dataset_name in m.depends_on]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "datasets": {
                name: {
                    "name": ds.name,
                    "code": ds.code,
                    "description": ds.description,
                    "primary_key": ds.primary_key,
                    "synonyms": ds.synonyms,
                    "keywords": ds.keywords,
                    "fields": [
                        {
                            "name": f.name,
                            "type": f.type,
                            "description": f.description
                        } for f in ds.fields
                    ]
                } for name, ds in self.datasets.items()
            },
            "relationships": [
                {
                    "name": r.name,
                    "from": r.from_entity,
                    "to": r.to_entity,
                    "description": r.description
                } for r in self.relationships
            ],
            "actions": [
                {
                    "id": a.id,
                    "name": a.name,
                    "kind": a.kind,
                    "operation": a.operation,
                    "entity_name": a.entity_name,
                    "description": a.description,
                    "synonyms": a.synonyms
                } for a in self.actions
            ],
            "scenarios": [
                {
                    "name": s.name,
                    "description": s.description
                } for s in self.scenarios
            ],
            "ai_context": self.ai_context
        }

    def to_system_prompt_context(self) -> str:
        """生成本体摘要文本，用于注入到 Agent 的 system prompt"""
        lines = [
            f"# 采购业务本体 (版本 {self.version})",
            "",
            "## 业务口径说明",
        ]
        
        if self.instructions:
            for inst in self.instructions:
                lines.append(f"- {inst}")
        
        lines.extend([
            "",
            "## 业务对象",
        ])
        
        for name, ds in self.datasets.items():
            lines.append(f"\n### {ds.label} ({name})")
            lines.append(f"{ds.description}")
            if ds.synonyms:
                lines.append(f"同义词: {', '.join(ds.synonyms)}")
            if ds.keywords:
                lines.append(f"关键词: {', '.join(ds.keywords)}")
            
            # 列出主要字段
            display_fields = [f.name for f in ds.fields[:8]]
            if display_fields:
                lines.append(f"主要字段: {', '.join(display_fields)}")
        
        lines.extend([
            "",
            "## 可用操作",
        ])
        
        for action in self.actions:
            lines.append(f"- **{action.name}** (`{action.id}`)")
            lines.append(f"  适用于: {action.entity_name} | 类型: {action.kind}")
        
        lines.extend([
            "",
            "## 业务规则",
        ])
        
        for rule in self.rules[:5]:  # 只显示前5条规则
            lines.append(f"- {rule.name}: {rule.message}")
        
        return "\n".join(lines)


def _parse_fields(fields_data: List[Dict]) -> List[Field]:
    fields = []
    for f in fields_data:
        expression = None
        if "expression" in f:
            expr_data = f["expression"]
            if isinstance(expr_data, dict) and "dialects" in expr_data:
                dialects = expr_data["dialects"]
                if isinstance(dialects, list) and len(dialects) > 0:
                    expression = dialects[0].get("expression")
            elif isinstance(expr_data, str):
                expression = expr_data

        fields.append(Field(
            name=f["name"],
            type=f["type"],
            description=f.get("description", ""),
            expression=expression
        ))
    return fields


def _parse_io_schema(schema_data: Optional[Dict]) -> Optional[IOSchema]:
    if not schema_data:
        return None
    properties = schema_data.get("properties", schema_data.get("input", {}))
    required = schema_data.get("required", [])
    return IOSchema(
        type="object",
        properties=properties,
        required=required
    )


# Name to label mapping for common entities
_NAME_TO_LABEL = {
    "purchase_requests": "采购需求",
    "purchase_inquiries": "询价单",
    "purchase_quotations": "报价单",
    "purchase_order_heads": "采购订单抬头",
    "purchase_order_items": "采购订单行项目",
    "purchase_order_receipt_history": "收货历史",
}

# Additional Chinese synonyms that should map to entities (not in YAML synonyms)
_EXTENDED_SYNONYMS = {
    "purchase_requests": ["采购计划", "采购单", "请购单", "PR"],
    "purchase_inquiries": ["询价请求", "询价采购", "RFQ请求"],
    "purchase_quotations": ["供应商报价", "报价", "quotation"],
    "purchase_order_heads": ["采购单", "PO", "订单抬头"],
    "purchase_order_items": ["采购明细", "订单行"],
}


def _derive_label_from_name(name: str) -> str:
    """Derive a display label from entity name."""
    return _NAME_TO_LABEL.get(name, name.replace("_", " ").title())


def load_ontology_from_yaml(path: str = None) -> ProcurementOntology:
    """从YAML文件加载采购本体"""
    ontology_path = Path(path) if path else DEFAULT_ONTOLOGY_PATH
    
    if not ontology_path.exists():
        raise FileNotFoundError(f"本体文件不存在: {ontology_path}")
    
    with open(ontology_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    ontology = ProcurementOntology(version=data.get("version", "0.1.2"))

    semantic_model = data.get("semantic_model", [])
    if semantic_model:
        model = semantic_model[0]

        # AI context
        if "ai_context" in model:
            ai_ctx = model["ai_context"]
            ontology.ai_context = ai_ctx
            
            # 解析 instructions
            if isinstance(ai_ctx, dict):
                if "instructions" in ai_ctx:
                    ontology.instructions = ai_ctx["instructions"]
                if "examples" in ai_ctx:
                    ontology.examples = ai_ctx["examples"]

        # 解析 datasets
        for ds_data in model.get("datasets", []):
            # 提取 synonyms 和 keywords
            ai_ctx_ds = ds_data.get("ai_context", {})
            synonyms = ai_ctx_ds.get("synonyms", []) if isinstance(ai_ctx_ds, dict) else []
            keywords = ai_ctx_ds.get("keywords", []) if isinstance(ai_ctx_ds, dict) else []
            
            # Derive label from name (e.g., "purchase_requests" -> "采购需求")
            name = ds_data["name"]
            label = _derive_label_from_name(name)
            
            # Base synonyms from YAML
            synonyms = ai_ctx_ds.get("synonyms", []) if isinstance(ai_ctx_ds, dict) else []
            # Add extended synonyms (e.g., "采购计划" -> "purchase_requests")
            synonyms.extend(_EXTENDED_SYNONYMS.get(name, []))
            
            # Base keywords from YAML
            keywords = ai_ctx_ds.get("keywords", []) if isinstance(ai_ctx_ds, dict) else []
            
            ds = DataSet(
                name=name,
                code=name,
                label=label,
                description=ds_data.get("description", ""),
                primary_key=ds_data.get("primary_key", []),
                fields=_parse_fields(ds_data.get("fields", [])),
                synonyms=synonyms,
                keywords=keywords
            )
            ontology.datasets[ds.name] = ds

        # 解析 relationships
        for rel_data in model.get("relationships", []):
            rel = Relationship(
                name=rel_data["name"],
                from_entity=rel_data["from"],
                to_entity=rel_data["to"],
                from_columns=rel_data.get("from_columns", []),
                to_columns=rel_data.get("to_columns", []),
                description=rel_data.get("description", "")
            )
            ontology.relationships.append(rel)

        # 解析 metrics
        for metric_data in model.get("metrics", []):
            depends_on = []
            extensions = metric_data.get("custom_extensions", [])
            for ext in extensions:
                if ext.get("vendor_name") == "COMMON":
                    depends_raw = ext.get("data", "{}")
                    if isinstance(depends_raw, str):
                        import json
                        try:
                            depends_data = json.loads(depends_raw)
                            depends_on = depends_data.get("depends_on_datasets", [])
                        except:
                            pass

            expr_data = metric_data.get("expression", {})
            expr_str = ""
            if isinstance(expr_data, dict) and "dialects" in expr_data:
                dialects = expr_data["dialects"]
                if isinstance(dialects, list) and len(dialects) > 0:
                    expr_str = dialects[0].get("expression", "")

            metric = Metric(
                name=metric_data["name"],
                expression=expr_str,
                description=metric_data.get("description", ""),
                depends_on=depends_on
            )
            ontology.metrics[metric.name] = metric

        # 解析 actions
        for action_data in model.get("behavior", {}).get("actions", []):
            io_schema_data = action_data.get("io_schema", {})
            input_schema_data = io_schema_data.get("input", io_schema_data.get("input_schema"))
            
            # 提取 synonyms
            synonyms = action_data.get("synonyms", [])
            
            action = Action(
                id=action_data["id"],
                name=action_data["name"],
                kind=action_data.get("kind", "query"),
                operation=action_data.get("operation", "read"),
                entity_name=action_data.get("entity_name", ""),
                description=action_data.get("description", ""),
                input_schema=_parse_io_schema(input_schema_data),
                labels=action_data.get("labels", []),
                synonyms=synonyms
            )
            ontology.actions.append(action)

        # 解析 rules
        for rule_data in model.get("behavior", {}).get("rules", []):
            rule = Rule(
                id=rule_data["id"],
                name=rule_data["name"],
                severity=rule_data.get("severity", "warn"),
                when=rule_data.get("when", {}),
                message=rule_data.get("message", ""),
                remediation=rule_data.get("remediation", "")
            )
            ontology.rules.append(rule)

        # 解析 scenarios
        for scenario_data in model.get("behavior", {}).get("scenarios", []):
            scenario = Scenario(
                name=scenario_data["name"],
                mode=scenario_data.get("mode"),
                description=scenario_data.get("description", ""),
                key_functions=scenario_data.get("key_functions"),
                prompt_example=scenario_data.get("prompt_example")
            )
            ontology.scenarios.append(scenario)

    return ontology


# ==================== Global Singleton ====================

_ontology_instance: Optional[ProcurementOntology] = None


def get_procurement_ontology(force_reload: bool = False) -> ProcurementOntology:
    """获取采购本体单例"""
    global _ontology_instance
    if _ontology_instance is None or force_reload:
        _ontology_instance = load_ontology_from_yaml(str(DEFAULT_ONTOLOGY_PATH))
    return _ontology_instance


def reload_ontology() -> ProcurementOntology:
    """重新加载本体"""
    return get_procurement_ontology(force_reload=True)


# ==================== 本体查询工具 ====================

class OntologyQuery:
    """本体查询接口，供 Step2 等模块使用"""
    
    def __init__(self, ontology: ProcurementOntology = None):
        self.ontology = ontology or get_procurement_ontology()
    
    def find_object_by_term(self, term: str) -> Optional[DataSet]:
        """根据术语查找匹配的本体对象（dataset）
        
        支持的匹配策略：
        1. 精确匹配 dataset name
        2. 精确匹配 dataset label
        3. 同义词匹配
        4. 关键词匹配
        5. 模糊匹配（包含关系）
        """
        term_lower = term.lower().strip()
        
        # 1. 精确匹配 name
        if term_lower in self.ontology.datasets:
            return self.ontology.datasets[term_lower]
        
        # 2. 精确匹配 label
        for ds in self.ontology.datasets.values():
            if ds.name.lower() == term_lower or ds.label.lower() == term_lower:
                return ds
        
        # 3. 同义词匹配
        for ds in self.ontology.datasets.values():
            for syn in ds.synonyms:
                if syn.lower() == term_lower:
                    return ds
        
        # 4. 关键词匹配
        for ds in self.ontology.datasets.values():
            for kw in ds.keywords:
                if kw.lower() == term_lower:
                    return ds
        
        # 5. 模糊匹配
        for ds in self.ontology.datasets.values():
            if (term_lower in ds.name.lower() or 
                term_lower in ds.label.lower() or
                any(term_lower in syn.lower() for syn in ds.synonyms)):
                return ds
        
        return None
    
    def find_object_by_intent(self, intent_id: str) -> Optional[DataSet]:
        """根据意图 ID 查找对应的本体对象"""
        action = self.ontology.get_action(intent_id)
        if action:
            return self.ontology.get_entity(action.entity_name)
        return None
    
    def get_entity_attributes(self, entity_name: str, usage: str = None) -> List[Dict[str, Any]]:
        """获取实体的属性列表"""
        ds = self.ontology.get_entity(entity_name)
        if not ds:
            return []
        
        attrs = []
        for field in ds.fields:
            attr = {
                "name": field.name,
                "type": field.type,
                "description": field.description,
                "expression": field.expression
            }
            if usage:
                attr["usage"] = usage
            attrs.append(attr)
        
        return attrs
    
    def get_actions_for_entity(self, entity_name: str, kind: str = None) -> List[Action]:
        """获取实体对应的操作列表"""
        actions = self.ontology.get_actions_for_entity(entity_name)
        if kind:
            actions = [a for a in actions if a.kind == kind]
        return actions
    
    def get_action_by_id(self, action_id: str) -> Optional[Action]:
        """根据 action_id 获取操作定义"""
        return self.ontology.get_action(action_id)
    
    def find_action_by_operation(self, operation: str, entity_name: str = None) -> List[Action]:
        """根据操作类型查找操作"""
        results = []
        for action in self.ontology.actions:
            if action.operation == operation:
                if entity_name is None or action.entity_name == entity_name:
                    results.append(action)
        return results
    
    def get_related_entities(self, entity_name: str) -> List[Dict[str, Any]]:
        """获取与指定实体相关的其他实体"""
        related = []
        for rel in self.ontology.relationships:
            if rel.from_entity == entity_name:
                related.append({
                    "name": rel.name,
                    "target": rel.to_entity,
                    "type": "outgoing",
                    "description": rel.description
                })
            elif rel.to_entity == entity_name:
                related.append({
                    "name": rel.name,
                    "target": rel.from_entity,
                    "type": "incoming",
                    "description": rel.description
                })
        return related
    
    def get_business_rules_for_action(self, action_id: str) -> List[Rule]:
        """获取操作对应的业务规则"""
        rules = []
        for rule in self.ontology.rules:
            when = rule.when
            if isinstance(when, dict):
                if when.get("action") == action_id:
                    rules.append(rule)
        return rules
    
    def to_context_for_llm(self) -> str:
        """生成本体上下文文本，用于 LLM 思考"""
        return self.ontology.to_system_prompt_context()
