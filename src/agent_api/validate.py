"""
Parameter Validation — POST /agent/tools/{toolName}/validate

Validates input parameters against the tool's input schema.
Reuses SlotCompletionEngine for slot-level validation.
"""

from typing import Any, Dict, List, Optional

from .manifest import get_tool_registry, ToolManifest


class ValidationResult:
    """Result of a parameter validation call."""

    def __init__(
        self,
        valid: bool,
        missing_slots: Optional[List[Dict[str, Any]]] = None,
        warnings: Optional[List[str]] = None,
        errors: Optional[List[Dict[str, str]]] = None,
        readiness_score: float = 100.0,
        schema_found: bool = True,
    ):
        self.valid = valid
        self.missing_slots = missing_slots or []
        self.warnings = warnings or []
        self.errors = errors or []
        self.readiness_score = readiness_score
        self.schema_found = schema_found

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "missing_slots": self.missing_slots,
            "warnings": self.warnings,
            "errors": self.errors,
            "readiness_score": self.readiness_score,
            "schema_found": self.schema_found,
        }


def validate_input(
    tool_name: str,
    input_params: Dict[str, Any],
    session_id: Optional[str] = None,
) -> ValidationResult:
    """
    Validate input parameters against a tool's input schema.

    Parameters
    ----------
    tool_name:
        The tool name to validate against (e.g., "create_purchase_request").
    input_params:
        The input parameter dictionary provided by the caller.
    session_id:
        Optional session ID for logging/tracing.

    Returns
    -------
    ValidationResult
        Contains valid flag, missing_slots list, warnings, errors.
    """
    registry = get_tool_registry()
    manifest = registry.get_tool_manifest(tool_name)

    if manifest is None:
        return ValidationResult(
            valid=False,
            schema_found=False,
            errors=[{"code": "TOOL_NOT_FOUND", "message": f"Tool '{tool_name}' not found"}],
        )

    schema = manifest.input_schema
    required_fields = schema.required

    # ── 1. Check required fields ──────────────────────────────────────────
    missing_slots: List[Dict[str, Any]] = []
    for field_name in required_fields:
        value = input_params.get(field_name)
        if value is None or value == "" or value == []:
            field_def = schema.properties.get(field_name)
            missing_slots.append(
                {
                    "field": field_name,
                    "title": field_def.title if field_def else field_name,
                    "description": field_def.description if field_def else "",
                    "required": True,
                    "question": _build_question(field_name, field_def),
                    "type": field_def.type if field_def else "string",
                }
            )

    # ── 2. Type / format checks ──────────────────────────────────────────
    errors: List[Dict[str, str]] = []
    warnings: List[str] = []

    for field_name, field_def in schema.properties.items():
        value = input_params.get(field_name)
        if value is None or value == "":
            continue

        # Number type check
        if field_def.type == "number":
            try:
                num_val = float(value)
                if field_def.minimum is not None and num_val < field_def.minimum:
                    errors.append(
                        {
                            "code": "VALIDATION_ERROR",
                            "field": field_name,
                            "message": f"{field_def.title} 必须 >= {field_def.minimum}",
                        }
                    )
            except (TypeError, ValueError):
                errors.append(
                    {
                        "code": "TYPE_ERROR",
                        "field": field_name,
                        "message": f"{field_def.title} 必须是数字",
                    }
                )

        # Enum check
        if field_def.enum_values:
            if value not in field_def.enum_values:
                warnings.append(
                    f"{field_def.title} 的值 '{value}' 不在推荐范围内，"
                    f"推荐值：{', '.join(field_def.enum_values)}"
                )

    # ── 3. Compute readiness score ───────────────────────────────────────
    total_required = len(required_fields)
    if total_required == 0:
        readiness_score = 100.0
    else:
        filled_required = total_required - len(missing_slots)
        readiness_score = round(filled_required / total_required * 100, 1)

    valid = len(missing_slots) == 0 and len(errors) == 0

    return ValidationResult(
        valid=valid,
        missing_slots=missing_slots,
        warnings=warnings,
        errors=errors,
        readiness_score=readiness_score,
        schema_found=True,
    )


def _build_question(field_name: str, field_def: Any) -> str:
    """Generate a natural-language question for a missing required field."""
    if field_def is None:
        return f"请提供 {field_name}"

    prompts = {
        "department": "请提供申请部门（例如：销售部、研发部）",
        "item_name": "请提供需要采购的物料名称",
        "quantity": "请提供采购数量",
        "delivery_date": "请提供期望交付日期（格式：YYYY-MM-DD）",
        "po_id": "请提供采购订单编号（例如：PO-20260604-001）",
        "pr_id": "请提供采购需求编号（例如：PR-20260604-001）",
        "quotation_id": "请提供报价单编号",
        "inquiry_id": "请提供询价单编号",
        "material_category": "请选择物料分类",
        "pr_type": "请选择采购类型",
        "decision": "请说明您的决定（通过 或 驳回）",
        "approver": "请指定审批人",
    }
    return prompts.get(field_name, f"请提供 {field_def.title or field_name}")
