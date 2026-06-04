"""
Slot Extractor
============
Extracts execution parameters from user input and checks slot completeness.

Based on Section 5.5 of BeBIOS_Intent_Processing_Architecture_v1.0.md

Note:
    Date/time extraction uses TemporalParser from src.temporal for consistent
    date resolution across the pipeline.

Example:
    extractor = SlotExtractor()
    result = extractor.extract(
        text="帮我查询销售部的采购需求",
        action="Query",
        object="PurchaseRequirement"
    )
    print(result.slots)  # {"department": "销售部"}
    print(result.missing_slots)  # []
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Slot Definition
# ---------------------------------------------------------------------------

@dataclass
class SlotDefinition:
    """Definition of a required or optional slot."""
    name: str                      # Slot name (e.g., "department")
    label: str                     # Human-readable label (e.g., "部门")
    slot_type: str = "string"      # Type: string, number, date, enum
    required: bool = False         # Whether this slot is required
    default: Any = None            # Default value
    description: str = ""           # Description for clarification
    placeholder: str = ""          # Placeholder text for user input
    enum_values: List[str] = field(default_factory=list)  # For enum type
    extraction_keywords: List[str] = field(default_factory=list)  # Keywords that indicate this slot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "slotType": self.slot_type,
            "required": self.required,
            "default": self.default,
            "description": self.description,
            "placeholder": self.placeholder or f"请输入{self.label}",
            "enumValues": self.enum_values,
            "extractionKeywords": self.extraction_keywords,
        }


# ---------------------------------------------------------------------------
# Default Slot Definitions by Object
# ---------------------------------------------------------------------------

DEFAULT_SLOT_DEFINITIONS: Dict[str, List[SlotDefinition]] = {
    "PurchaseRequirement": [
        SlotDefinition(
            name="department",
            label="部门",
            slot_type="string",
            required=False,
            description="申请部门",
            extraction_keywords=["部门", "哪个部门", "什么部门", "哪个部门的"],
        ),
        SlotDefinition(
            name="material",
            label="物料",
            slot_type="string",
            required=False,
            description="物料名称或编码",
            extraction_keywords=["物料", "什么东西", "采购什么"],
        ),
        SlotDefinition(
            name="quantity",
            label="数量",
            slot_type="number",
            required=False,
            description="采购数量",
            extraction_keywords=["数量", "多少", "几"],
        ),
        SlotDefinition(
            name="execution_status",
            label="执行状态",
            slot_type="enum",
            required=False,
            enum_values=["NOT_EXECUTED", "PARTIAL_EXECUTED", "EXECUTED"],
            description="执行状态",
            extraction_keywords=["状态", "执行情况"],
        ),
        SlotDefinition(
            name="date_range",
            label="日期范围",
            slot_type="date",
            required=False,
            description="查询的日期范围",
            extraction_keywords=["日期", "时间", "什么时候"],
        ),
    ],
    "Inquiry": [
        SlotDefinition(
            name="material",
            label="物料",
            slot_type="string",
            required=False,
            description="询价的物料",
        ),
        SlotDefinition(
            name="quantity",
            label="数量",
            slot_type="number",
            required=False,
            description="采购数量",
        ),
        SlotDefinition(
            name="vendor",
            label="供应商",
            slot_type="string",
            required=False,
            description="询价供应商",
        ),
    ],
    "Quotation": [
        SlotDefinition(
            name="inquiry_id",
            label="询价单号",
            slot_type="string",
            required=False,
            description="关联的询价单",
        ),
        SlotDefinition(
            name="vendor",
            label="供应商",
            slot_type="string",
            required=False,
            description="报价供应商",
        ),
    ],
    "PurchaseOrder": [
        SlotDefinition(
            name="pr_id",
            label="采购需求单号",
            slot_type="string",
            required=False,
            description="关联的采购需求单",
            extraction_keywords=["需求单", "采购需求"],
        ),
        SlotDefinition(
            name="vendor",
            label="供应商",
            slot_type="string",
            required=False,
            description="订单供应商",
        ),
        SlotDefinition(
            name="delivery_date",
            label="交货日期",
            slot_type="date",
            required=False,
            description="预计交货日期",
            extraction_keywords=["交货", "什么时候到", "交期"],
        ),
    ],
    "Supplier": [
        SlotDefinition(
            name="name",
            label="供应商名称",
            slot_type="string",
            required=False,
            description="供应商名称",
        ),
        SlotDefinition(
            name="category",
            label="供应商类别",
            slot_type="enum",
            required=False,
            enum_values=["MATERIAL", "SERVICE", "EQUIPMENT"],
            description="供应商类别",
        ),
    ],
    "Material": [
        SlotDefinition(
            name="code",
            label="物料编码",
            slot_type="string",
            required=False,
            description="物料编码",
        ),
        SlotDefinition(
            name="name",
            label="物料名称",
            slot_type="string",
            required=False,
            description="物料名称",
        ),
        SlotDefinition(
            name="category",
            label="物料类别",
            slot_type="string",
            required=False,
            description="物料类别",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Extraction Result
# ---------------------------------------------------------------------------

@dataclass
class SlotExtractionResult:
    """Result of slot extraction."""
    object: str                          # Ontology object
    action: str                           # Action intent
    slots: Dict[str, Any]               # Extracted slots {name: value}
    required_slots: List[str]            # List of required slot names
    filled_slots: List[str]              # List of filled slot names
    missing_slots: List[str]             # List of missing slot names
    optional_slots: List[str]            # List of optional slot names (filled)
    confidence: float = 1.0             # Extraction confidence
    status: str = "partial"             # "complete", "partial", "empty"
    definitions: List[SlotDefinition] = field(default_factory=list)  # Slot definitions used

    def __post_init__(self):
        if not self.required_slots:
            self.required_slots = [d.name for d in self.definitions if d.required]
        if not self.filled_slots:
            self.filled_slots = list(self.slots.keys())

    @property
    def is_complete(self) -> bool:
        """Check if all required slots are filled."""
        return len(self.missing_slots) == 0

    @property
    def completion_ratio(self) -> float:
        """Get completion ratio of required slots."""
        if not self.required_slots:
            return 1.0
        filled_required = [s for s in self.required_slots if s in self.slots]
        return len(filled_required) / len(self.required_slots)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object": self.object,
            "action": self.action,
            "slots": self.slots,
            "requiredSlots": self.required_slots,
            "filledSlots": self.filled_slots,
            "missingSlots": self.missing_slots,
            "optionalSlots": self.optional_slots,
            "confidence": self.confidence,
            "status": self.status,
            "completionRatio": self.completion_ratio,
            "isComplete": self.is_complete,
        }

    def to_frame_slots(self) -> Dict[str, Any]:
        """Convert to SemanticFrame slots format."""
        return {
            "object": self.object,
            "slots": self.slots,
            "slotFillStatus": self.status,
            "missingSlots": self.missing_slots,
            "completionRatio": self.completion_ratio,
        }


# ---------------------------------------------------------------------------
# Slot Extractor
# ---------------------------------------------------------------------------

class SlotExtractor:
    """Extracts execution parameters from user input.

    This implements Section 5.5 of the BeBIOS architecture.

    Example:
        extractor = SlotExtractor()
        result = extractor.extract(
            text="帮我查询销售部的采购需求",
            action="Query",
            object="PurchaseRequirement"
        )
    """

    def __init__(
        self,
        custom_definitions: Optional[Dict[str, List[SlotDefinition]]] = None,
    ):
        """Initialize the slot extractor.

        Args:
            custom_definitions: Override default slot definitions for objects
        """
        self.definitions = {**DEFAULT_SLOT_DEFINITIONS}
        if custom_definitions:
            for obj, slots in custom_definitions.items():
                if obj in self.definitions:
                    self.definitions[obj].extend(slots)
                else:
                    self.definitions[obj] = slots

    def get_definitions(self, object: str) -> List[SlotDefinition]:
        """Get slot definitions for an object."""
        return self.definitions.get(object, [])

    def extract(
        self,
        text: str,
        action: str,
        object: str,
        existing_slots: Optional[Dict[str, Any]] = None,
    ) -> SlotExtractionResult:
        """Extract slots from user input.

        Args:
            text: User input text
            action: Action intent (e.g., "Query", "Create")
            object: Ontology object (e.g., "PurchaseRequirement")
            existing_slots: Already extracted slots to merge

        Returns:
            SlotExtractionResult with extracted slots and completion status
        """
        existing = existing_slots or {}
        slots: Dict[str, Any] = dict(existing)

        # Get slot definitions for this object
        definitions = self.get_definitions(object)
        required_names = [d.name for d in definitions if d.required]
        optional_names = [d.name for d in definitions if not d.required]

        # Extract slots using keyword matching and LLM (if available)
        extracted = self._extract_from_text(text, definitions)

        # Merge extracted slots
        for name, value in extracted.items():
            if value and (name not in slots or not slots[name]):
                slots[name] = value

        # Determine filled and missing slots
        filled_required = [s for s in required_names if s in slots and slots[s]]
        missing_required = [s for s in required_names if s not in slots or not slots[s]]
        filled_optional = [s for s in optional_names if s in slots and slots[s]]

        # Determine status
        if not slots:
            status = "empty"
        elif not missing_required:
            status = "complete"
        else:
            status = "partial"

        # Calculate confidence
        confidence = self._calculate_confidence(
            len(filled_required), len(required_names),
            len(filled_optional), len(optional_names)
        )

        return SlotExtractionResult(
            object=object,
            action=action,
            slots=slots,
            required_slots=required_names,
            filled_slots=filled_required + filled_optional,
            missing_slots=missing_required,
            optional_slots=filled_optional,
            confidence=confidence,
            status=status,
            definitions=definitions,
        )

    def _extract_from_text(
        self,
        text: str,
        definitions: List[SlotDefinition],
    ) -> Dict[str, Any]:
        """Extract slot values from text using keyword matching.

        In production, this should use LLM-based extraction.
        """
        import re
        from datetime import datetime

        result: Dict[str, Any] = {}

        for definition in definitions:
            value = None

            # Try to extract based on slot type
            if definition.slot_type == "number":
                value = self._extract_number(text, definition)
            elif definition.slot_type == "date":
                value = self._extract_date(text, definition)
            elif definition.enum_values:
                value = self._extract_enum(text, definition)
            else:
                value = self._extract_string(text, definition)

            if value:
                result[definition.name] = value

        return result

    def _extract_number(self, text: str, definition: SlotDefinition) -> Optional[Any]:
        """Extract numeric value from text."""
        import re

        # Common number patterns
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:个|台|箱|件|套|份|张|批)?' + (definition.label or definition.name),
            r'(?:数量|多少)(?:是|有)?(\d+(?:\.\d+)?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    def _extract_date(self, text: str, definition: SlotDefinition) -> Optional[str]:
        """Extract date value from text.

        Uses TemporalParser for consistent date resolution across the pipeline.
        Falls back to simple patterns for basic date extraction.
        """
        # Use TemporalParser for consistent date resolution
        try:
            from ..temporal.temporal_parser import TemporalParser
            parser = TemporalParser()
            ctx = parser.parse(text)
            if ctx.dates:
                # Return the first resolved date in ISO format
                first_date = next(iter(ctx.dates.values()))
                return first_date.isoformat()
        except ImportError:
            pass  # Fall back to simple extraction

        # Simple fallback patterns
        from datetime import datetime, timedelta
        import re

        # ISO format
        iso_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
        if iso_match:
            return iso_match.group()

        # Relative dates - basic support
        now = datetime.now()
        if '今天' in text or '今日' in text:
            return now.strftime('%Y-%m-%d')
        if '明天' in text or '明日' in text:
            return (now + timedelta(days=1)).strftime('%Y-%m-%d')
        if '昨天' in text or '昨日' in text:
            return (now - timedelta(days=1)).strftime('%Y-%m-%d')

        # Week-related
        if '本周' in text:
            return now.strftime('%Y-%m-%d')
        if '下周' in text:
            return (now + timedelta(weeks=1)).strftime('%Y-%m-%d')

        # Month-related
        month_match = re.search(r'(\d{1,2})月', text)
        if month_match:
            month = int(month_match.group(1))
            return f'{now.year}-{month:02d}-01'

        return None

    def _extract_enum(self, text: str, definition: SlotDefinition) -> Optional[str]:
        """Extract enum value from text."""
        text_lower = text.lower()

        for enum_val in definition.enum_values:
            if enum_val.lower() in text_lower:
                return enum_val

        return None

    def _extract_string(self, text: str, definition: SlotDefinition) -> Optional[str]:
        """Extract string value using extraction keywords."""
        import re

        # Use extraction keywords to find values
        for keyword in definition.extraction_keywords:
            # Pattern: keyword + value
            pattern = f'{keyword}[是为：:]*([^\\s,，]+)'
            match = re.search(pattern, text)
            if match and match.group(1):
                return match.group(1)

            # Check if keyword exists, try to extract adjacent text
            if keyword in text:
                idx = text.index(keyword)
                # Get text after keyword
                after = text[idx + len(keyword):].strip()
                # Get next few characters as potential value
                value_match = re.match(r'[是为：:]*\\s*([^\\s,，。]+)', after)
                if value_match:
                    return value_match.group(1)

        return None

    def _calculate_confidence(
        self,
        filled_required: int,
        total_required: int,
        filled_optional: int,
        total_optional: int,
    ) -> float:
        """Calculate extraction confidence."""
        if total_required == 0 and total_optional == 0:
            return 1.0

        required_weight = 0.7
        optional_weight = 0.3

        required_ratio = filled_required / total_required if total_required > 0 else 1.0
        optional_ratio = filled_optional / total_optional if total_optional > 0 else 1.0

        return round(required_ratio * required_weight + optional_ratio * optional_weight, 2)

    def generate_clarification_question(
        self,
        result: SlotExtractionResult,
        object_name: str = "",
    ) -> str:
        """Generate a clarification question for missing slots."""
        if not result.missing_slots:
            return ""

        # Get labels for missing slots
        missing_labels = []
        for name in result.missing_slots:
            definition = next((d for d in result.definitions if d.name == name), None)
            if definition:
                label = definition.label or definition.name
                placeholder = definition.placeholder or f"请输入{label}"
                missing_labels.append(f"{label}（{placeholder}）")

        if not missing_labels:
            return "请提供更多信息"

        if object_name:
            return f"创建{object_name}还需要以下信息：\n" + "\n".join(f"{i+1}. {l}" for i, l in enumerate(missing_labels))
        else:
            return "请补充以下信息：\n" + "\n".join(f"{i+1}. {l}" for i, l in enumerate(missing_labels))


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_default_extractor: Optional[SlotExtractor] = None


def get_default_extractor() -> SlotExtractor:
    """Get the default extractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = SlotExtractor()
    return _default_extractor


def extract_slots(
    text: str,
    action: str,
    object: str,
    existing_slots: Optional[Dict[str, Any]] = None,
) -> SlotExtractionResult:
    """Convenience function to extract slots."""
    return get_default_extractor().extract(text, action, object, existing_slots)
