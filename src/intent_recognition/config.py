"""
Intent Recognition Configuration
=================================
Unified configuration for the intent recognition pipeline.

Usage:
    # Default configuration
    config = IntentRecognitionConfig.default()

    # Load from environment
    config = IntentRecognitionConfig.from_env()

    # Load from YAML/JSON file
    config = IntentRecognitionConfig.from_file("config.yaml")

    # Modify specific values
    config.layer3.threshold_execute = 0.9

    # Get current config as dict
    config.to_dict()
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Layer 3: Light LLM (A) + Ontology Match (B) Fusion
# ---------------------------------------------------------------------------

@dataclass
class Layer3FusionConfig:
    """Layer 3: Light LLM (A) + Ontology Match (B) fusion configuration."""

    # Fusion weights
    weight_a: float = 0.45  # LLM-light confidence weight
    weight_b: float = 0.55  # Ontology match confidence weight

    # Decision thresholds
    threshold_deep_reasoning: float = 0.75  # < this → deep reasoning
    threshold_continue: float = 0.75        # >= this → continue
    threshold_execute: float = 0.85        # >= this → execute directly

    def validate(self) -> None:
        """Validate configuration values."""
        assert 0 <= self.weight_a <= 1, f"weight_a must be in [0, 1], got {self.weight_a}"
        assert 0 <= self.weight_b <= 1, f"weight_b must be in [0, 1], got {self.weight_b}"
        assert abs(self.weight_a + self.weight_b - 1.0) < 0.01, \
            f"weights must sum to 1, got {self.weight_a} + {self.weight_b}"
        assert 0 <= self.threshold_deep_reasoning <= 1
        assert 0 <= self.threshold_continue <= 1
        assert 0 <= self.threshold_execute <= 1


# ---------------------------------------------------------------------------
# Layer 4: Deep Reasoning Configuration
# ---------------------------------------------------------------------------

@dataclass
class Layer4DeepReasoningConfig:
    """Layer 4: Deep reasoning configuration."""

    enabled: bool = True
    thinking_budget: int = 1200  # LLM thinking budget
    trigger_threshold: float = 0.75  # Auto-trigger if score below this

    # Conditions to skip deep reasoning
    skip_for_create_actions: bool = False  # Create actions skip deep reasoning
    skip_if_slots_missing: bool = True  # Skip if slots are missing


# ---------------------------------------------------------------------------
# Layer 5: Deep Fusion (A + B + C)
# ---------------------------------------------------------------------------

@dataclass
class Layer5DeepFusionConfig:
    """Layer 5: A + B + C (deep reasoning) fusion configuration."""

    # Fusion weights
    weight_a: float = 0.25  # LLM-light confidence
    weight_b: float = 0.35  # Ontology match confidence
    weight_c: float = 0.40  # Deep reasoning confidence

    # Decision thresholds
    threshold_execute: float = 0.80  # >= this + slots complete → execute
    threshold_continue: float = 0.70  # >= this + slots complete → continue
    threshold_clarify: float = 0.70  # < this → clarify

    def validate(self) -> None:
        """Validate configuration values."""
        total = self.weight_a + self.weight_b + self.weight_c
        assert abs(total - 1.0) < 0.01, \
            f"weights must sum to 1, got {total}"


# ---------------------------------------------------------------------------
# Layer 6: HITL Configuration
# ---------------------------------------------------------------------------

@dataclass
class Layer6HITLConfig:
    """Layer 6: Human-in-the-loop configuration."""

    enabled: bool = True
    trigger_threshold: float = 0.75

    # Force HITL conditions
    force_on_missing_slots: bool = True  # Missing required slots → HITL
    force_on_create_actions: bool = True  # Create actions always require HITL
    force_on_low_confidence: bool = True   # Low confidence → HITL

    # Clarification generation
    max_options: int = 4  # Max number of clarification options
    include_ambiguities: bool = True  # Include detected ambiguities


# ---------------------------------------------------------------------------
# Layer 7: Dynamic Resolution Configuration
# ---------------------------------------------------------------------------

@dataclass
class Layer7DynamicResolutionConfig:
    """Layer 7: Dynamic slot resolution configuration."""

    enabled: bool = True

    # Resolution targets
    resolve_organizations: bool = True  # Resolve org names to IDs
    resolve_materials: bool = True     # Resolve material names to codes
    resolve_vendors: bool = True       # Resolve vendor names to IDs

    # Fallback behavior
    fallback_on_failure: bool = True  # Fall back to raw value if resolution fails
    cache_enabled: bool = True         # Cache resolved values


# ---------------------------------------------------------------------------
# Layer 2.5: Slot Completion Configuration
# ---------------------------------------------------------------------------

@dataclass
class SlotCompletionConfig:
    """Slot completion check configuration."""

    # Readiness thresholds
    threshold_execute: float = 100.0  # 100% slots required for execute
    threshold_clarify: float = 50.0   # >= 50% → clarify, < 50% → block

    # Required vs optional slots
    require_all_slots: bool = True  # All slots required (strict mode)
    allow_partial: bool = False     # Allow partial slot filling

    # Priority rules
    priority_required_over_confidence: bool = True  # Missing slots override confidence


# ---------------------------------------------------------------------------
# Blocked Actions Configuration
# ---------------------------------------------------------------------------

@dataclass
class ActionRulesConfig:
    """Action-specific rules configuration."""

    # Actions that should NEVER be final execution targets
    blocked_final_actions: List[str] = field(default_factory=lambda: [
        "query_object",
        "create_object",
        "update_object",
        "delete_object",
    ])

    # Actions that always require HITL
    hitl_required_actions: List[str] = field(default_factory=lambda: [
        "create_object",
        "submit_approval",
    ])

    # Cancel keywords for pending HITL sessions
    cancel_keywords: List[str] = field(default_factory=lambda: [
        "取消", "重新开始", "不是这个", "算了", "换话题",
        "quit", "cancel", "start over", "wrong",
    ])


# ---------------------------------------------------------------------------
# Slot Fill Protocol Configuration
# ---------------------------------------------------------------------------

@dataclass
class SlotFillProtocolConfig:
    """Slot-fill protocol configuration."""

    enabled: bool = True
    protocol_prefix: str = "[slot-fill]"
    pattern: str = r"^\[slot-fill\]\s*([^\s|]+)\s*\|?\s*(.+)?$"

    # Persistence
    persist_tasks: bool = True  # Persist pending tasks to file
    task_expiry_hours: int = 24  # Task expires after this many hours


# ---------------------------------------------------------------------------
# Main Configuration Class
# ---------------------------------------------------------------------------

@dataclass
class IntentRecognitionConfig:
    """Unified configuration for the intent recognition pipeline.

    Example usage:
        config = IntentRecognitionConfig.default()
        config.layer3.threshold_execute = 0.9

        # Use in pipeline
        pipeline = CompositeIntentPipeline(config=config)
    """

    # Version for migration support
    version: str = "1.0.0"

    # Named preset (overrides other settings when set)
    preset: Optional[str] = None  # "fast", "accurate", "balanced"

    # Layer configurations
    layer3: Layer3FusionConfig = field(default_factory=Layer3FusionConfig)
    layer4: Layer4DeepReasoningConfig = field(default_factory=Layer4DeepReasoningConfig)
    layer5: Layer5DeepFusionConfig = field(default_factory=Layer5DeepFusionConfig)
    layer6: Layer6HITLConfig = field(default_factory=Layer6HITLConfig)
    layer7: Layer7DynamicResolutionConfig = field(default_factory=Layer7DynamicResolutionConfig)
    slot_completion: SlotCompletionConfig = field(default_factory=SlotCompletionConfig)
    action_rules: ActionRulesConfig = field(default_factory=ActionRulesConfig)
    slot_fill_protocol: SlotFillProtocolConfig = field(default_factory=SlotFillProtocolConfig)

    @classmethod
    def default(cls) -> "IntentRecognitionConfig":
        """Create configuration with sensible defaults."""
        return cls()

    @classmethod
    def fast(cls) -> "IntentRecognitionConfig":
        """Fast configuration: skip deep reasoning, lower thresholds."""
        config = cls()
        config.preset = "fast"
        config.layer4.enabled = False
        config.layer3.threshold_execute = 0.80
        config.layer3.threshold_continue = 0.70
        config.layer3.threshold_deep_reasoning = 0.70
        config.layer6.force_on_low_confidence = False
        return config

    @classmethod
    def accurate(cls) -> "IntentRecognitionConfig":
        """Accurate configuration: always use deep reasoning, high thresholds."""
        config = cls()
        config.preset = "accurate"
        config.layer4.enabled = True
        config.layer3.threshold_execute = 0.90
        config.layer3.threshold_continue = 0.85
        config.layer3.threshold_deep_reasoning = 0.80
        config.layer5.threshold_execute = 0.85
        config.layer5.threshold_continue = 0.80
        return config

    @classmethod
    def balanced(cls) -> "IntentRecognitionConfig":
        """Balanced configuration: moderate thresholds."""
        return cls.default()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentRecognitionConfig":
        """Create configuration from dictionary."""
        def nested_from_dict(data_class, data_dict):
            if data_dict is None:
                return data_class()
            fields = {}
            for field_name in data_class.__dataclass_fields__:
                if field_name in data_dict:
                    field_type = data_class.__dataclass_fields__[field_name].type
                    if hasattr(field_type, "__dataclass_fields__"):
                        # Nested dataclass
                        fields[field_name] = nested_from_dict(field_type, data_dict[field_name])
                    else:
                        fields[field_name] = data_dict[field_name]
            return data_class(**fields)

        return nested_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if hasattr(value, "__dataclass_fields__"):
                result[field_name] = asdict(value)
            else:
                result[field_name] = value
        return result

    @classmethod
    def from_json(cls, json_str: str) -> "IntentRecognitionConfig":
        """Create configuration from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, file_path: str) -> "IntentRecognitionConfig":
        """Load configuration from JSON or YAML file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(path) as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
        elif suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")

        return cls.from_dict(data)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_to_file(self, file_path: str, indent: int = 2) -> None:
        """Save configuration to JSON file."""
        suffix = Path(file_path).suffix.lower()
        if suffix in (".yaml", ".yml"):
            import yaml
            with open(file_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            with open(file_path, "w") as f:
                f.write(self.to_json(indent=indent))

    @classmethod
    def from_env(cls, prefix: str = "INTENT_RECOGNITION_") -> "IntentRecognitionConfig":
        """Load configuration from environment variables.

        Environment variables follow the pattern: INTENT_RECOGNITION_SECTION_KEY
        Example: INTENT_RECOGNITION_LAYER3_WEIGHT_A = 0.5
        """
        config = cls()

        # Map of env vars to config paths
        env_mappings = {
            "LAYER3_WEIGHT_A": ("layer3", "weight_a", float),
            "LAYER3_WEIGHT_B": ("layer3", "weight_b", float),
            "LAYER3_THRESHOLD_EXECUTE": ("layer3", "threshold_execute", float),
            "LAYER3_THRESHOLD_CONTINUE": ("layer3", "threshold_continue", float),
            "LAYER3_THRESHOLD_DEEP_REASONING": ("layer3", "threshold_deep_reasoning", float),
            "LAYER4_ENABLED": ("layer4", "enabled", lambda x: x.lower() == "true"),
            "LAYER4_THINKING_BUDGET": ("layer4", "thinking_budget", int),
            "LAYER5_WEIGHT_A": ("layer5", "weight_a", float),
            "LAYER5_WEIGHT_B": ("layer5", "weight_b", float),
            "LAYER5_WEIGHT_C": ("layer5", "weight_c", float),
            "LAYER5_THRESHOLD_EXECUTE": ("layer5", "threshold_execute", float),
            "LAYER5_THRESHOLD_CONTINUE": ("layer5", "threshold_continue", float),
            "LAYER6_ENABLED": ("layer6", "enabled", lambda x: x.lower() == "true"),
            "LAYER6_FORCE_ON_MISSING_SLOTS": ("layer6", "force_on_missing_slots", lambda x: x.lower() == "true"),
            "LAYER6_FORCE_ON_CREATE_ACTIONS": ("layer6", "force_on_create_actions", lambda x: x.lower() == "true"),
            "LAYER7_ENABLED": ("layer7", "enabled", lambda x: x.lower() == "true"),
            "SLOT_COMPLETION_THRESHOLD_EXECUTE": ("slot_completion", "threshold_execute", float),
            "SLOT_COMPLETION_REQUIRE_ALL_SLOTS": ("slot_completion", "require_all_slots", lambda x: x.lower() == "true"),
            "SLOT_FILL_PROTOCOL_ENABLED": ("slot_fill_protocol", "enabled", lambda x: x.lower() == "true"),
        }

        for env_key, (section, key, converter) in env_mappings.items():
            full_key = f"{prefix}{env_key}"
            if full_key in os.environ:
                value = converter(os.environ[full_key])
                section_obj = getattr(config, section)
                setattr(section_obj, key, value)

        return config

    def validate(self) -> None:
        """Validate the entire configuration."""
        self.layer3.validate()
        self.layer5.validate()


# ---------------------------------------------------------------------------
# Convenience: Default config instance
# ---------------------------------------------------------------------------

_default_config: Optional[IntentRecognitionConfig] = None


def get_default_config() -> IntentRecognitionConfig:
    """Get the default configuration instance (singleton)."""
    global _default_config
    if _default_config is None:
        _default_config = IntentRecognitionConfig.default()
    return _default_config


def set_default_config(config: IntentRecognitionConfig) -> None:
    """Set the default configuration instance."""
    global _default_config
    config.validate()
    _default_config = config
