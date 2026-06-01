"""Input preprocessing pipeline: L1 BasicCleaner → L2 TermProtector → L3 TypoCorrector.

All stages are deterministic, require no LLM, and are driven by config files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cleaner import BasicCleaner, CorrectionRecord, PreprocessingResult
from .term_protector import TermProtector
from .typo_corrector import TypoCorrector


def load_preprocessing_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load preprocessing configuration from JSON file.

    Defaults to config/preprocessing_config.json if not specified.
    """
    import json
    import os
    from pathlib import Path

    if config_path is None:
        config_path = os.environ.get(
            "PREPROCESSING_CONFIG_PATH",
            str(Path(__file__).parent.parent.parent / "config" / "preprocessing_config.json"),
        )

    defaults: Dict[str, Any] = {
        "enabled": True,
        "stages": {
            "L1_cleaner": {"enabled": True},
            "L2_term_protector": {
                "enabled": True,
                "config_path": None,  # will use default
            },
            "L3_typo_corrector": {
                "enabled": True,
                "confidence_threshold": 0.95,
                "model": "pycorrector",
            },
        },
    }

    if not os.path.exists(config_path):
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
            # Deep merge
            result = defaults.copy()
            if "stages" in user_cfg:
                for stage, values in user_cfg["stages"].items():
                    if stage in result["stages"]:
                        result["stages"][stage].update(values)
            result.update({k: v for k, v in user_cfg.items() if k != "stages"})
            return result
    except Exception:
        return defaults


class InputPreprocessingPipeline:
    """L1-L3 preprocessing pipeline with lazy initialization from config.

    Usage:
        cfg = load_preprocessing_config()
        pipeline = InputPreprocessingPipeline.from_config(cfg)
        result = pipeline.process("帮我写一分简历")
        # result.normalized_input  → "帮我写一份简历"
        # result.corrections       → [CorrectionRecord(...)]
        # result.raw_input        → "帮我写一分简历"  (original, never modified)
    """

    def __init__(
        self,
        cleaner: Optional[BasicCleaner] = None,
        protector: Optional[TermProtector] = None,
        corrector: Optional[TypoCorrector] = None,
    ):
        self._cleaner = cleaner
        self._protector = protector
        self._corrector = corrector

    @classmethod
    def from_config(cls, config: Dict[str, Any], sem_skill: Optional[Any] = None) -> "InputPreprocessingPipeline":
        """Build a pipeline from a loaded config dict.

        Args:
            config: The loaded preprocessing config dict.
            sem_skill: Optional SemanticSkill instance; if provided and
                       config.phases.perceive.use_ontology_synonyms is True,
                       ontology synonyms are loaded into the TermProtector.
        """
        stages = config.get("stages", {})

        cleaner_cfg = stages.get("L1_cleaner", {})
        cleaner = BasicCleaner() if cleaner_cfg.get("enabled", True) else None

        protector_cfg = stages.get("L2_term_protector", {})
        protector = None
        if protector_cfg.get("enabled", True):
            path = protector_cfg.get("config_path")
            protector = TermProtector(config_path=path)

            # Phase 5 (Perceive): extend term whitelist with ontology synonyms
            inj_cfg = config.get("phases", {})
            perceive_cfg = inj_cfg.get("perceive", {})
            if (
                sem_skill is not None
                and perceive_cfg.get("use_ontology_synonyms", False)
                and protector is not None
            ):
                max_syn = perceive_cfg.get("max_synonyms", 50)
                added = protector.add_terms_from_ontology(sem_skill, max_synonyms=max_syn)
                if added > 0:
                    import logging
                    logging.getLogger("od_agent").info(
                        "TermProtector loaded %d synonyms from ontology", added
                    )

        corrector_cfg = stages.get("L3_typo_corrector", {})
        corrector = None
        if corrector_cfg.get("enabled", True):
            threshold = corrector_cfg.get("confidence_threshold", 0.95)
            corrector = TypoCorrector(confidence_threshold=threshold)

        return cls(cleaner=cleaner, protector=protector, corrector=corrector)

    def process(self, raw_input: str) -> PreprocessingResult:
        """Run the full L1→L2→L3 pipeline on raw_input.

        The raw_input is never modified — all corrections are recorded as
        CorrectionRecord entries and returned separately.
        """
        original = raw_input
        working = raw_input
        corrections: List[CorrectionRecord] = []
        protected_terms: List[str] = []
        work_to_orig: List[Tuple[int, int, str]] = []  # (start, end, placeholder)

        # L1: BasicCleaner
        if self._cleaner:
            working = self._cleaner.clean(working)

        # L2: TermProtector — replace terms with placeholders BEFORE correction
        if self._protector:
            working, mapping, terms = self._protector.protect_with_mapping(working)
            protected_terms = terms
            work_to_orig = mapping  # [(start, end, orig_term), ...]

        # L3: TypoCorrector — only runs on unprotected text
        if self._corrector:
            working, batch = self._corrector.correct_with_details(working)
            corrections.extend(batch)

        # L2: Restore protected terms
        if self._protector:
            working = self._protector.restore(working)

        return PreprocessingResult(
            raw_input=original,
            normalized_input=working,
            corrections=corrections,
            protected_terms=protected_terms,
        )

    @property
    def is_enabled(self) -> bool:
        """True if at least one stage is active."""
        return bool(self._cleaner or self._protector or self._corrector)
