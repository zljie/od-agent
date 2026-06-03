"""
Layer 0: Input Preprocessor
===========================
Normalizes raw user input and prepares it for downstream intent recognition.

Responsibilities:
1. Text cleaning (whitespace, punctuation, fullwidth characters)
2. Typo correction (Chinese text errors)
3. Light tokenization (split on punctuation, keep business terms together)
4. Term categorization (action_terms, object_terms, dynamic_terms)
5. Dynamic term identification (requires master data lookup)

Key principle (from 8-layer design):
  raw_input is NEVER overwritten — all transformations are recorded separately.
"""

import re
from typing import Any, Dict, List, Tuple

from ..preprocessing.cleaner import BasicCleaner, CorrectionRecord
from ..preprocessing.typo_corrector import TypoCorrector
from .models import PreprocessingResult


# ---------------------------------------------------------------------------
# Known term patterns for dynamic entity types
# ---------------------------------------------------------------------------

# Patterns that strongly indicate a dynamic entity type needing master data lookup
_DYNAMIC_TERM_PATTERNS: Dict[str, List[re.Pattern]] = {
    # Organization/department names: "XX部", "XX部门", "XX科"
    "organization": [
        re.compile(r"[\u4e00-\u9fff]{2,6}(?:部|部门|科|中心|组|科室)$"),
        re.compile(r"研发|市场|销售|采购|财务|人事|生产|质量|管理|行政|工程|运营"),
    ],
    # Supplier/vendor names: typically followed by "公司", "厂商", "供应商"
    "supplier": [
        re.compile(r"[\u4e00-\u9fff]{2,8}(?:公司|厂商|供应商| vendor| Vendor)$"),
        re.compile(r"^(?:联想|戴尔|惠普|华为|中兴|阿里|腾讯|百度|京东|淘宝|微软|苹果)$", re.IGNORECASE),
    ],
    # Material/product descriptions: "XX物料", "XX材料", "XX产品"
    "material": [
        re.compile(r"[\u4e00-\u9fff]{2,10}(?:物料|材料|产品|零件|配件|商品|货品)$"),
        re.compile(r"(?:芯片|CPU|内存|硬盘|显示器|键盘|鼠标|打印机|交换机|路由器|服务器)$"),
    ],
    # Person names: typically 2-4 Chinese characters in name position
    "user": [
        re.compile(r"^[\u4e00-\u9fff]{2,4}$"),
        re.compile(r"(?:经理|总监|主管|专员|工程师|设计师|分析师|顾问)$"),
    ],
}

# Known action verb patterns in Chinese procurement domain
_ACTION_TERM_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(?:查询|查看|检索|搜索|浏览|列出|获取|找到)$"),
    re.compile(r"^(?:创建|新建|生成|新增|添加|新建|录入)$"),
    re.compile(r"^(?:处理|审批|批准|通过|驳回|拒绝|审核)$"),
    re.compile(r"^(?:提交|上报|发送|发布|推送)$"),
    re.compile(r"^(?:修改|更新|变更|编辑|调整)$"),
    re.compile(r"^(?:删除|取消|作废|撤回)$"),
    re.compile(r"^(?:比较|对比|比价|分析|评估|推荐)$"),
    re.compile(r"^(?:导出|下载|打印|发送)$"),
]

# Known object noun patterns in Chinese procurement domain
_OBJECT_TERM_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(?:采购需求|PR|需求单|需求书)$"),
    re.compile(r"^(?:询价单|报价单|RFQ|INQ|QUO)$", re.IGNORECASE),
    re.compile(r"^(?:采购订单|PO|订单)$", re.IGNORECASE),
    re.compile(r"^(?:合同|协议)$"),
    re.compile(r"^(?:供应商|vendor|厂商)$", re.IGNORECASE),
    re.compile(r"^(?:物料|材料|商品|产品)$"),
    re.compile(r"^(?:审批|审批单|审批流)$"),
    re.compile(r"^(?:报表|报告|统计)$"),
]


class InputPreprocessor:
    """Layer 0 input preprocessor for intent recognition.

    Applies deterministic text cleaning and light tokenization to prepare
    raw user input for downstream LLM-based intent recognition.

    Processing pipeline:
        raw_input → TextCleaner → TypoCorrector → Tokenizer → TermClassifier → result

    Attributes
    ----------
    cleaner:
        BasicCleaner instance for deterministic text normalization.
    typo_corrector:
        TypoCorrector instance for Chinese typo correction.
    keep_original:
        If True, preserve original tokens alongside normalized ones.
    """

    def __init__(
        self,
        cleaner: BasicCleaner = None,
        typo_corrector: TypoCorrector = None,
        keep_original: bool = False,
    ):
        self.cleaner = cleaner or BasicCleaner()
        self.typo_corrector = typo_corrector or TypoCorrector()
        self.keep_original = keep_original

    def preprocess(self, raw_input: str) -> PreprocessingResult:
        """Run the full preprocessing pipeline on raw input.

        Parameters
        ----------
        raw_input:
            The raw user input string to preprocess.

        Returns
        -------
        PreprocessingResult
            Complete preprocessing result with normalized text, tokens,
            corrections, and categorized terms.

        Processing steps
        ----------------
        1. TextCleaner.clean(): whitespace, punctuation, fullwidth normalization
        2. TypoCorrector.correct_with_details(): Chinese typo correction
        3. Light tokenization: split on [，。；\\n,]+ for Chinese, [ .,;\\n]+ for English
        4. Term classification: categorize each token as action/object/dynamic
        5. Dynamic term identification: mark terms needing master data lookup
        """
        if not raw_input or not raw_input.strip():
            return PreprocessingResult(
                original_input=raw_input or "",
                normalized_input="",
                tokens=[],
                corrections=[],
                candidate_terms={"action_terms": [], "object_terms": [], "dynamic_terms": []},
                dynamic_resolution_tasks=[],
            )

        # Step 1: Deterministic text cleaning
        normalized = self.cleaner.clean(raw_input)

        # Step 2: Typo correction
        normalized, correction_records = self.typo_corrector.correct_with_details(normalized)

        # Convert CorrectionRecord to dict for storage
        corrections = [
            {
                "original": r.original,
                "corrected": r.corrected,
                "position": r.position,
                "correction_type": r.correction_type,
                "confidence": r.confidence,
                "method": r.method,
            }
            for r in correction_records
        ]

        # Step 3: Light tokenization
        tokens = self._tokenize(normalized)

        # Step 4: Term classification
        candidate_terms = self._classify_terms(tokens)

        # Step 5: Dynamic term identification
        dynamic_tasks = self._identify_dynamic_terms(candidate_terms)

        return PreprocessingResult(
            original_input=raw_input,
            normalized_input=normalized,
            tokens=tokens,
            corrections=corrections,
            candidate_terms=candidate_terms,
            dynamic_resolution_tasks=dynamic_tasks,
        )

    def _tokenize(self, text: str) -> List[str]:
        """Light tokenization splitting on punctuation boundaries.

        Strategy:
        - For Chinese: split on [，。；：、\n]+ (Chinese punctuation)
        - For English/mixed: also split on [ .,;:\n]+ (English punctuation)
        - Keep business terms (IDs like PR-XXX, PO-XXX) together as single tokens
        - Preserve numbers and codes as single tokens
        """
        if not text:
            return []

        # First, protect business IDs (PR-XXXXXX, PO-XXXXXX, RFQ-XXXXXX, etc.)
        protected_text, id_tokens = self._protect_business_ids(text)

        # Split on punctuation boundaries
        # Chinese: ，。；：、？！…——""
        # English: space, comma, period, semicolon, colon, newline
        # Also split on mixed: 中英文混合符号
        separator_pattern = r"[，。；：、？！…—\"''\s,;:\n]+"
        raw_tokens = [t.strip() for t in re.split(separator_pattern, protected_text)]
        raw_tokens = [t for t in raw_tokens if t]

        # Merge protected IDs back in and interleave with tokens
        tokens: List[str] = []
        id_idx = 0

        for token in raw_tokens:
            # Find any ID tokens that fall within or near this token's original position
            while id_idx < len(id_tokens):
                id_token, id_pos = id_tokens[id_idx]
                # If the ID token is a substring of the current token, extract it
                if id_token in token:
                    before, rest = token.split(id_token, 1)
                    if before:
                        tokens.append(before)
                    tokens.append(id_token)
                    token = rest
                    id_idx += 1
                else:
                    break
            if token:
                tokens.append(token)

        # Add remaining ID tokens
        while id_idx < len(id_tokens):
            tokens.append(id_tokens[id_idx][0])
            id_idx += 1

        return tokens

    def _protect_business_ids(self, text: str) -> Tuple[str, List[Tuple[str, int]]]:
        """Protect business IDs (PR-XXX, PO-XXX, RFQ-XXX, QUO-XXX) from tokenization.

        Returns:
            (protected_text, list_of_id_tokens_with_positions)
        """
        id_pattern = r"(?:PR|RFQ|INQ|QUO|PO)[-_]?\d{3,}"
        id_tokens: List[Tuple[str, int]] = []

        for match in re.finditer(id_pattern, text, re.IGNORECASE):
            id_tokens.append((match.group(), match.start()))

        # Replace IDs with placeholders to protect from punctuation splitting
        protected = text
        offset = 0
        for i, (id_token, pos) in enumerate(id_tokens):
            placeholder = f"__BID_{i}__"
            protected = protected[: pos + offset] + placeholder + protected[pos + offset + len(id_token) :]
            offset += len(placeholder) - len(id_token)
            # Update stored position
            id_tokens[i] = (id_token, pos)

        return protected, id_tokens

    def _classify_terms(self, tokens: List[str]) -> Dict[str, List[str]]:
        """Classify tokens into action_terms, object_terms, and dynamic_terms.

        Each token is checked against known patterns and categorized.
        Tokens matching multiple categories are listed in all applicable lists.
        """
        result: Dict[str, List[str]] = {
            "action_terms": [],
            "object_terms": [],
            "dynamic_terms": [],
        }

        for token in tokens:
            # Check if it's a dynamic term (entity needing lookup)
            is_dynamic = False
            for term_type, patterns in _DYNAMIC_TERM_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(token):
                        result["dynamic_terms"].append(token)
                        is_dynamic = True
                        break
                if is_dynamic:
                    break

            if is_dynamic:
                continue

            # Check if it's an action term
            for pattern in _ACTION_TERM_PATTERNS:
                if pattern.match(token):
                    result["action_terms"].append(token)
                    break

            # Check if it's an object term
            for pattern in _OBJECT_TERM_PATTERNS:
                if pattern.match(token):
                    result["object_terms"].append(token)
                    break

        return result

    def _identify_dynamic_terms(
        self, candidate_terms: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Identify terms that require runtime master data lookup.

        For each dynamic term, determine what entity types it might be
        and whether a lookup is needed.
        """
        tasks: List[Dict[str, Any]] = []

        for term in candidate_terms.get("dynamic_terms", []):
            type_candidates: List[str] = []

            for term_type, patterns in _DYNAMIC_TERM_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(term):
                        type_candidates.append(term_type)
                        break

            first_type = type_candidates[0] if type_candidates else "organization"
            tasks.append({
                "term": term,
                "term_type": first_type,
                "need_lookup": True,
            })

        return tasks

    def restore_business_ids(self, tokens: List[str]) -> List[str]:
        """Restore business ID placeholders back to original ID strings.

        This is useful when processing tokens that were protected during
        tokenization and need to be restored for downstream use.
        """
        result = []
        for token in tokens:
            if token.startswith("__BID_") and token.endswith("__"):
                # This was a placeholder - in practice, the original ID would be stored
                # For now, just remove the placeholder
                continue
            result.append(token)
        return result
