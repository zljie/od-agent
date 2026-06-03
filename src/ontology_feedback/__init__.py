"""本体反馈机制 - Ontology Feedback Mechanism

本模块实现文档定义的"本体能力飞轮"：
场景运行 → 发现本体缺口 → 补充对象/属性/关系/行为/规则 → Agent能力增强 → 支持更多场景

功能：
1. 本体缺口自动检测
2. 缺口报告生成
3. 本体优化建议输出
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from collections import defaultdict


class GapType(str, Enum):
    """本体缺口类型"""
    MISSING_ATTRIBUTE = "missing_attribute"      # 缺失属性
    MISSING_ENTITY = "missing_entity"           # 缺失实体
    MISSING_RELATIONSHIP = "missing_relationship"  # 缺失关系
    MISSING_ACTION = "missing_action"          # 缺失行为
    MISSING_RULE = "missing_rule"             # 缺失规则
    MISSING_SYNONYM = "missing_synonym"       # 缺失同义词
    MISSING_METRIC = "missing_metric"         # 缺失指标
    UNDEFINED_STATUS = "undefined_status"     # 未定义状态口径
    UNCLEAR_DEFINITION = "unclear_definition"  # 定义不清晰


class GapSeverity(str, Enum):
    """缺口严重程度"""
    BLOCKING = "blocking"      # 阻断性 - 功能无法正常使用
    HIGH = "high"            # 高 - 影响核心功能
    MEDIUM = "medium"        # 中 - 影响用户体验
    LOW = "low"             # 低 - 优化建议


@dataclass
class OntologyGap:
    """本体缺口"""
    gap_type: GapType                         # 缺口类型
    entity_name: str                           # 涉及的实体名
    description: str                           # 缺口描述
    suggestion: str                            # 修复建议
    severity: GapSeverity = GapSeverity.MEDIUM  # 严重程度
    detected_in_context: str = ""             # 检测上下文
    detected_at: str = ""                     # 检测时间
    frequency: int = 1                       # 出现频率
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()


@dataclass
class GapReport:
    """缺口报告"""
    session_id: str = ""
    request_id: str = ""
    timestamp: str = ""
    total_gaps: int = 0
    blocking_gaps: int = 0
    high_gaps: int = 0
    medium_gaps: int = 0
    low_gaps: int = 0
    gaps: List[OntologyGap] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        self.total_gaps = len(self.gaps)
        self.blocking_gaps = sum(1 for g in self.gaps if g.severity == GapSeverity.BLOCKING)
        self.high_gaps = sum(1 for g in self.gaps if g.severity == GapSeverity.HIGH)
        self.medium_gaps = sum(1 for g in self.gaps if g.severity == GapSeverity.MEDIUM)
        self.low_gaps = sum(1 for g in self.gaps if g.severity == GapSeverity.LOW)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total_gaps,
                "blocking": self.blocking_gaps,
                "high": self.high_gaps,
                "medium": self.medium_gaps,
                "low": self.low_gaps,
            },
            "gaps": [
                {
                    "type": g.gap_type.value,
                    "entity": g.entity_name,
                    "description": g.description,
                    "suggestion": g.suggestion,
                    "severity": g.severity.value,
                    "detected_at": g.detected_at,
                    "frequency": g.frequency,
                }
                for g in self.gaps
            ],
        }


class OntologyGapDetector:
    """本体缺口检测器"""
    
    # 已知缺口模式 - 用于检测
    KNOWN_GAPS = {
        "purchase_requests": {
            "missing_attributes": [
                ("source_type", "来源类型（系统接口集成/手工创建/批导）", GapSeverity.HIGH),
                ("material_category", "物料分类", GapSeverity.MEDIUM),
                ("execution_status", "执行状态口径", GapSeverity.HIGH),
            ],
        },
        "purchase_inquiries": {
            "missing_attributes": [
                ("publish_status", "发布状态", GapSeverity.HIGH),
                ("quotation_count", "有效报价数量", GapSeverity.MEDIUM),
            ],
        },
        "purchase_quotations": {
            "missing_attributes": [
                ("lock_status", "锁定状态", GapSeverity.MEDIUM),
                ("history_avg_price", "历史成交均价", GapSeverity.MEDIUM),
            ],
        },
    }
    
    # 常见检测关键词
    DETECTION_KEYWORDS = {
        "source_type": ["来源", "source", "手工", "批导", "系统对接"],
        "material_category": ["物料分类", "category", "分类"],
        "execution_status": ["执行状态", "未执行", "部分执行", "状态口径"],
        "publish_status": ["发布状态", "发布", "撤回", "草稿"],
        "lock_status": ["锁定", "lock", "解锁"],
        "quotation_count": ["报价数量", "有效报价"],
        "history_avg_price": ["历史均价", "历史价格", "成交均价"],
    }
    
    def __init__(self):
        self._detected_gaps: Dict[str, OntologyGap] = {}  # 按key去重
    
    def detect_gaps(
        self,
        intent: str,
        action: str,
        params: Dict[str, Any],
        result: Any,
        error: Optional[str] = None
    ) -> List[OntologyGap]:
        """检测本体缺口
        
        Args:
            intent: 识别的意图
            action: 执行的动作
            params: 动作参数
            result: 执行结果
            error: 错误信息
            
        Returns:
            检测到的缺口列表
        """
        gaps: List[OntologyGap] = []
        
        # 1. 检测结果数据中的缺失字段
        gaps.extend(self._detect_missing_fields(intent, action, params, result))
        
        # 2. 检测参数中的未知字段
        gaps.extend(self._detect_unknown_params(intent, action, params))
        
        # 3. 检测执行错误相关的缺口
        if error:
            gaps.extend(self._detect_error_gaps(intent, action, error))
        
        # 4. 更新全局缺口统计
        for gap in gaps:
            self._update_gap_stats(gap)
        
        return gaps
    
    def _detect_missing_fields(
        self,
        intent: str,
        action: str,
        params: Dict[str, Any],
        result: Any
    ) -> List[OntologyGap]:
        """检测缺失字段"""
        gaps: List[OntologyGap] = []
        
        if not result or not isinstance(result, dict):
            return gaps
        
        items = result.get("items", [])
        if not items or not isinstance(items, list):
            return gaps
        
        first_item = items[0] if items else {}
        
        # 检查purchase_requests相关查询
        if "purchase_request" in action.lower() or "pr" in intent.lower():
            # 检测source_type
            if "source_type" not in first_item:
                gaps.append(OntologyGap(
                    gap_type=GapType.MISSING_ATTRIBUTE,
                    entity_name="purchase_requests",
                    description="本体缺少 'source_type' 字段，无法稳定统计需求来源分布",
                    suggestion="补充 source_type 字段，枚举值：system(系统接口集成)/manual(手工创建)/batch(批导)",
                    severity=GapSeverity.HIGH,
                    detected_in_context=f"action={action}, intent={intent}",
                ))
            
            # 检测material_category
            if "material_category" not in first_item:
                gaps.append(OntologyGap(
                    gap_type=GapType.MISSING_ATTRIBUTE,
                    entity_name="purchase_requests",
                    description="本体缺少 'material_category' 字段，无法按物料分类统计",
                    suggestion="补充 material_category 字段，枚举值：office/production/it/engineering等",
                    severity=GapSeverity.MEDIUM,
                    detected_in_context=f"action={action}, intent={intent}",
                ))
        
        # 检查purchase_inquiries相关查询
        if "inquiry" in action.lower() or "inquiry" in intent.lower():
            if "publish_status" not in first_item:
                gaps.append(OntologyGap(
                    gap_type=GapType.MISSING_ATTRIBUTE,
                    entity_name="purchase_inquiries",
                    description="本体缺少 'publish_status' 字段，无法判断询价单发布状态",
                    suggestion="补充 publish_status 字段，枚举值：draft/published/withdrawn/closed",
                    severity=GapSeverity.HIGH,
                    detected_in_context=f"action={action}, intent={intent}",
                ))
        
        # 检查purchase_quotations相关查询
        if "quotation" in action.lower():
            if "lock_status" not in first_item:
                gaps.append(OntologyGap(
                    gap_type=GapType.MISSING_ATTRIBUTE,
                    entity_name="purchase_quotations",
                    description="本体缺少 'lock_status' 字段，无法判断报价单是否被锁定",
                    suggestion="补充 lock_status 字段",
                    severity=GapSeverity.MEDIUM,
                    detected_in_context=f"action={action}, intent={intent}",
                ))
        
        return gaps
    
    def _detect_unknown_params(
        self,
        intent: str,
        action: str,
        params: Dict[str, Any]
    ) -> List[OntologyGap]:
        """检测未知参数"""
        gaps: List[OntologyGap] = []
        
        # 检测是否有模糊的状态值
        for key, value in params.items():
            if isinstance(value, str):
                # 检测可能的状态值但未被定义
                status_keywords = ["状态", "status", "类型", "type"]
                if any(kw in key.lower() for kw in status_keywords):
                    # 检查值是否包含可能需要定义的概念
                    uncertain_values = ["其他", "其他类型", "unknown", "undefined"]
                    if any(uv in str(value).lower() for uv in uncertain_values):
                        gaps.append(OntologyGap(
                            gap_type=GapType.UNDEFINED_STATUS,
                            entity_name=key,
                            description=f"字段 '{key}' 存在未定义的状态值 '{value}'",
                            suggestion="在本体中明确定义该字段的所有可能状态值及其含义",
                            severity=GapSeverity.MEDIUM,
                            detected_in_context=f"action={action}, param_key={key}",
                        ))
        
        return gaps
    
    def _detect_error_gaps(
        self,
        intent: str,
        action: str,
        error: str
    ) -> List[OntologyGap]:
        """检测错误相关的缺口"""
        gaps: List[OntologyGap] = []
        
        # 规则阻断错误
        rule_keywords = ["规则", "rule", "不允许", "不符合"]
        if any(kw in error.lower() for kw in rule_keywords):
            gaps.append(OntologyGap(
                gap_type=GapType.MISSING_RULE,
                entity_name="rules",
                description=f"执行 {action} 时触发规则阻断，但规则可能不完整",
                suggestion="检查并完善相关业务规则定义",
                severity=GapSeverity.MEDIUM,
                detected_in_context=f"action={action}, error={error[:100]}",
            ))
        
        # 连接器错误
        connector_keywords = ["连接器", "connector", "系统", "系统调用"]
        if any(kw in error.lower() for kw in connector_keywords):
            gaps.append(OntologyGap(
                gap_type=GapType.MISSING_ACTION,
                entity_name="connectors",
                description=f"执行 {action} 时连接器调用失败",
                suggestion="检查连接器配置或补充缺失的连接器实现",
                severity=GapSeverity.HIGH,
                detected_in_context=f"action={action}, error={error[:100]}",
            ))
        
        return gaps
    
    def _update_gap_stats(self, gap: OntologyGap) -> None:
        """更新缺口统计"""
        gap_key = f"{gap.gap_type.value}:{gap.entity_name}:{gap.description[:50]}"
        
        if gap_key in self._detected_gaps:
            existing = self._detected_gaps[gap_key]
            existing.frequency += 1
            # 更新严重程度（如果发现更严重）
            severity_order = [GapSeverity.LOW, GapSeverity.MEDIUM, GapSeverity.HIGH, GapSeverity.BLOCKING]
            if severity_order.index(existing.severity) > severity_order.index(gap.severity):
                existing.severity = gap.severity
        else:
            self._detected_gaps[gap_key] = gap
    
    def get_gap_summary(self) -> Dict[str, Any]:
        """获取缺口摘要"""
        gaps = list(self._detected_gaps.values())
        
        by_severity = defaultdict(list)
        for gap in gaps:
            by_severity[gap.severity.value].append(gap)
        
        by_type = defaultdict(list)
        for gap in gaps:
            by_type[gap.gap_type.value].append(gap)
        
        return {
            "total_unique_gaps": len(gaps),
            "total_detections": sum(g.frequency for g in gaps),
            "by_severity": {
                severity: len(gaps_list)
                for severity, gaps_list in by_severity.items()
            },
            "by_type": {
                gtype: len(gaps_list)
                for gtype, gaps_list in by_type.items()
            },
            "top_gaps": sorted(
                gaps,
                key=lambda g: g.frequency,
                reverse=True
            )[:10],
        }


class OntologyFeedbackGenerator:
    """本体反馈生成器"""
    
    def __init__(self, detector: Optional[OntologyGapDetector] = None):
        self._detector = detector or OntologyGapDetector()
    
    def generate_feedback(
        self,
        gaps: List[OntologyGap],
        context: Optional[Dict[str, Any]] = None
    ) -> GapReport:
        """生成缺口报告
        
        Args:
            gaps: 检测到的缺口列表
            context: 额外上下文
            
        Returns:
            缺口报告
        """
        report = GapReport(
            session_id=context.get("session_id", "") if context else "",
            request_id=context.get("request_id", "") if context else "",
            gaps=gaps,
        )
        
        return report
    
    def generate_user_friendly_message(
        self,
        gaps: List[OntologyGap],
        max_gaps: int = 3
    ) -> str:
        """生成用户友好的反馈消息
        
        Args:
            gaps: 检测到的缺口列表
            max_gaps: 最大显示的缺口数
            
        Returns:
            反馈消息
        """
        if not gaps:
            return ""
        
        parts = ["📝 **本体优化建议**"]
        parts.append("")
        
        # 按严重程度排序
        sorted_gaps = sorted(
            gaps,
            key=lambda g: [
                GapSeverity.BLOCKING,
                GapSeverity.HIGH,
                GapSeverity.MEDIUM,
                GapSeverity.LOW
            ].index(g.severity)
        )
        
        for i, gap in enumerate(sorted_gaps[:max_gaps], 1):
            severity_icon = {
                GapSeverity.BLOCKING: "🔴",
                GapSeverity.HIGH: "🟠",
                GapSeverity.MEDIUM: "🟡",
                GapSeverity.LOW: "🟢",
            }
            
            parts.append(f"{severity_icon.get(gap.severity, '⚪')} **{gap.description}**")
            parts.append(f"   建议: {gap.suggestion}")
            parts.append("")
        
        if len(gaps) > max_gaps:
            parts.append(f"_还有 {len(gaps) - max_gaps} 个优化建议_")
        
        return "\n".join(parts)
    
    def generate_ontology_yaml_patch(
        self,
        gaps: List[OntologyGap]
    ) -> str:
        """生成本体YAML补丁
        
        Args:
            gaps: 检测到的缺口列表
            
        Returns:
            YAML格式的补丁内容
        """
        lines = ["# 本体优化补丁 - 由本体反馈机制自动生成"]
        lines.append(f"# 生成时间: {datetime.now().isoformat()}")
        lines.append("")
        
        # 按实体分组
        by_entity: Dict[str, List[OntologyGap]] = defaultdict(list)
        for gap in gaps:
            by_entity[gap.entity_name].append(gap)
        
        for entity_name, entity_gaps in by_entity.items():
            lines.append(f"# === {entity_name} ===")
            
            for gap in entity_gaps:
                if gap.gap_type == GapType.MISSING_ATTRIBUTE:
                    field_name = gap.suggestion.split("'")[1] if "'" in gap.suggestion else f"new_{gap.gap_type.value}"
                    lines.append(f"# 新增字段: {field_name}")
                    lines.append(f"#   描述: {gap.description}")
                    lines.append(f"#   建议: {gap.suggestion}")
                    lines.append("")
                elif gap.gap_type == GapType.MISSING_RULE:
                    lines.append(f"# 新增规则")
                    lines.append(f"#   描述: {gap.description}")
                    lines.append(f"#   建议: {gap.suggestion}")
                    lines.append("")
            
            lines.append("")
        
        return "\n".join(lines)


class OntologyFeedbackManager:
    """本体反馈管理器 - 管理整个反馈流程"""
    
    def __init__(self):
        self._detector = OntologyGapDetector()
        self._generator = OntologyFeedbackGenerator(self._detector)
        self._feedback_history: List[GapReport] = []
    
    def record_execution(
        self,
        intent: str,
        action: str,
        params: Dict[str, Any],
        result: Any,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> GapReport:
        """记录执行并检测缺口
        
        Args:
            intent: 识别的意图
            action: 执行的动作
            params: 动作参数
            result: 执行结果
            error: 错误信息
            context: 额外上下文
            
        Returns:
            缺口报告
        """
        # 检测缺口
        gaps = self._detector.detect_gaps(intent, action, params, result, error)
        
        # 生成报告
        report = self._generator.generate_feedback(gaps, context)
        
        # 记录历史
        self._feedback_history.append(report)
        
        return report
    
    def get_cumulative_report(self, session_id: str = "") -> Dict[str, Any]:
        """获取累计缺口报告
        
        Args:
            session_id: 可选的会话ID过滤
            
        Returns:
            累计报告摘要
        """
        if session_id:
            reports = [r for r in self._feedback_history if r.session_id == session_id]
        else:
            reports = self._feedback_history
        
        # 汇总缺口
        all_gaps: Dict[str, OntologyGap] = {}
        for report in reports:
            for gap in report.gaps:
                gap_key = f"{gap.gap_type.value}:{gap.entity_name}:{gap.description[:50]}"
                if gap_key in all_gaps:
                    all_gaps[gap_key].frequency += gap.frequency
                else:
                    all_gaps[gap_key] = gap
        
        return {
            "total_sessions": len(set(r.session_id for r in reports)),
            "total_executions": len(reports),
            "unique_gaps": len(all_gaps),
            "total_gap_occurrences": sum(g.frequency for g in all_gaps.values()),
            "top_gaps": sorted(
                all_gaps.values(),
                key=lambda g: g.frequency,
                reverse=True
            )[:20],
        }
    
    def get_detector(self) -> OntologyGapDetector:
        """获取检测器"""
        return self._detector
    
    def export_report(self, format: str = "dict") -> Any:
        """导出报告
        
        Args:
            format: 导出格式 (dict/json/yaml)
            
        Returns:
            格式化后的报告
        """
        summary = self.get_cumulative_report()
        
        if format == "dict":
            return summary
        elif format == "json":
            import json
            return json.dumps(summary, ensure_ascii=False, indent=2, default=str)
        elif format == "yaml":
            return self._generator.generate_ontology_yaml_patch(
                summary.get("top_gaps", [])
            )
        
        return summary


# 全局管理器实例
_feedback_manager: Optional[OntologyFeedbackManager] = None


def get_feedback_manager() -> OntologyFeedbackManager:
    """获取反馈管理器单例"""
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = OntologyFeedbackManager()
    return _feedback_manager
