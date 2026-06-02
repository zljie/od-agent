"""
采购场景槽位提取器
从用户输入中提取业务条件槽位：
- 时间范围 (上周、本月、上个月等)
- 物料分类 (办公用品、生产物料等)
- 采购类型 (NB、标准采购、紧急采购等)
- 部门 (研发部、市场部等)
- 执行状态 (未执行、已执行、待审批等)
"""

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set


# ============================================================
# 数据结构定义
# ============================================================


@dataclass
class TimeRange:
    """时间范围"""
    label: str  # 原始标签，如"上周"、"本月"
    date_from: str  # 开始日期 ISO 格式
    date_to: str  # 结束日期 ISO 格式
    resolved: bool = False  # 是否已解析为具体日期


@dataclass
class SlotValue:
    """槽位值"""
    slot_name: str
    value: Any
    display_value: str
    confidence: float = 1.0
    source: str = "extracted"  # extracted, inferred, default


@dataclass
class ExtractedSlots:
    """提取的所有槽位"""
    time_range: Optional[TimeRange] = None
    temporal_context: Optional[Dict[str, Any]] = None
    material_category: Optional[SlotValue] = None
    pr_type: Optional[SlotValue] = None
    department: Optional[SlotValue] = None
    execution_status: Optional[SlotValue] = None
    vendor_id: Optional[SlotValue] = None
    vendor_name: Optional[SlotValue] = None
    other_slots: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        if self.time_range:
            result["time_range"] = {
                "label": self.time_range.label,
                "date_from": self.time_range.date_from,
                "date_to": self.time_range.date_to,
            }
        if self.material_category:
            result["material_category"] = self.material_category.value
        if self.pr_type:
            result["pr_type"] = self.pr_type.value
        if self.department:
            result["department"] = self.department.value
        if self.execution_status:
            result["execution_status"] = self.execution_status.value
        if self.vendor_id:
            result["vendor_id"] = self.vendor_id.value
        if self.vendor_name:
            result["vendor_name"] = self.vendor_name.value
        result.update(self.other_slots)
        return result
    
    def to_connector_params(self) -> Dict[str, Any]:
        """转换为 Connector 查询参数"""
        params = {}
        if self.time_range:
            params["date_from"] = self.time_range.date_from
            params["date_to"] = self.time_range.date_to
        if self.material_category:
            params["material_category"] = self.material_category.value
        if self.pr_type:
            params["pr_type"] = self.pr_type.value
        if self.department:
            params["apply_dep"] = self.department.value
        if self.execution_status:
            params["execution_status"] = self.execution_status.value
        if self.vendor_id:
            params["vendor_id"] = self.vendor_id.value
        if self.vendor_name:
            params["vendor_name"] = self.vendor_name.value
        params.update(self.other_slots)
        return params


# ============================================================
# 时间范围提取器
# ============================================================


class TimeRangeExtractor:
    """时间范围提取器 - 将自然语言时间转换为日期范围"""
    
    # 支持的时间表达式
    TIME_EXPRESSIONS = {
        # 相对时间
        "今天": (0, 0),
        "昨天": (-1, -1),
        "前天": (-2, -2),
        "明天": (1, 1),
        "后天": (2, 2),
        # 周维度
        "上周": (-1, "week"),
        "本周": (0, "week"),
        "下周": (1, "week"),
        "上上周": (-2, "week"),
        # 月维度
        "上月": (-1, "month"),
        "本月": (0, "month"),
        "下月": (1, "month"),
        "上个月": (-1, "month"),
        "这个月": (0, "month"),
        "下个月": (1, "month"),
        # 年维度
        "去年": (-1, "year"),
        "今年": (0, "year"),
        "明年": (1, "year"),
    }
    
    # 上周/本周/下周的完整星期
    QUALIFIED_WEEKS = [
        "上周一", "上周二", "上周三", "上周四", "上周五", "上周六", "上周日",
        "本周一", "本周二", "本周三", "本周四", "本周五", "本周六", "本周日",
        "下周一", "下周二", "下周三", "下周四", "下周五", "下周六", "下周日",
    ]
    
    def __init__(self, today: Optional[date] = None):
        self.today = today or date.today()
    
    def extract(self, message: str) -> Optional[TimeRange]:
        """从消息中提取时间范围"""
        # 1. 尝试匹配带星期的时间 (上周五、下周一等)
        qualified = self._extract_qualified_weekday(message)
        if qualified:
            return qualified
        
        # 2. 尝试匹配基础时间表达式 (上周、本月、上个月等)
        basic = self._extract_basic_expression(message)
        if basic:
            return basic
        
        # 3. 尝试匹配日期范围表达式 (2024-01-01到2024-01-31)
        date_range = self._extract_date_range(message)
        if date_range:
            return date_range
        
        return None
    
    def _extract_qualified_weekday(self, message: str) -> Optional[TimeRange]:
        """提取带星期的时间表达式，如上周五、下周一"""
        for qw in self.QUALIFIED_WEEKS:
            if qw in message:
                return self._resolve_qualified_weekday(qw)
        return None
    
    def _resolve_qualified_weekday(self, label: str) -> TimeRange:
        """解析带星期的时间表达式"""
        prefix_map = {"上": -1, "本": 0, "下": 1}
        weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}
        
        prefix = label[:1]  # 上、本、下
        weekday_name = label[1:]  # 一、二、三...
        weekday_idx = weekday_map[weekday_name]
        weeks_offset = prefix_map[prefix]
        
        # 计算目标日期
        this_monday = self.today - timedelta(days=self.today.weekday())
        target_monday = this_monday + timedelta(weeks=weeks_offset)
        target_date = target_monday + timedelta(days=weekday_idx)
        
        # 同一周内的范围
        week_start = target_monday
        week_end = target_monday + timedelta(days=6)
        
        return TimeRange(
            label=label,
            date_from=week_start.isoformat(),
            date_to=week_end.isoformat(),
            resolved=True
        )
    
    def _extract_basic_expression(self, message: str) -> Optional[TimeRange]:
        """提取基础时间表达式"""
        for expr, (offset, unit) in self.TIME_EXPRESSIONS.items():
            if expr in message:
                return self._resolve_expression(expr, offset, unit)
        return None
    
    def _resolve_expression(self, label: str, offset: int, unit: str) -> TimeRange:
        """解析时间表达式"""
        if unit == "week":
            return self._resolve_week_range(label, offset)
        elif unit == "month":
            return self._resolve_month_range(label, offset)
        elif unit == "year":
            return self._resolve_year_range(label, offset)
        else:
            # 固定天数
            d = self.today + timedelta(days=offset)
            return TimeRange(
                label=label,
                date_from=d.isoformat(),
                date_to=d.isoformat(),
                resolved=True
            )
    
    def _resolve_week_range(self, label: str, weeks_offset: int) -> TimeRange:
        """解析周范围"""
        this_monday = self.today - timedelta(days=self.today.weekday())
        target_monday = this_monday + timedelta(weeks=weeks_offset)
        week_end = target_monday + timedelta(days=6)
        
        return TimeRange(
            label=label,
            date_from=target_monday.isoformat(),
            date_to=week_end.isoformat(),
            resolved=True
        )
    
    def _resolve_month_range(self, label: str, months_offset: int) -> TimeRange:
        """解析月范围"""
        year = self.today.year
        month = self.today.month + months_offset
        
        # 处理跨年
        while month < 1:
            month += 12
            year -= 1
        while month > 12:
            month -= 12
            year += 1
        
        # 月初
        month_start = date(year, month, 1)
        
        # 月末
        if month == 12:
            month_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(year, month + 1, 1) - timedelta(days=1)
        
        return TimeRange(
            label=label,
            date_from=month_start.isoformat(),
            date_to=month_end.isoformat(),
            resolved=True
        )
    
    def _resolve_year_range(self, label: str, years_offset: int) -> TimeRange:
        """解析年范围"""
        year = self.today.year + years_offset
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        
        return TimeRange(
            label=label,
            date_from=year_start.isoformat(),
            date_to=year_end.isoformat(),
            resolved=True
        )
    
    def _extract_date_range(self, message: str) -> Optional[TimeRange]:
        """提取日期范围 (2024-01-01到2024-01-31)"""
        pattern = r"(\d{4}-\d{2}-\d{2})\s*(?:到|至|-)\s*(\d{4}-\d{2}-\d{2})"
        match = re.search(pattern, message)
        if match:
            return TimeRange(
                label=f"{match.group(1)}至{match.group(2)}",
                date_from=match.group(1),
                date_to=match.group(2),
                resolved=True
            )
        return None


# ============================================================
# 物料分类提取器
# ============================================================


class MaterialCategoryExtractor:
    """物料分类提取器"""
    
    # 物料分类映射 - 支持多种表达
    MATERIAL_CATEGORIES = {
        # 主分类
        "办公用品": ["办公用品", "办公", "文具", "办公文具", "办公设备"],
        "生产物料": ["生产物料", "原材料", "生产原料", "原料"],
        "IT设备": ["IT设备", "IT", "电脑", "笔记本", "台式机", "服务器", "网络设备"],
        "工程物资": ["工程物资", "工程", "建材", "施工材料"],
        "办公家具": ["办公家具", "家具", "桌椅", "办公桌", "办公椅"],
        "劳保用品": ["劳保用品", "劳保", "防护用品", "安全用品"],
        "备品备件": ["备品备件", "备件", "配件"],
        "包装材料": ["包装材料", "包装", "纸箱", "包装箱"],
    }
    
    def __init__(self):
        # 构建反向索引：关键词 -> 主分类
        self._keyword_to_category: Dict[str, str] = {}
        # 按关键词长度排序，确保优先匹配更长的关键词
        for category, keywords in self.MATERIAL_CATEGORIES.items():
            for kw in keywords:
                self._keyword_to_category[kw] = category
    
    def extract(self, message: str) -> Optional[SlotValue]:
        """从消息中提取物料分类
        
        策略：优先匹配最长的关键词，避免"办公"被单独匹配
        """
        # 按关键词长度降序排序，优先匹配更长的关键词
        sorted_keywords = sorted(self._keyword_to_category.keys(), key=len, reverse=True)
        
        for keyword in sorted_keywords:
            if keyword in message:
                category = self._keyword_to_category[keyword]
                return SlotValue(
                    slot_name="material_category",
                    value=category,
                    display_value=category,
                    confidence=1.0,
                    source="extracted"
                )
        
        # 尝试模糊匹配
        fuzzy = self._fuzzy_match(message)
        if fuzzy:
            return fuzzy
        
        return None
    
    def _fuzzy_match(self, message: str) -> Optional[SlotValue]:
        """模糊匹配物料分类"""
        # 简单的模糊匹配：如果消息中包含某个分类的部分关键词
        for category, keywords in self.MATERIAL_CATEGORIES.items():
            match_count = sum(1 for kw in keywords if kw in message)
            if match_count > 0 and match_count < len(keywords):
                # 部分匹配，返回最可能的结果
                return SlotValue(
                    slot_name="material_category",
                    value=category,
                    display_value=f"可能的{category}",
                    confidence=0.7,
                    source="inferred"
                )
        return None
    
    def get_all_categories(self) -> List[str]:
        """获取所有物料分类"""
        return list(self.MATERIAL_CATEGORIES.keys())


# ============================================================
# 采购类型提取器
# ============================================================


class PrTypeExtractor:
    """采购类型提取器"""
    
    # 采购类型映射
    PR_TYPES = {
        "NB": ["NB", "标准采购", "标准"],
        "紧急采购": ["紧急采购", "紧急", "加急采购", "加急"],
        "生产采购": ["生产采购", "生产物料采购"],
        "项目采购": ["项目采购", "项目"],
        "研发采购": ["研发采购", "研发"],
        "MRO采购": ["MRO采购", "MRO", "维护维修采购"],
    }
    
    def __init__(self):
        # 构建反向索引
        self._keyword_to_type: Dict[str, str] = {}
        # 按关键词长度排序，确保优先匹配更长的关键词
        for pr_type, keywords in self.PR_TYPES.items():
            for kw in keywords:
                self._keyword_to_type[kw] = pr_type
    
    def extract(self, message: str) -> Optional[SlotValue]:
        """从消息中提取采购类型
        
        策略：优先匹配更长的关键词，避免短关键词误匹配
        """
        # 按关键词长度降序排序，优先匹配更长的关键词
        sorted_keywords = sorted(self._keyword_to_type.keys(), key=len, reverse=True)
        
        for keyword in sorted_keywords:
            if keyword in message:
                pr_type = self._keyword_to_type[keyword]
                return SlotValue(
                    slot_name="pr_type",
                    value=pr_type,
                    display_value=pr_type,
                    confidence=1.0,
                    source="extracted"
                )
        return None
    
    def get_all_types(self) -> List[str]:
        """获取所有采购类型"""
        return list(self.PR_TYPES.keys())


# ============================================================
# 部门提取器
# ============================================================


class DepartmentExtractor:
    """部门提取器"""
    
    # 部门映射
    DEPARTMENTS = {
        "研发部": ["研发部", "研发", "RD"],
        "市场部": ["市场部", "市场", "营销部"],
        "销售部": ["销售部", "销售"],
        "采购部": ["采购部", "采购"],
        "财务部": ["财务部", "财务"],
        "人事部": ["人事部", "人事", "人力资源部", "HR"],
        "生产部": ["生产部", "生产", "制造部"],
        "质量管理部": ["质量管理部", "质量部", "品管部", "QA"],
        "物流部": ["物流部", "物流"],
        "行政部": ["行政部", "行政"],
        "信息技术部": ["信息技术部", "IT部", "信息部"],
        "仓储部": ["仓储部", "仓储", "仓库"],
    }
    
    def __init__(self):
        self._keyword_to_dept: Dict[str, str] = {}
        # 按关键词长度排序，确保优先匹配更长的关键词
        for dept, keywords in self.DEPARTMENTS.items():
            for kw in keywords:
                self._keyword_to_dept[kw] = dept
    
    def extract(self, message: str) -> Optional[SlotValue]:
        """从消息中提取部门
        
        策略：优先匹配更长的关键词，如"质量管理部"优先于"质量部"
        """
        # 按关键词长度降序排序，优先匹配更长的关键词
        sorted_keywords = sorted(self._keyword_to_dept.keys(), key=len, reverse=True)
        
        for keyword in sorted_keywords:
            if keyword in message:
                dept = self._keyword_to_dept[keyword]
                return SlotValue(
                    slot_name="apply_dep",
                    value=dept,
                    display_value=dept,
                    confidence=1.0,
                    source="extracted"
                )
        return None
    
    def get_all_departments(self) -> List[str]:
        """获取所有部门"""
        return list(self.DEPARTMENTS.keys())


# ============================================================
# 执行状态提取器
# ============================================================


class ExecutionStatusExtractor:
    """执行状态提取器"""
    
    # 执行状态映射
    EXECUTION_STATUSES = {
        "未执行": ["未执行", "未完成", "待处理", "待执行", "待下达"],
        "已执行": ["已执行", "已完成", "已下达", "已下单"],
        "待审批": ["待审批", "审批中", "审批中"],
        "已审批": ["已审批", "审批通过", "已批准"],
        "已驳回": ["已驳回", "审批拒绝", "已拒绝"],
        "已取消": ["已取消", "已作废", "已删除"],
    }
    
    def __init__(self):
        self._keyword_to_status: Dict[str, str] = {}
        for status, keywords in self.EXECUTION_STATUSES.items():
            for kw in keywords:
                self._keyword_to_status[kw] = status
    
    def extract(self, message: str) -> Optional[SlotValue]:
        """从消息中提取执行状态"""
        for keyword, status in self._keyword_to_status.items():
            if keyword in message:
                return SlotValue(
                    slot_name="execution_status",
                    value=status,
                    display_value=status,
                    confidence=1.0,
                    source="extracted"
                )
        return None
    
    def get_all_statuses(self) -> List[str]:
        """获取所有执行状态"""
        return list(self.EXECUTION_STATUSES.keys())


# ============================================================
# 供应商提取器
# ============================================================


class VendorExtractor:
    """供应商提取器"""
    
    def extract(self, message: str) -> Optional[SlotValue]:
        """从消息中提取供应商信息"""
        # 模式1: "供应商XXX" 或 "XXX供应商"
        patterns = [
            r"供应商([^\s，,。]+)",
            r"([^\s，,。]+供应商)",
            r"来自([^\s，,。]+)报价",
            r"([^\s，,。]+)的报价",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                vendor_name = match.group(1)
                return SlotValue(
                    slot_name="vendor_name",
                    value=vendor_name,
                    display_value=vendor_name,
                    confidence=0.9,
                    source="extracted"
                )
        
        return None


# ============================================================
# 综合槽位提取器
# ============================================================


class ProcurementSlotExtractor:
    """采购场景综合槽位提取器
    
    使用示例:
        extractor = ProcurementSlotExtractor()
        slots = extractor.extract("先处理上周采购类型为办公用品的采购需求")
        
        # slots.time_range.date_from  # "2026-05-26"
        # slots.time_range.date_to    # "2026-06-01"
        # slots.pr_type.value          # "NB"
        # slots.material_category.value # "办公用品"
    """
    
    def __init__(self, today: Optional[date] = None):
        self.time_extractor = TimeRangeExtractor(today)
        self.material_extractor = MaterialCategoryExtractor()
        self.pr_type_extractor = PrTypeExtractor()
        self.department_extractor = DepartmentExtractor()
        self.status_extractor = ExecutionStatusExtractor()
        self.vendor_extractor = VendorExtractor()
    
    def extract(self, message: str) -> ExtractedSlots:
        """从消息中提取所有槽位"""
        slots = ExtractedSlots()
        
        # 1. 提取时间范围
        time_range = self.time_extractor.extract(message)
        if time_range:
            slots.time_range = time_range
        
        # 2. 提取物料分类
        material = self.material_extractor.extract(message)
        if material:
            slots.material_category = material
        
        # 3. 提取采购类型
        pr_type = self.pr_type_extractor.extract(message)
        if pr_type:
            slots.pr_type = pr_type
        
        # 4. 提取部门
        department = self.department_extractor.extract(message)
        if department:
            slots.department = department
        
        # 5. 提取执行状态
        status = self.status_extractor.extract(message)
        if status:
            slots.execution_status = status
        
        # 6. 提取供应商
        vendor = self.vendor_extractor.extract(message)
        if vendor:
            slots.vendor_name = vendor
        
        # 7. 提取其他槽位 (ID类)
        slots.other_slots = self._extract_id_slots(message)
        
        return slots
    
    def _extract_id_slots(self, message: str) -> Dict[str, Any]:
        """提取ID类槽位"""
        slots = {}
        
        # 采购需求ID (PR-XXXXXX)
        pr_match = re.search(r'PR[_-]?\d{8}[_-]?\d{3,}', message, re.IGNORECASE)
        if pr_match:
            slots['pr_id'] = pr_match.group().upper().replace('-', '').replace('_', '')
        
        # 询价单号 (RFQ-XXXXXX)
        rfq_match = re.search(r'(?:RFQ|INQ)[_-]?\d{8}[_-]?\d{3,}', message, re.IGNORECASE)
        if rfq_match:
            slots['inquiry_id'] = rfq_match.group().upper().replace('-', '').replace('_', '')
        
        # 报价单号 (QUO-XXXXXX)
        quo_match = re.search(r'QUO[_-]?\d{3,}', message, re.IGNORECASE)
        if quo_match:
            slots['quotation_id'] = quo_match.group().upper().replace('-', '').replace('_', '')
        
        # 采购订单号 (PO-XXXXXX)
        po_match = re.search(r'PO[_-]?\d{8}[_-]?\d{3,}', message, re.IGNORECASE)
        if po_match:
            slots['po_id'] = po_match.group().upper().replace('-', '').replace('_', '')
        
        return slots
    
    def extract_for_display(self, message: str) -> str:
        """提取槽位并格式化为可读文本"""
        slots = self.extract(message)
        parts = []
        
        if slots.time_range:
            parts.append(f"时间范围：{slots.time_range.label} ({slots.time_range.date_from} 至 {slots.time_range.date_to})")
        if slots.pr_type:
            parts.append(f"采购类型：{slots.pr_type.display_value}")
        if slots.material_category:
            parts.append(f"物料分类：{slots.material_category.display_value}")
        if slots.department:
            parts.append(f"部门：{slots.department.display_value}")
        if slots.execution_status:
            parts.append(f"执行状态：{slots.execution_status.display_value}")
        if slots.vendor_name:
            parts.append(f"供应商：{slots.vendor_name.display_value}")
        
        if slots.other_slots.get('pr_id'):
            parts.append(f"采购需求ID：{slots.other_slots['pr_id']}")
        if slots.other_slots.get('inquiry_id'):
            parts.append(f"询价单号：{slots.other_slots['inquiry_id']}")
        if slots.other_slots.get('quotation_id'):
            parts.append(f"报价单号：{slots.other_slots['quotation_id']}")
        if slots.other_slots.get('po_id'):
            parts.append(f"采购订单号：{slots.other_slots['po_id']}")
        
        return "\n".join(parts) if parts else "（未识别到特定条件）"


# ============================================================
# 便捷函数
# ============================================================


def extract_slots(message: str, today: Optional[date] = None) -> ExtractedSlots:
    """便捷函数：从消息中提取槽位"""
    extractor = ProcurementSlotExtractor(today)
    return extractor.extract(message)


def extract_slots_to_dict(message: str, today: Optional[date] = None) -> Dict[str, Any]:
    """便捷函数：从消息中提取槽位并转为字典"""
    return extract_slots(message, today).to_dict()


def extract_slots_to_connector_params(message: str, today: Optional[date] = None) -> Dict[str, Any]:
    """便捷函数：从消息中提取槽位并转为Connector参数"""
    return extract_slots(message, today).to_connector_params()
