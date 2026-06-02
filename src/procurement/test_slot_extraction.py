"""
采购意图识别与条件提取测试脚本

测试案例：
1. 时间范围提取：上周、本月、上个月等
2. 业务条件提取：采购类型、物料分类、部门等
3. 综合意图识别：识别带条件的采购需求查询

运行方式：
    python -m src.procurement.test_slot_extraction
"""

import sys
from datetime import date
from src.procurement.slot_extractors import (
    ProcurementSlotExtractor,
    extract_slots,
    extract_slots_to_connector_params,
)


def test_time_range_extraction():
    """测试时间范围提取"""
    print("\n" + "=" * 60)
    print("测试：时间范围提取")
    print("=" * 60)
    
    test_cases = [
        ("上周的采购需求", "上周"),
        ("本月的生产采购", "本月"),
        ("上个月各部门的需求", "上个月"),
        ("看看下周有什么计划", "下周"),
        ("昨天的订单", "昨天"),
        ("明天的安排", "明天"),
    ]
    
    extractor = ProcurementSlotExtractor()
    
    for message, expected_label in test_cases:
        slots = extractor.extract(message)
        time_range = slots.time_range
        
        print(f"\n输入: 「{message}」")
        print(f"  提取的时间标签: {time_range.label if time_range else '未识别'}")
        print(f"  日期范围: {time_range.date_from if time_range else 'N/A'} 至 {time_range.date_to if time_range else 'N/A'}")
        
        if time_range and time_range.label == expected_label:
            print(f"  ✓ 识别正确")
        else:
            print(f"  ✗ 预期: {expected_label}")


def test_pr_type_extraction():
    """测试采购类型提取"""
    print("\n" + "=" * 60)
    print("测试：采购类型提取")
    print("=" * 60)
    
    test_cases = [
        ("采购类型为NB的采购需求", "NB"),
        ("生产采购的物料需求", "生产采购"),
        ("紧急采购的订单", "紧急采购"),
        ("NB类型的采购", "NB"),
    ]
    
    extractor = ProcurementSlotExtractor()
    
    for message, expected_type in test_cases:
        slots = extractor.extract(message)
        pr_type = slots.pr_type
        
        print(f"\n输入: 「{message}」")
        print(f"  提取的采购类型: {pr_type.value if pr_type else '未识别'}")
        
        if pr_type and pr_type.value == expected_type:
            print(f"  ✓ 识别正确")
        else:
            print(f"  ✗ 预期: {expected_type}")


def test_material_category_extraction():
    """测试物料分类提取"""
    print("\n" + "=" * 60)
    print("测试：物料分类提取")
    print("=" * 60)
    
    test_cases = [
        ("办公用品的采购需求", "办公用品"),
        ("IT设备包括笔记本电脑和服务器", "IT设备"),
        ("生产物料的需求", "生产物料"),
        ("工程物资的采购", "工程物资"),
        ("办公文具的需求", "办公用品"),
    ]
    
    extractor = ProcurementSlotExtractor()
    
    for message, expected_category in test_cases:
        slots = extractor.extract(message)
        category = slots.material_category
        
        print(f"\n输入: 「{message}」")
        print(f"  提取的物料分类: {category.value if category else '未识别'}")
        
        if category and category.value == expected_category:
            print(f"  ✓ 识别正确")
        else:
            print(f"  ✗ 预期: {expected_category}")


def test_department_extraction():
    """测试部门提取"""
    print("\n" + "=" * 60)
    print("测试：部门提取")
    print("=" * 60)
    
    test_cases = [
        ("研发部申请的采购", "研发部"),
        ("市场部需求", "市场部"),
        ("销售部的订单", "销售部"),
        ("看看采购部的计划", "采购部"),
    ]
    
    extractor = ProcurementSlotExtractor()
    
    for message, expected_dept in test_cases:
        slots = extractor.extract(message)
        dept = slots.department
        
        print(f"\n输入: 「{message}」")
        print(f"  提取的部门: {dept.value if dept else '未识别'}")
        
        if dept and dept.value == expected_dept:
            print(f"  ✓ 识别正确")
        else:
            print(f"  ✗ 预期: {expected_dept}")


def test_combined_extraction():
    """测试综合条件提取"""
    print("\n" + "=" * 60)
    print("测试：综合条件提取")
    print("=" * 60)
    
    test_cases = [
        "先处理上周采购类型为办公用品的采购需求",
        "看看研发部上周有哪些未执行的采购计划",
        "查一下本月生产采购的物料需求",
        "上个月有哪些采购需求没有执行",
    ]
    
    extractor = ProcurementSlotExtractor()
    
    for message in test_cases:
        slots = extractor.extract(message)
        
        print(f"\n输入: 「{message}」")
        print(f"\n提取的槽位:")
        
        if slots.time_range:
            print(f"  时间范围: {slots.time_range.label}")
            print(f"    日期: {slots.time_range.date_from} 至 {slots.time_range.date_to}")
        
        if slots.pr_type:
            print(f"  采购类型: {slots.pr_type.value}")
        
        if slots.material_category:
            print(f"  物料分类: {slots.material_category.value}")
        
        if slots.department:
            print(f"  部门: {slots.department.value}")
        
        if slots.execution_status:
            print(f"  执行状态: {slots.execution_status.value}")
        
        # Connector参数
        params = slots.to_connector_params()
        print(f"\nConnector 参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")


def test_main_case():
    """测试主案例：上周采购类型为办公用品的采购需求
    
    注意：这个案例中"办公用品"是物料分类，不是采购类型。
    "采购类型"通常指 NB（标准采购）、紧急采购等。
    """
    print("\n" + "=" * 60)
    print("测试：主案例 - 上周办公用品采购需求")
    print("=" * 60)
    
    message = "先处理上周采购类型为办公用品的采购需求"
    
    # 使用便捷函数
    params = extract_slots_to_connector_params(message)
    
    print(f"\n输入: 「{message}」")
    print(f"\n提取的 Connector 参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 验证关键参数 - 办公用品是物料分类，不是采购类型
    assert "date_from" in params, "缺少 date_from"
    assert "date_to" in params, "缺少 date_to"
    assert "material_category" in params, "缺少 material_category (办公用品)"
    
    print(f"\n✓ 所有关键参数均已提取")
    print(f"  - 时间范围: 上周")
    print(f"  - 物料分类: 办公用品")
    print(f"  - 注: '采购类型'在这个上下文中指采购需求类型(默认NB)，而不是物料分类")


def test_pr_type_in_real_cases():
    """测试真实场景中的采购类型识别"""
    print("\n" + "=" * 60)
    print("测试：采购类型的真实场景识别")
    print("=" * 60)
    
    test_cases = [
        # 明确说采购类型的
        ("采购类型为NB的采购需求", "NB"),
        ("采购类型是紧急采购", "紧急采购"),
        # 通过语义推断的
        ("生产采购的物料需求", "生产采购"),
    ]
    
    extractor = ProcurementSlotExtractor()
    
    for message, expected_type in test_cases:
        slots = extractor.extract(message)
        pr_type = slots.pr_type
        
        print(f"\n输入: 「{message}」")
        print(f"  提取的采购类型: {pr_type.value if pr_type else '未识别'}")
        
        if pr_type and pr_type.value == expected_type:
            print(f"  ✓ 识别正确")
        else:
            print(f"  ✗ 预期: {expected_type}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("采购意图识别与条件提取 - 测试套件")
    print("=" * 60)
    print(f"测试基准日期: {date.today()}")
    
    try:
        test_time_range_extraction()
        test_pr_type_extraction()
        test_material_category_extraction()
        test_department_extraction()
        test_combined_extraction()
        test_main_case()
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
