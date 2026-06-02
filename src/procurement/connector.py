"""
采购连接器服务
将本体的 action 转换为模拟数据调用
所有数据获取和操作都通过此服务进行
数据读取遵循本体规范：
1) 默认删除标记 delete_flag=0 或为空的数据视为有效
2) 采购需求执行状态以采购需求行项目是否生成采购订单行项目判断
3) 有效报价需关联询价单，且价格有效期覆盖当前业务日期
4) 比价默认按同一物料、同一询价单下的供应商报价净价从低到高排序
5) 采购订单执行进度以订单行项目的发货状态、已完成标识以及收货历史数量进行判断
"""

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ConnectorResponse:
    success: bool
    data: Any
    message: str = ""
    error: Optional[str] = None


class ProcurementConnector:
    """采购连接器服务 - 基于本体的数据访问层"""

    def __init__(self, mock_data_dir: Optional[str] = None):
        if mock_data_dir:
            self.data_dir = Path(mock_data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "procurement" / "mock"

        self._purchase_requests: Optional[List[Dict]] = None
        self._purchase_inquiries: Optional[List[Dict]] = None
        self._purchase_quotations: Optional[List[Dict]] = None
        self._purchase_order_heads: Optional[List[Dict]] = None
        self._purchase_order_items: Optional[List[Dict]] = None
        self._purchase_order_receipt_history: Optional[List[Dict]] = None

    def _load_json(self, filename: str) -> List[Dict]:
        """加载JSON文件"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 返回第一个key对应的列表
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        return value
            return []

    def _save_json(self, filename: str, data: List[Dict]) -> None:
        """保存JSON文件"""
        filepath = self.data_dir / filename
        key = filename.replace('.json', '')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({key: data}, f, ensure_ascii=False, indent=2)

    def _is_valid_record(self, record: Dict) -> bool:
        """判断记录是否有效（默认排除delete_flag=1的记录）"""
        delete_flag = record.get('delete_flag', '0')
        return delete_flag == '0' or delete_flag == '' or delete_flag is None

    def _get_purchase_requests(self, include_deleted: bool = False) -> List[Dict]:
        if self._purchase_requests is None:
            self._purchase_requests = self._load_json('purchase_requests.json')
        if include_deleted:
            return self._purchase_requests
        return [r for r in self._purchase_requests if self._is_valid_record(r)]

    def _get_purchase_inquiries(self, include_deleted: bool = False) -> List[Dict]:
        if self._purchase_inquiries is None:
            self._purchase_inquiries = self._load_json('purchase_inquiries.json')
        if include_deleted:
            return self._purchase_inquiries
        return [i for i in self._purchase_inquiries if self._is_valid_record(i)]

    def _get_purchase_quotations(self, include_deleted: bool = False) -> List[Dict]:
        if self._purchase_quotations is None:
            self._purchase_quotations = self._load_json('purchase_quotations.json')
        if include_deleted:
            return self._purchase_quotations
        return [q for q in self._purchase_quotations if self._is_valid_record(q)]

    def _get_purchase_order_heads(self, include_deleted: bool = False) -> List[Dict]:
        if self._purchase_order_heads is None:
            self._purchase_order_heads = self._load_json('purchase_orders.json')
        if include_deleted:
            return self._purchase_order_heads
        return [o for o in self._purchase_order_heads if self._is_valid_record(o)]

    def _get_purchase_order_items(self, include_deleted: bool = False) -> List[Dict]:
        if self._purchase_order_items is None:
            self._purchase_order_items = self._load_json('purchase_orders.json')
            # 尝试从purchase_orders中提取purchase_order_items
            if self._purchase_order_items and isinstance(self._purchase_order_items, dict):
                self._purchase_order_items = self._purchase_order_items.get('purchase_order_items', [])
            elif self._purchase_order_items and isinstance(self._purchase_order_items, list):
                # 如果是列表，检查第一个元素是否包含purchase_order_items
                if len(self._purchase_order_items) > 0 and isinstance(self._purchase_order_items[0], dict):
                    if 'purchase_order_items' in self._purchase_order_items[0]:
                        all_orders = self._purchase_order_items
                        self._purchase_order_items = []
                        for order in all_orders:
                            if 'purchase_order_items' in order:
                                self._purchase_order_items.extend(order['purchase_order_items'])
        if include_deleted:
            return self._purchase_order_items
        return [i for i in self._purchase_order_items if self._is_valid_record(i)]

    def _get_purchase_order_receipt_history(self, include_deleted: bool = False) -> List[Dict]:
        if self._purchase_order_receipt_history is None:
            self._purchase_order_receipt_history = self._load_json('purchase_order_history.json')
        if include_deleted:
            return self._purchase_order_receipt_history
        return [h for h in self._purchase_order_receipt_history if self._is_valid_record(h)]

    def _get_valid_quotations(self, inquiry_id: str, as_of_date: Optional[str] = None) -> List[Dict]:
        """获取有效报价（价格有效期覆盖当前业务日期）"""
        if as_of_date is None:
            as_of_date = date.today().isoformat()
        
        quotations = self._get_purchase_quotations()
        valid = []
        for q in quotations:
            # 必须关联询价单
            if q.get('inquiry_id') != inquiry_id:
                continue
            # 价格有效期检查
            start_date = q.get('price_start_date', '')
            end_date = q.get('price_end_date', '')
            if start_date and as_of_date < start_date:
                continue
            if end_date and as_of_date > end_date:
                continue
            valid.append(q)
        return valid

    # ==================== 查询类操作 ====================
    # 本体定义的数据集:
    # - purchase_requests (采购需求行项目)
    # - purchase_inquiries (询价单行项目)
    # - purchase_quotations (报价单行项目)
    # - purchase_order_heads (采购订单抬头)
    # - purchase_order_items (采购订单行项目)
    # - purchase_order_receipt_history (采购订单收货历史)

    def purchase_request_list(self, params: Dict[str, Any]) -> ConnectorResponse:
        """分页查询采购需求 (purchase_requests)"""
        requests = self._get_purchase_requests()
        page = params.get('page', 1)
        page_size = params.get('page_size', 50)
        q = params.get('q', '')
        date_from = params.get('date_from')
        date_to = params.get('date_to')

        # 过滤
        filtered = requests
        if q:
            filtered = [r for r in filtered if q.lower() in r.get('material_d', '').lower() or q in r.get('pr_id', '')]
        if date_from:
            filtered = [r for r in filtered if r.get('delivery_date', '') >= date_from]
        if date_to:
            filtered = [r for r in filtered if r.get('delivery_date', '') <= date_to]

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        items = filtered[start:end]

        return ConnectorResponse(
            success=True,
            data={
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size
            },
            message=f"查询到 {total} 条采购需求"
        )

    def purchase_request_get_by_id(self, params: Dict[str, Any]) -> ConnectorResponse:
        """按主键查询采购需求 (purchase_requests)"""
        pr_id = params.get('pr_id')
        pr_item = params.get('pr_item')

        requests = self._get_purchase_requests()
        for r in requests:
            if r.get('pr_id') == pr_id and r.get('pr_item') == pr_item:
                return ConnectorResponse(success=True, data=r, message="采购需求查询成功")

        return ConnectorResponse(success=False, data=None, error=f"未找到采购需求 {pr_id}/{pr_item}")

    def purchase_inquiry_list(self, params: Dict[str, Any]) -> ConnectorResponse:
        """分页查询询价单 (purchase_inquiries)"""
        inquiries = self._get_purchase_inquiries()
        page = params.get('page', 1)
        page_size = params.get('page_size', 50)

        total = len(inquiries)
        start = (page - 1) * page_size
        end = start + page_size
        items = inquiries[start:end]

        return ConnectorResponse(
            success=True,
            data={
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        )

    def purchase_inquiry_get_by_id(self, params: Dict[str, Any]) -> ConnectorResponse:
        """按主键查询询价单 (purchase_inquiries)"""
        inquiry_id = params.get('inquiry_id')
        inquiry_item = params.get('inquiry_item')

        inquiries = self._get_purchase_inquiries()
        for i in inquiries:
            if i.get('inquiry_id') == inquiry_id and i.get('inquiry_item') == inquiry_item:
                return ConnectorResponse(success=True, data=i, message="询价单查询成功")

        return ConnectorResponse(success=False, data=None, error=f"未找到询价单 {inquiry_id}/{inquiry_item}")

    def purchase_quotation_list(self, params: Dict[str, Any]) -> ConnectorResponse:
        """分页查询报价单 (purchase_quotations)"""
        quotations = self._get_purchase_quotations()
        page = params.get('page', 1)
        page_size = params.get('page_size', 50)

        total = len(quotations)
        start = (page - 1) * page_size
        end = start + page_size
        items = quotations[start:end]

        return ConnectorResponse(
            success=True,
            data={
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        )

    def purchase_quotation_get_by_id(self, params: Dict[str, Any]) -> ConnectorResponse:
        """按主键查询报价单 (purchase_quotations)"""
        quotation_id = params.get('quotation_id')
        quotation_item = params.get('quotation_item')

        quotations = self._get_purchase_quotations()
        for q in quotations:
            if q.get('quotation_id') == quotation_id and q.get('quotation_item') == quotation_item:
                return ConnectorResponse(success=True, data=q, message="报价单查询成功")

        return ConnectorResponse(success=False, data=None, error=f"未找到报价单 {quotation_id}/{quotation_item}")

    def purchase_quotation_compare(self, params: Dict[str, Any]) -> ConnectorResponse:
        """报价单自动比价 - 按同一询价单和物料维度，对多家供应商报价进行比价"""
        inquiry_id = params.get('inquiry_id')
        inquiry_item = params.get('inquiry_item')
        material_id = params.get('material_id')

        quotations = self._get_purchase_quotations()
        # 筛选同一询价单的报价
        filtered = quotations
        
        # 本体规范：按同一物料、同一询价单下的供应商报价净价从低到高排序
        if inquiry_id:
            filtered = [q for q in filtered if q.get('inquiry_id') == inquiry_id]
        if inquiry_item:
            filtered = [q for q in filtered if q.get('inquiry_item') == inquiry_item]
        if material_id:
            filtered = [q for q in filtered if q.get('material_id') == material_id]

        # 按净价从低到高排序
        sorted_quotations = sorted(filtered, key=lambda x: float(x.get('net_price', 0) or 0))

        # 计算历史均价
        prices = [float(q.get('net_price', 0) or 0) for q in sorted_quotations]
        history_avg_price = sum(prices) / len(prices) if prices else 0

        # 推荐最低价
        recommended = sorted_quotations[0] if sorted_quotations else None
        recommended_reason = ""
        if recommended:
            if len(sorted_quotations) >= 3:
                recommended_reason = "价格最低，符合比价规范"
            else:
                recommended_reason = "价格最低（注意：供应商不足3家，比价结果仅供参考）"

        result = {
            "inquiry_id": inquiry_id,
            "inquiry_item": inquiry_item,
            "material_id": material_id,
            "quotations": sorted_quotations,
            "recommended": recommended,
            "recommended_reason": recommended_reason,
            "supplier_count": len(sorted_quotations),
            "is_valid": len(sorted_quotations) >= 3,
            "avg_price": history_avg_price,
        }

        return ConnectorResponse(
            success=True,
            data=result,
            message=f"比价完成，共 {len(sorted_quotations)} 家供应商报价"
        )

    def purchase_order_head_list(self, params: Dict[str, Any]) -> ConnectorResponse:
        """分页查询采购订单抬头 (purchase_order_heads)"""
        orders = self._get_purchase_order_heads()
        page = params.get('page', 1)
        page_size = params.get('page_size', 50)

        total = len(orders)
        start = (page - 1) * page_size
        end = start + page_size
        items = orders[start:end]

        return ConnectorResponse(
            success=True,
            data={
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        )

    def purchase_order_head_get_by_id(self, params: Dict[str, Any]) -> ConnectorResponse:
        """按主键查询采购订单抬头 (purchase_order_heads)"""
        po_id = params.get('po_id')

        orders = self._get_purchase_order_heads()
        for o in orders:
            if o.get('po_id') == po_id:
                return ConnectorResponse(success=True, data=o, message="采购订单查询成功")

        return ConnectorResponse(success=False, data=None, error=f"未找到采购订单 {po_id}")

    def purchase_order_item_list(self, params: Dict[str, Any]) -> ConnectorResponse:
        """分页查询采购订单行项目 (purchase_order_items)"""
        items = self._get_purchase_order_items()
        page = params.get('page', 1)
        page_size = params.get('page_size', 50)

        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        result_items = items[start:end]

        return ConnectorResponse(
            success=True,
            data={
                "items": result_items,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        )

    def purchase_order_receipt_history_list(self, params: Dict[str, Any]) -> ConnectorResponse:
        """分页查询采购订单收货历史 (purchase_order_receipt_history)"""
        history = self._get_purchase_order_receipt_history()
        page = params.get('page', 1)
        page_size = params.get('page_size', 50)

        total = len(history)
        start = (page - 1) * page_size
        end = start + page_size
        items = history[start:end]

        return ConnectorResponse(
            success=True,
            data={
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        )

    def purchase_order_query_execution_status(self, params: Dict[str, Any]) -> ConnectorResponse:
        """查询采购订单执行情况"""
        po_id = params.get('po_id')

        # 获取订单抬头
        orders = self._get_purchase_order_heads()
        order = None
        for o in orders:
            if o.get('po_id') == po_id:
                order = o
                break

        if not order:
            return ConnectorResponse(success=False, data=None, error=f"未找到采购订单 {po_id}")

        # 获取订单行项目
        items = self._get_purchase_order_items()
        order_items = [i for i in items if i.get('po_id') == po_id]

        # 获取收货历史
        history = self._get_purchase_order_receipt_history()
        receipts = [h for h in history if h.get('po_id') == po_id]

        # 计算收货进度
        total_quantity = sum(float(i.get('quantity', 0) or 0) for i in order_items)
        received_quantity = sum(float(h.get('quantity', 0) or 0) for h in receipts)
        receipt_progress = (received_quantity / total_quantity * 100) if total_quantity > 0 else 0

        result = {
            "po_id": po_id,
            "approval_status": order.get('flow_status', '未知'),
            "document_status": order.get('status', '未知'),
            "vendor": order.get('vendor_d', ''),
            "contract_amount": order.get('contract_amount', 0),
            "currency": order.get('currency', 'CNY'),
            "items": order_items,
            "receipts": receipts,
            "summary": {
                "total_quantity": total_quantity,
                "received_quantity": received_quantity,
                "receipt_progress": f"{receipt_progress:.1f}%"
            },
            "status_details": {
                "审批状态": order.get('flow_status', '未知'),
                "发货状态": order_items[0].get('inbound_state', '未知') if order_items else '未知',
                "收货状态": "部分收货" if received_quantity > 0 else "未收货",
                "发票状态": "未开票"
            }
        }

        return ConnectorResponse(
            success=True,
            data=result,
            message=f"查询到采购订单 {po_id} 的执行情况"
        )

    # ==================== 命令类操作 ====================
    # 本体定义的 action 操作:
    # - procurement/create_inquiry_from_pr: 根据采购需求创建询价单
    # - procurement/submit_award_approval: 提交中标报价审批
    # - procurement/create_purchase_order_from_pr_and_quotation: 创建采购订单
    # - analytics/find_unexecuted_purchase_requests: 查询未执行采购需求
    # - analytics/generate_price_comparison: 生成比价分析
    # - analytics/get_purchase_order_execution_status: 查询采购订单执行情况

    def collect_quotations(self, params: Dict[str, Any]) -> ConnectorResponse:
        """回收报价 - 汇总某个询价单下供应商报价进度和报价明细"""
        inquiry_id = params.get('inquiry_id')

        # 获取询价单
        inquiries = self._get_purchase_inquiries()
        inquiry = None
        for i in inquiries:
            if i.get('inquiry_id') == inquiry_id:
                inquiry = i
                break

        if not inquiry:
            return ConnectorResponse(success=False, data=None, error=f"未找到询价单 {inquiry_id}")

        # 获取该询价单的所有报价
        quotations = self._get_purchase_quotations()
        inquiry_quotations = [q for q in quotations if q.get('inquiry_id') == inquiry_id]

        # 汇总
        vendor_count = len(set(q.get('vendor_id') for q in inquiry_quotations))
        quotation_list = sorted(inquiry_quotations, key=lambda x: float(x.get('net_price', 0) or 0))

        response_summary = {
            "total_vendors": vendor_count,
            "total_quotations": len(inquiry_quotations),
            "status": inquiry.get('status', ''),
            "inquiry_start_date": inquiry.get('inquiry_start_date', ''),
            "inquiry_end_date": inquiry.get('inquiry_end_date', ''),
        }

        return ConnectorResponse(
            success=True,
            data={
                "quotation_list": quotation_list,
                "response_summary": response_summary
            },
            message=f"询价单 {inquiry_id} 共有 {vendor_count} 家供应商报价，共 {len(inquiry_quotations)} 条报价记录"
        )

    def purchase_request_approve(self, params: Dict[str, Any]) -> ConnectorResponse:
        """审批采购需求"""
        pr_id = params.get('pr_id')
        pr_item = params.get('pr_item')
        decision = params.get('decision')  # approve, reject
        comment = params.get('comment', '')

        requests = self._get_purchase_requests()
        for r in requests:
            if r.get('pr_id') == pr_id and r.get('pr_item') == pr_item:
                if decision == 'approve':
                    r['flow_status'] = 'APPROVED'
                    message = f"采购需求 {pr_id}/{pr_item} 已审批通过"
                else:
                    r['flow_status'] = 'REJECTED'
                    message = f"采购需求 {pr_id}/{pr_item} 已驳回"
                self._save_json('purchase_requests.json', requests)
                return ConnectorResponse(success=True, data=r, message=message)

        return ConnectorResponse(success=False, data=None, error=f"未找到采购需求 {pr_id}/{pr_item}")

    def create_inquiry_from_pr(self, params: Dict[str, Any]) -> ConnectorResponse:
        """根据采购需求创建询价单"""
        pr_id = params.get('pr_id')
        pr_item = params.get('pr_item')
        supplier_scope = params.get('supplier_scope', [])
        deadline = params.get('deadline')

        # 获取采购需求
        requests = self._get_purchase_requests()
        pr = None
        for r in requests:
            if r.get('pr_id') == pr_id and r.get('pr_item') == pr_item:
                pr = r
                break

        if not pr:
            return ConnectorResponse(success=False, data=None, error=f"未找到采购需求 {pr_id}/{pr_item}")

        # 规则检查：采购需求必须已审批
        flow_status = pr.get('flow_status', '')
        if flow_status not in ('S0', 'APPROVED'):
            return ConnectorResponse(
                success=False,
                data=None,
                error="采购需求未审批通过或已删除，不能发起询价。"
            )

        # 生成询价单号
        import datetime
        today = datetime.date.today().strftime('%Y%m%d')
        inquiry_id = f"RFQ-{today}-{len(self._get_purchase_inquiries()) + 1:03d}"

        new_inquiry = {
            "inquiry_id": inquiry_id,
            "inquiry_item": "0001",
            "inquiry_type": "公开询价",
            "material_id": pr.get('material_id'),
            "material_d": pr.get('material_d'),
            "delivery_date": pr.get('delivery_date'),
            "company_id": pr.get('company_id'),
            "factory_id": pr.get('factory_id'),
            "quantity": str(pr.get('quantity')),
            "unit_id": pr.get('unit_id'),
            "purgroup_id": "PG01",
            "purgroup_d": "IT设备采购组",
            "inquiry_start_date": datetime.date.today().strftime('%Y-%m-%d'),
            "inquiry_end_date": deadline or (datetime.date.today() + datetime.timedelta(days=7)).strftime('%Y-%m-%d'),
            "pr_id": pr_id,
            "pr_item": pr_item,
            "status": "已发布",
            "delete_flag": "0"
        }

        inquiries = self._get_purchase_inquiries(include_deleted=True)
        inquiries.append(new_inquiry)
        self._save_json('purchase_inquiries.json', inquiries)

        return ConnectorResponse(
            success=True,
            data={"inquiry_id": inquiry_id, "status": "已创建"},
            message=f"已基于采购需求 {pr_id} 创建询价单 {inquiry_id}"
        )

    def submit_award_approval(self, params: Dict[str, Any]) -> ConnectorResponse:
        """提交中标报价审批"""
        quotation_id = params.get('quotation_id')
        quotation_item = params.get('quotation_item')
        reason = params.get('reason', '')

        quotations = self._get_purchase_quotations()
        for q in quotations:
            if q.get('quotation_id') == quotation_id and q.get('quotation_item') == quotation_item:
                # 提交审批
                q['status'] = 'PENDING_APPROVAL'
                self._save_json('purchase_quotations.json', quotations)
                
                # 生成审批流程ID
                import datetime
                workflow_id = f"WA-{datetime.date.today().strftime('%Y%m%d')}-{quotation_id}"
                
                return ConnectorResponse(
                    success=True,
                    data={
                        "workflow_id": workflow_id,
                        "approval_status": "PENDING_APPROVAL"
                    },
                    message=f"报价单 {quotation_id} 已提交审批，审批流程ID: {workflow_id}"
                )

        return ConnectorResponse(success=False, data=None, error=f"未找到报价单 {quotation_id}/{quotation_item}")

    def purchase_quotation_approve_award(self, params: Dict[str, Any]) -> ConnectorResponse:
        """审批中标报价"""
        quotation_id = params.get('quotation_id')
        quotation_item = params.get('quotation_item')
        decision = params.get('decision')  # approve, reject
        comment = params.get('comment', '')

        quotations = self._get_purchase_quotations()
        for q in quotations:
            if q.get('quotation_id') == quotation_id and q.get('quotation_item') == quotation_item:
                if decision == 'approve':
                    q['status'] = 'APPROVED'
                    message = f"报价单 {quotation_id} 已审批通过，中标生效"
                else:
                    q['status'] = 'REJECTED'
                    message = f"报价单 {quotation_id} 已驳回"
                self._save_json('purchase_quotations.json', quotations)
                return ConnectorResponse(success=True, data=q, message=message)

        return ConnectorResponse(success=False, data=None, error=f"未找到报价单 {quotation_id}/{quotation_item}")

    def create_purchase_order_from_pr_and_quotation(self, params: Dict[str, Any]) -> ConnectorResponse:
        """基于采购需求和报价创建采购订单"""
        pr_id = params.get('pr_id')
        pr_item = params.get('pr_item')
        quotation_id = params.get('quotation_id')
        quotation_item = params.get('quotation_item')
        submit_approval = params.get('submit_approval', False)

        # 获取采购需求
        requests = self._get_purchase_requests()
        pr = None
        for r in requests:
            if r.get('pr_id') == pr_id and r.get('pr_item') == pr_item:
                pr = r
                break

        if not pr:
            return ConnectorResponse(success=False, data=None, error=f"未找到采购需求 {pr_id}/{pr_item}")

        # 获取报价单
        quotations = self._get_purchase_quotations()
        quotation = None
        for q in quotations:
            if q.get('quotation_id') == quotation_id and q.get('quotation_item') == quotation_item:
                quotation = q
                break

        if not quotation:
            return ConnectorResponse(success=False, data=None, error=f"未找到报价单 {quotation_id}/{quotation_item}")

        # 规则检查：报价必须有效（审批通过且在价格有效期内）
        status = quotation.get('status', '')
        today = date.today().isoformat()
        start_date = quotation.get('price_start_date', '')
        end_date = quotation.get('price_end_date', '')
        
        if status not in ('APPROVED', 'EFFECTIVE', 'S0'):
            return ConnectorResponse(
                success=False,
                data=None,
                error="报价未审批通过，不能创建采购订单。"
            )
        
        if start_date and today < start_date:
            return ConnectorResponse(
                success=False,
                data=None,
                error="报价价格尚未生效。"
            )
        
        if end_date and today > end_date:
            return ConnectorResponse(
                success=False,
                data=None,
                error="报价已过期。"
            )

        # 生成订单号
        import datetime
        today_str = datetime.date.today().strftime('%Y%m%d')
        po_id = f"PO-{today_str}-{len(self._get_purchase_order_heads()) + 1:03d}"

        # 计算订单金额
        quantity = float(pr.get('quantity', 0) or 0)
        unit_price = float(quotation.get('net_price', 0) or 0)
        total_amount = quantity * unit_price

        # 创建订单抬头
        new_order = {
            "po_id": po_id,
            "company_id": pr.get('company_id'),
            "company_d": "总公司",
            "potype_id": pr.get('pr_type'),
            "vendor_id": quotation.get('vendor_id'),
            "vendor_d": quotation.get('vendor_d'),
            "purgroup_id": quotation.get('purgroup_id'),
            "purgroup_d": quotation.get('purgroup_d'),
            "contract_amount": total_amount,
            "currency": "CNY",
            "appro_whe": "Y",
            "flow_status": "PENDING" if submit_approval else "DRAFT",
            "status": "待审批" if submit_approval else "草稿",
            "delete_flag": "0"
        }

        # 创建订单行项目
        new_order_item = {
            "po_id": po_id,
            "po_item": "0001",
            "delete_flag": "0",
            "material_id": pr.get('material_id'),
            "material_d": pr.get('material_d'),
            "factory_id": pr.get('factory_id'),
            "location_id": "L001",
            "quantity": quantity,
            "unit_id": pr.get('unit_id'),
            "net_price": unit_price,
            "tax_id": "TAX13",
            "price_unit": "1",
            "delivery_date": pr.get('delivery_date'),
            "pr_id": pr_id,
            "pr_item": pr_item,
            "inbound_state": "未发货",
            "returns": "N",
            "free": "N",
            "delivery_complete": "N",
            "quotation_id": quotation_id,
            "quotation_item": quotation_item
        }

        # 保存订单
        orders = self._get_purchase_order_heads(include_deleted=True)
        orders.append(new_order)
        self._save_json('purchase_orders.json', orders)

        # 保存订单行项目
        items = self._get_purchase_order_items(include_deleted=True)
        items.append(new_order_item)
        self._save_json('purchase_orders.json', {
            "purchase_orders": orders,
            "purchase_order_items": items
        })

        result = {
            "order": new_order,
            "item": new_order_item,
            "po_id": po_id
        }

        return ConnectorResponse(
            success=True,
            data=result,
            message=f"已创建采购订单 {po_id}，关联报价单 {quotation_id}"
        )

    def find_unexecuted_purchase_requests(self, params: Dict[str, Any]) -> ConnectorResponse:
        """查询未执行采购需求 - 已审批但尚未生成采购订单的采购需求行项目"""
        company_id = params.get('company_id')
        factory_id = params.get('factory_id')
        apply_dep = params.get('apply_dep')
        as_of_date = params.get('as_of_date', date.today().isoformat())

        # 获取所有采购需求（已审批的）
        requests = self._get_purchase_requests()
        approved_requests = [
            r for r in requests
            if r.get('flow_status') in ('S0', 'APPROVED')
        ]

        # 获取所有采购订单行项目（已关联PR的）
        po_items = self._get_purchase_order_items()
        executed_pr_keys = set()
        for item in po_items:
            pr_id = item.get('pr_id')
            pr_item = item.get('pr_item')
            if pr_id and pr_item:
                executed_pr_keys.add((pr_id, pr_item))

        # 筛选未执行的采购需求
        unexecuted = []
        for r in approved_requests:
            pr_id = r.get('pr_id')
            pr_item = r.get('pr_item')
            if (pr_id, pr_item) not in executed_pr_keys:
                # 应用过滤条件
                if company_id and r.get('company_id') != company_id:
                    continue
                if factory_id and r.get('factory_id') != factory_id:
                    continue
                if apply_dep and r.get('apply_dep') != apply_dep:
                    continue
                unexecuted.append(r)

        # 风险汇总
        risk_summary = {
            "total_unexecuted": len(unexecuted),
            "urgent_count": len([r for r in unexecuted if r.get('delivery_date', '') < as_of_date]),
            "by_department": {}
        }

        for r in unexecuted:
            dep = r.get('apply_dep', '未知')
            if dep not in risk_summary["by_department"]:
                risk_summary["by_department"][dep] = 0
            risk_summary["by_department"][dep] += 1

        return ConnectorResponse(
            success=True,
            data={
                "purchase_requests": unexecuted,
                "risk_summary": risk_summary
            },
            message=f"查询到 {len(unexecuted)} 条未执行的采购需求"
        )

    def find_purchase_requests_with_conditions(self, params: Dict[str, Any]) -> ConnectorResponse:
        """带条件的采购需求查询
        
        支持的过滤条件：
        - date_from, date_to: 需求日期范围
        - pr_type: 采购类型 (NB, 标准采购, 紧急采购, 生产采购等)
        - material_category: 物料分类 (办公用品, IT设备, 生产物料等)
        - apply_dep: 申请部门
        - execution_status: 执行状态 (未执行, 已执行, 待审批等)
        - company_id: 公司代码
        - factory_id: 工厂代码
        """
        # 提取过滤参数
        date_from = params.get('date_from')
        date_to = params.get('date_to')
        pr_type = params.get('pr_type')
        material_category = params.get('material_category')
        apply_dep = params.get('apply_dep') or params.get('department')
        execution_status = params.get('execution_status')
        company_id = params.get('company_id')
        factory_id = params.get('factory_id')
        include_executed = params.get('include_executed', True)

        # 获取采购需求
        requests = self._get_purchase_requests()

        # 获取已执行的PR键集合
        po_items = self._get_purchase_order_items()
        executed_pr_keys = set()
        for item in po_items:
            pr_id = item.get('pr_id')
            pr_item = item.get('pr_item')
            if pr_id and pr_item:
                executed_pr_keys.add((pr_id, pr_item))

        # 筛选
        filtered = []
        for r in requests:
            # 1. 排除已删除
            if not self._is_valid_record(r):
                continue

            # 2. 日期范围过滤 (基于需求日期 delivery_date)
            if date_from and r.get('delivery_date', '') < date_from:
                continue
            if date_to and r.get('delivery_date', '') > date_to:
                continue

            # 3. 采购类型过滤
            if pr_type and r.get('pr_type') != pr_type:
                continue

            # 4. 物料分类过滤 (通过物料描述匹配)
            if material_category:
                material_d = r.get('material_d', '').lower()
                # 匹配物料分类关键词
                category_keywords = {
                    "办公用品": ["办公", "文具", "打印", "纸张", "笔", "本子", "文件夹"],
                    "IT设备": ["电脑", "笔记本", "台式机", "服务器", "显示器", "键盘", "鼠标", "IT"],
                    "生产物料": ["原材料", "原料", "生产", "化工", "金属", "塑料"],
                    "工程物资": ["工程", "建材", "水泥", "钢材", "木材"],
                }
                keywords = category_keywords.get(material_category, [])
                if not any(kw in material_d for kw in keywords):
                    continue

            # 5. 部门过滤
            if apply_dep and r.get('apply_dep') != apply_dep:
                continue

            # 6. 执行状态过滤
            pr_key = (r.get('pr_id'), r.get('pr_item'))
            is_executed = pr_key in executed_pr_keys
            if not include_executed and is_executed:
                continue

            # 7. 状态过滤
            flow_status = r.get('flow_status', '')
            if execution_status:
                if execution_status == "未执行" and is_executed:
                    continue
                if execution_status == "已执行" and not is_executed:
                    continue
                status_map = {
                    "待审批": ["PENDING", "P"],
                    "已审批": ["S0", "APPROVED", "APPROVED_"],
                    "已驳回": ["REJECTED", "REJECT"],
                }
                if execution_status in status_map:
                    if flow_status not in status_map[execution_status]:
                        continue

            # 8. 公司/工厂过滤
            if company_id and r.get('company_id') != company_id:
                continue
            if factory_id and r.get('factory_id') != factory_id:
                continue

            filtered.append(r)

        # 汇总统计
        summary = {
            "total": len(filtered),
            "executed": sum(1 for r in filtered if (r.get('pr_id'), r.get('pr_item')) in executed_pr_keys),
            "by_pr_type": {},
            "by_department": {},
            "by_material_category": {},
        }

        # 按采购类型统计
        for r in filtered:
            pt = r.get('pr_type', '未知')
            if pt not in summary["by_pr_type"]:
                summary["by_pr_type"][pt] = 0
            summary["by_pr_type"][pt] += 1

        # 按部门统计
        for r in filtered:
            dep = r.get('apply_dep', '未知')
            if dep not in summary["by_department"]:
                summary["by_department"][dep] = 0
            summary["by_department"][dep] += 1

        # 构建条件标签
        condition_labels = []
        if date_from and date_to:
            condition_labels.append(f"需求日期：{date_from}至{date_to}")
        if pr_type:
            condition_labels.append(f"采购类型：{pr_type}")
        if material_category:
            condition_labels.append(f"物料分类：{material_category}")
        if apply_dep:
            condition_labels.append(f"部门：{apply_dep}")
        if execution_status:
            condition_labels.append(f"状态：{execution_status}")

        total_label = "、".join(condition_labels) if condition_labels else "全部"

        return ConnectorResponse(
            success=True,
            data={
                "purchase_requests": filtered,
                "summary": summary,
                "conditions_applied": condition_labels,
            },
            message=f"查询到 {len(filtered)} 条{total_label}的采购需求"
        )

    def generate_price_comparison(self, params: Dict[str, Any]) -> ConnectorResponse:
        """生成比价分析"""
        inquiry_id = params.get('inquiry_id')
        material_id = params.get('material_id')
        min_supplier_count = params.get('min_supplier_count', 3)

        quotations = self._get_purchase_quotations()
        
        # 筛选同一询价单和物料的报价
        filtered = [q for q in quotations if q.get('inquiry_id') == inquiry_id]
        if material_id:
            filtered = [q for q in filtered if q.get('material_id') == material_id]

        # 按净价从低到高排序
        sorted_quotations = sorted(filtered, key=lambda x: float(x.get('net_price', 0) or 0))

        # 推荐最低价
        recommended_quotation = sorted_quotations[0] if sorted_quotations else None
        recommended_id = recommended_quotation.get('quotation_id', '') if recommended_quotation else ''
        
        # 生成推荐理由
        reason = ""
        if not sorted_quotations:
            reason = "无供应商报价"
        elif len(sorted_quotations) < min_supplier_count:
            reason = f"推荐供应商{recommended_quotation.get('vendor_d', '')}，价格最低（注意：供应商不足{min_supplier_count}家，比价结果仅供参考）"
        else:
            reason = f"推荐供应商{recommended_quotation.get('vendor_d', '')}，价格最低，符合比价规范"

        return ConnectorResponse(
            success=True,
            data={
                "ranked_quotations": sorted_quotations,
                "recommended_quotation_id": recommended_id,
                "reason": reason,
                "supplier_count": len(sorted_quotations)
            },
            message=f"比价分析完成，共 {len(sorted_quotations)} 家供应商报价"
        )

    def get_purchase_order_execution_status(self, params: Dict[str, Any]) -> ConnectorResponse:
        """查询采购订单执行情况"""
        po_id = params.get('po_id')

        # 获取订单抬头
        orders = self._get_purchase_order_heads()
        order = None
        for o in orders:
            if o.get('po_id') == po_id:
                order = o
                break

        if not order:
            return ConnectorResponse(success=False, data=None, error=f"未找到采购订单 {po_id}")

        # 获取订单行项目
        items = self._get_purchase_order_items()
        order_items = [i for i in items if i.get('po_id') == po_id]

        # 获取收货历史
        history = self._get_purchase_order_receipt_history()
        receipts = [h for h in history if h.get('po_id') == po_id]

        # 计算执行状态
        total_quantity = sum(float(i.get('quantity', 0) or 0) for i in order_items)
        received_quantity = sum(float(h.get('quantity', 0) or 0) for h in receipts)
        
        # 发货状态
        shipment_status = "未发货"
        if any(i.get('inbound_state') == '已发货' for i in order_items):
            shipment_status = "已发货"
        if all(i.get('delivery_complete') == 'Y' for i in order_items):
            shipment_status = "已完成"

        # 收货状态
        receipt_status = "未收货"
        if received_quantity > 0 and received_quantity < total_quantity:
            receipt_status = "部分收货"
        elif received_quantity >= total_quantity:
            receipt_status = "已完成"

        return ConnectorResponse(
            success=True,
            data={
                "approval_status": order.get('flow_status', '未知'),
                "shipment_status": shipment_status,
                "receipt_status": receipt_status,
                "invoice_status": "未开票"
            },
            message=f"查询到采购订单 {po_id} 的执行情况"
        )

    # ==================== 统一调用入口 ====================
    # 本体定义的 action ID 映射

    def call(self, action_id: str, params: Dict[str, Any]) -> ConnectorResponse:
        """统一调用入口 - 基于本体的 action 操作"""
        method_map = {
            # 数据集查询操作 (query)
            'purchase_requests/list': self.purchase_request_list,
            'purchase_requests/get_by_id': self.purchase_request_get_by_id,
            'purchase_inquiries/list': self.purchase_inquiry_list,
            'purchase_inquiries/get_by_id': self.purchase_inquiry_get_by_id,
            'purchase_quotations/list': self.purchase_quotation_list,
            'purchase_quotations/get_by_id': self.purchase_quotation_get_by_id,
            'purchase_order_heads/list': self.purchase_order_head_list,
            'purchase_order_heads/get_by_id': self.purchase_order_head_get_by_id,
            'purchase_order_items/list': self.purchase_order_item_list,
            'purchase_order_receipt_history/list': self.purchase_order_receipt_history_list,
            # 分析操作 (analytics)
            'analytics/find_unexecuted_purchase_requests': self.find_unexecuted_purchase_requests,
            'analytics/generate_price_comparison': self.generate_price_comparison,
            'analytics/get_purchase_order_execution_status': self.get_purchase_order_execution_status,
            # 采购命令操作 (command)
            'procurement/create_inquiry_from_pr': self.create_inquiry_from_pr,
            'procurement/submit_award_approval': self.submit_award_approval,
            'procurement/create_purchase_order_from_pr_and_quotation': self.create_purchase_order_from_pr_and_quotation,
            # RFQ操作
            'rfq/collect_quotations': self.collect_quotations,
            # 带条件的采购需求查询
            'analytics/find_purchase_requests_with_conditions': self.find_purchase_requests_with_conditions,
            # 兼容旧版 action_id
            'purchase_request/list': self.purchase_request_list,
            'purchase_request/get_by_id': self.purchase_request_get_by_id,
            'purchase_inquiry/list': self.purchase_inquiry_list,
            'purchase_inquiry/get_by_id': self.purchase_inquiry_get_by_id,
            'purchase_quotation/list': self.purchase_quotation_list,
            'purchase_quotation/get_by_id': self.purchase_quotation_get_by_id,
            'purchase_quotation/compare': self.purchase_quotation_compare,
            'purchase_order_head/list': self.purchase_order_head_list,
            'purchase_order_head/get_by_id': self.purchase_order_head_get_by_id,
            'purchase_order_item/list': self.purchase_order_item_list,
            'purchase_order_history/list': self.purchase_order_receipt_history_list,
            'purchase_order/query_execution_status': self.get_purchase_order_execution_status,
            'purchase_request/approve': self.purchase_request_approve,
            'purchase_inquiry/create_from_pr': self.create_inquiry_from_pr,
            'purchase_quotation/approve_award': self.purchase_quotation_approve_award,
            'purchase_order/create_from_pr_and_quotation': self.create_purchase_order_from_pr_and_quotation,
        }

        method = method_map.get(action_id)
        if not method:
            return ConnectorResponse(
                success=False,
                data=None,
                error=f"未知操作: {action_id}"
            )

        try:
            return method(params)
        except Exception as e:
            return ConnectorResponse(
                success=False,
                data=None,
                error=f"执行失败: {str(e)}"
            )

    def get_available_actions(self) -> List[Dict[str, str]]:
        """获取所有可用操作 - 基于本体定义"""
        return [
            # 数据集查询操作
            {"id": "purchase_requests/list", "name": "分页查询采购需求", "kind": "query"},
            {"id": "purchase_requests/get_by_id", "name": "按主键查询采购需求", "kind": "query"},
            {"id": "purchase_inquiries/list", "name": "分页查询询价单", "kind": "query"},
            {"id": "purchase_inquiries/get_by_id", "name": "按主键查询询价单", "kind": "query"},
            {"id": "purchase_quotations/list", "name": "分页查询报价单", "kind": "query"},
            {"id": "purchase_quotations/get_by_id", "name": "按主键查询报价单", "kind": "query"},
            {"id": "purchase_order_heads/list", "name": "分页查询采购订单抬头", "kind": "query"},
            {"id": "purchase_order_heads/get_by_id", "name": "按主键查询采购订单抬头", "kind": "query"},
            {"id": "purchase_order_items/list", "name": "分页查询采购订单行项目", "kind": "query"},
            {"id": "purchase_order_receipt_history/list", "name": "分页查询采购订单收货历史", "kind": "query"},
            # 分析操作
            {"id": "analytics/find_unexecuted_purchase_requests", "name": "查询未执行采购需求", "kind": "query"},
            {"id": "analytics/find_purchase_requests_with_conditions", "name": "带条件的采购需求查询", "kind": "query"},
            {"id": "analytics/generate_price_comparison", "name": "生成比价分析", "kind": "query"},
            {"id": "analytics/get_purchase_order_execution_status", "name": "查询采购订单执行情况", "kind": "query"},
            # 业务命令操作
            {"id": "procurement/create_inquiry_from_pr", "name": "根据采购需求创建询价单", "kind": "command"},
            {"id": "procurement/submit_award_approval", "name": "提交中标报价审批", "kind": "command"},
            {"id": "procurement/create_purchase_order_from_pr_and_quotation", "name": "基于采购需求和报价创建采购订单", "kind": "command"},
            # 兼容旧版操作
            {"id": "rfq/collect_quotations", "name": "回收报价", "kind": "query"},
            {"id": "purchase_request/approve", "name": "审批采购需求", "kind": "command"},
            {"id": "purchase_quotation/approve_award", "name": "审批中标报价", "kind": "command"},
        ]


# 全局连接器实例
_connector_instance: Optional[ProcurementConnector] = None


def get_procurement_connector() -> ProcurementConnector:
    """获取采购连接器单例"""
    global _connector_instance
    if _connector_instance is None:
        _connector_instance = ProcurementConnector()
    return _connector_instance
