"""工作流系统连接器 - Workflow Connector

实现审批工作流的系统连接，包括：
- 查询审批工作流状态
- 查询审批历史
- 提交审批申请
- 审批通过/驳回
- 撤回审批申请

注意：当前为Mock实现，需替换为真实工作流系统对接
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class WorkflowResponse:
    """工作流操作响应"""
    success: bool
    data: Any
    message: str = ""
    error: Optional[str] = None


class WorkflowConnector:
    """工作流系统连接器 - 基于本体的审批工作流数据访问层"""
    
    def __init__(self, mock_data_dir: Optional[str] = None):
        if mock_data_dir:
            self.data_dir = Path(mock_data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "workflow" / "mock"
        
        self._workflows: Optional[List[Dict]] = None
        self._history: Optional[List[Dict]] = None
    
    def _load_json(self, filename: str) -> List[Dict]:
        """加载JSON文件"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        return value
            return []
    
    def _save_json(self, filename: str, data: List[Dict]) -> None:
        """保存JSON文件"""
        filepath = self.data_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        key = filename.replace('.json', '')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({key: data}, f, ensure_ascii=False, indent=2)
    
    def _get_workflows(self, include_closed: bool = False) -> List[Dict]:
        """获取工作流列表"""
        if self._workflows is None:
            self._workflows = self._load_json('workflows.json')
        if not include_closed:
            return [w for w in self._workflows if w.get('status') not in ['cancelled', 'expired']]
        return self._workflows
    
    def _get_history(self) -> List[Dict]:
        """获取审批历史"""
        if self._history is None:
            self._history = self._load_json('approval_history.json')
        return self._history
    
    # ==================== 查询操作 ====================
    
    def query_workflow_status(self, params: Dict[str, Any]) -> WorkflowResponse:
        """查询审批工作流状态
        
        Args:
            params: 查询参数
                - workflow_id: 工作流ID
                - related_document_type: 关联单据类型
                - related_document_id: 关联单据ID
                
        Returns:
            WorkflowResponse with workflow status
        """
        workflow_id = params.get('workflow_id')
        related_document_type = params.get('related_document_type')
        related_document_id = params.get('related_document_id')
        
        workflows = self._get_workflows()
        
        # 按workflow_id查询
        if workflow_id:
            for w in workflows:
                if w.get('workflow_id') == workflow_id:
                    return WorkflowResponse(
                        success=True,
                        data=w,
                        message=f"查询到工作流 {workflow_id}"
                    )
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"未找到工作流 {workflow_id}"
            )
        
        # 按关联单据查询
        if related_document_type and related_document_id:
            matched = [
                w for w in workflows
                if w.get('related_document_type') == related_document_type
                and w.get('related_document_id') == related_document_id
            ]
            if matched:
                return WorkflowResponse(
                    success=True,
                    data=matched,
                    message=f"查询到 {len(matched)} 条工作流"
                )
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"未找到关联单据 {related_document_type}/{related_document_id} 的工作流"
            )
        
        # 返回所有待审批
        pending = [w for w in workflows if w.get('status') == 'pending']
        return WorkflowResponse(
            success=True,
            data=pending,
            message=f"查询到 {len(pending)} 条待审批工作流"
        )
    
    def query_workflow_history(self, params: Dict[str, Any]) -> WorkflowResponse:
        """查询审批历史
        
        Args:
            params: 查询参数
                - workflow_id: 工作流ID
                
        Returns:
            WorkflowResponse with approval history
        """
        workflow_id = params.get('workflow_id')
        
        if not workflow_id:
            return WorkflowResponse(
                success=False,
                data=None,
                error="workflow_id is required"
            )
        
        history = self._get_history()
        matched = [h for h in history if h.get('workflow_id') == workflow_id]
        
        # 按步骤排序
        matched.sort(key=lambda x: x.get('step', 0))
        
        return WorkflowResponse(
            success=True,
            data={
                "workflow_id": workflow_id,
                "history_records": matched,
                "total_steps": len(matched),
            },
            message=f"查询到 {len(matched)} 条审批历史"
        )
    
    # ==================== 命令操作 ====================
    
    def submit_for_approval(self, params: Dict[str, Any]) -> WorkflowResponse:
        """提交审批申请
        
        Args:
            params: 提交参数
                - related_document_type: 关联单据类型 (purchase_quotation/purchase_order等)
                - related_document_id: 关联单据ID
                - workflow_type: 工作流类型 (quotation_approval/po_approval等)
                - reason: 提交理由
                
        Returns:
            WorkflowResponse with new workflow info
        """
        related_document_type = params.get('related_document_type')
        related_document_id = params.get('related_document_id')
        workflow_type = params.get('workflow_type', 'general_approval')
        reason = params.get('reason', '')
        
        if not related_document_type or not related_document_id:
            return WorkflowResponse(
                success=False,
                data=None,
                error="related_document_type and related_document_id are required"
            )
        
        # 检查是否已有进行中的工作流
        workflows = self._get_workflows()
        existing = [
            w for w in workflows
            if w.get('related_document_id') == related_document_id
            and w.get('status') == 'pending'
        ]
        if existing:
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"该单据已有进行中的审批流程: {existing[0]['workflow_id']}"
            )
        
        # 生成工作流ID
        today = datetime.now().strftime('%Y%m%d')
        workflow_id = f"WF-{today}-{len(workflows) + 1:04d}"
        
        # 创建工作流实例
        new_workflow = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "related_document_type": related_document_type,
            "related_document_id": related_document_id,
            "status": "pending",
            "current_step": 1,
            "total_steps": 1,
            "approver_id": "approver001",
            "approver_name": "审批管理员",
            "create_time": datetime.now().isoformat(),
            "update_time": datetime.now().isoformat(),
            "remark": reason,
        }
        
        workflows.append(new_workflow)
        self._save_json('workflows.json', workflows)
        self._workflows = workflows
        
        return WorkflowResponse(
            success=True,
            data={
                "workflow_id": workflow_id,
                "status": "pending",
                "current_step": 1,
                "total_steps": 1,
            },
            message=f"审批申请已提交，工作流ID: {workflow_id}"
        )
    
    def approve_workflow(self, params: Dict[str, Any]) -> WorkflowResponse:
        """审批通过
        
        Args:
            params: 审批参数
                - workflow_id: 工作流ID
                - comment: 审批意见
                
        Returns:
            WorkflowResponse with approval result
        """
        workflow_id = params.get('workflow_id')
        comment = params.get('comment', '')
        
        if not workflow_id:
            return WorkflowResponse(
                success=False,
                data=None,
                error="workflow_id is required"
            )
        
        workflows = self._get_workflows()
        workflow = None
        for w in workflows:
            if w.get('workflow_id') == workflow_id:
                workflow = w
                break
        
        if not workflow:
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"未找到工作流 {workflow_id}"
            )
        
        if workflow.get('status') != 'pending':
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"工作流状态不是待审批，当前状态: {workflow.get('status')}"
            )
        
        # 更新工作流状态
        workflow['status'] = 'approved'
        workflow['update_time'] = datetime.now().isoformat()
        
        # 添加审批历史
        history = self._get_history()
        new_history = {
            "history_id": f"HIST-{workflow_id}-{workflow.get('current_step', 1)}",
            "workflow_id": workflow_id,
            "step": workflow.get('current_step', 1),
            "action": "approve",
            "approver_id": "approver001",
            "approver_name": "审批管理员",
            "comment": comment,
            "action_time": datetime.now().isoformat(),
        }
        history.append(new_history)
        
        self._save_json('workflows.json', workflows)
        self._save_json('approval_history.json', history)
        self._workflows = workflows
        self._history = history
        
        return WorkflowResponse(
            success=True,
            data={
                "workflow_id": workflow_id,
                "status": "approved",
                "next_step": None,
            },
            message=f"审批已通过，工作流 {workflow_id} 完成"
        )
    
    def reject_workflow(self, params: Dict[str, Any]) -> WorkflowResponse:
        """审批驳回
        
        Args:
            params: 审批参数
                - workflow_id: 工作流ID
                - reason: 驳回原因
                
        Returns:
            WorkflowResponse with rejection result
        """
        workflow_id = params.get('workflow_id')
        reason = params.get('reason', '')
        
        if not workflow_id:
            return WorkflowResponse(
                success=False,
                data=None,
                error="workflow_id is required"
            )
        
        workflows = self._get_workflows()
        workflow = None
        for w in workflows:
            if w.get('workflow_id') == workflow_id:
                workflow = w
                break
        
        if not workflow:
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"未找到工作流 {workflow_id}"
            )
        
        if workflow.get('status') != 'pending':
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"工作流状态不是待审批，当前状态: {workflow.get('status')}"
            )
        
        # 更新工作流状态
        workflow['status'] = 'rejected'
        workflow['update_time'] = datetime.now().isoformat()
        
        # 添加审批历史
        history = self._get_history()
        new_history = {
            "history_id": f"HIST-{workflow_id}-{workflow.get('current_step', 1)}",
            "workflow_id": workflow_id,
            "step": workflow.get('current_step', 1),
            "action": "reject",
            "approver_id": "approver001",
            "approver_name": "审批管理员",
            "comment": reason,
            "action_time": datetime.now().isoformat(),
        }
        history.append(new_history)
        
        self._save_json('workflows.json', workflows)
        self._save_json('approval_history.json', history)
        self._workflows = workflows
        self._history = history
        
        return WorkflowResponse(
            success=True,
            data={
                "workflow_id": workflow_id,
                "status": "rejected",
            },
            message=f"审批已驳回，工作流 {workflow_id} 结束"
        )
    
    def withdraw_workflow(self, params: Dict[str, Any]) -> WorkflowResponse:
        """撤回审批申请
        
        Args:
            params: 撤回参数
                - workflow_id: 工作流ID
                
        Returns:
            WorkflowResponse with withdrawal result
        """
        workflow_id = params.get('workflow_id')
        
        if not workflow_id:
            return WorkflowResponse(
                success=False,
                data=None,
                error="workflow_id is required"
            )
        
        workflows = self._get_workflows()
        workflow = None
        for w in workflows:
            if w.get('workflow_id') == workflow_id:
                workflow = w
                break
        
        if not workflow:
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"未找到工作流 {workflow_id}"
            )
        
        if workflow.get('status') != 'pending':
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"只能撤回待审批的工作流，当前状态: {workflow.get('status')}"
            )
        
        # 更新工作流状态
        workflow['status'] = 'withdrawn'
        workflow['update_time'] = datetime.now().isoformat()
        
        self._save_json('workflows.json', workflows)
        self._workflows = workflows
        
        return WorkflowResponse(
            success=True,
            data={
                "workflow_id": workflow_id,
                "status": "withdrawn",
            },
            message=f"审批申请已撤回，工作流 {workflow_id} 结束"
        )
    
    # ==================== 统一调用入口 ====================
    
    def call(self, action_id: str, params: Dict[str, Any]) -> WorkflowResponse:
        """统一调用入口 - 基于本体的action操作"""
        method_map = {
            'approval/query_workflow_status': self.query_workflow_status,
            'approval/query_workflow_history': self.query_workflow_history,
            'approval/submit_for_approval': self.submit_for_approval,
            'approval/approve_workflow': self.approve_workflow,
            'approval/reject_workflow': self.reject_workflow,
            'approval/withdraw_workflow': self.withdraw_workflow,
        }
        
        method = method_map.get(action_id)
        if not method:
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"未知操作: {action_id}"
            )
        
        try:
            return method(params)
        except Exception as e:
            return WorkflowResponse(
                success=False,
                data=None,
                error=f"执行失败: {str(e)}"
            )
    
    def get_available_actions(self) -> List[Dict[str, str]]:
        """获取所有可用操作 - 基于本体定义"""
        return [
            {"id": "approval/query_workflow_status", "name": "查询审批工作流状态", "kind": "query"},
            {"id": "approval/query_workflow_history", "name": "查询审批历史", "kind": "query"},
            {"id": "approval/submit_for_approval", "name": "提交审批申请", "kind": "command"},
            {"id": "approval/approve_workflow", "name": "审批通过", "kind": "command"},
            {"id": "approval/reject_workflow", "name": "审批驳回", "kind": "command"},
            {"id": "approval/withdraw_workflow", "name": "撤回审批申请", "kind": "command"},
        ]


# 全局连接器实例
_workflow_connector: Optional[WorkflowConnector] = None


def get_workflow_connector() -> WorkflowConnector:
    """获取工作流连接器单例"""
    global _workflow_connector
    if _workflow_connector is None:
        _workflow_connector = WorkflowConnector()
    return _workflow_connector
