"""Step 4: Execution for the 5-step pipeline."""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from .models import ExecutionResult, ExecutionRecord


class Step4ConnectorExecutor:
    """Step 4: Execution.

    Executes the planned actions by calling the ProcurementConnector.
    """

    def __init__(self):
        self.connector = None  # Lazy import

    def _get_connector(self):
        if self.connector is None:
            from ..procurement.connector import get_procurement_connector
            self.connector = get_procurement_connector()
        return self.connector

    async def execute(
        self,
        plan_result,
        task_id: str,
        extracted_params: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute the planned actions.

        Args:
            plan_result: Result from Step 3
            task_id: The task ID for tracking
            extracted_params: Slot values extracted from user input

        Returns:
            ExecutionResult with all execution records
        """
        connector = self._get_connector()
        executions: List[ExecutionRecord] = []
        params = extracted_params or {}

        # Add query conditions to params
        for cond in plan_result.query_conditions:
            params[cond.field] = cond.value

        for action in plan_result.planned_actions:
            record = await self._execute_action(
                connector, action.action_id, params, action.connector
            )
            executions.append(record)

        return ExecutionResult(executions=executions, task_id=task_id)

    async def _execute_action(
        self,
        connector,
        action_id: str,
        params: Dict[str, Any],
        connector_name: str
    ) -> ExecutionRecord:
        """Execute a single action."""
        start_time = datetime.now()
        start_iso = start_time.isoformat()

        try:
            result = connector.call(action_id, params)

            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            # Extract result count
            result_count = None
            response_summary = ""
            data = None

            if result.success:
                data = result.data
                if isinstance(data, dict):
                    items = data.get("items", [])
                    result_count = data.get("total", len(items))
                    response_summary = result.message or f"返回 {result_count} 条记录"
                elif isinstance(data, list):
                    result_count = len(data)
                    response_summary = f"返回 {result_count} 条记录"

            return ExecutionRecord(
                connector_name=connector_name,
                connector_id=action_id,
                action_id=action_id,
                status="success",
                start_time=start_iso,
                end_time=end_time.isoformat(),
                latency_ms=latency_ms,
                request_params=params,
                response_summary=response_summary,
                result_count=result_count,
                data=data,
            )

        except Exception as e:
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            return ExecutionRecord(
                connector_name=connector_name,
                connector_id=action_id,
                action_id=action_id,
                status="error",
                start_time=start_iso,
                end_time=end_time.isoformat(),
                latency_ms=latency_ms,
                request_params=params,
                error_message=str(e),
                error_code="EXECUTION_ERROR",
            )
