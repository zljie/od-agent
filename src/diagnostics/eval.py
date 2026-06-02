"""Agent Eval - batch testing and evaluation capabilities."""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .models import (
    TestCase,
    TestCaseResult,
    TestRun,
    TestRunStatus,
    ResourceStatus,
    StepStatus,
)
from .test_cases import TestCaseManager
from .collector import DiagnosticsCollector, generate_run_id


class BatchEvalResult:
    """Result of a batch evaluation run."""

    def __init__(
        self,
        batch_id: str,
        total: int,
        passed: int,
        failed: int,
        results: List[Dict[str, Any]],
        duration_ms: int,
        run_mode: str = "sequential",
    ):
        self.batch_id = batch_id
        self.total = total
        self.passed = passed
        self.failed = failed
        self.results = results
        self.duration_ms = duration_ms
        self.run_mode = run_mode
        self.created_at = datetime.now()

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": f"{self.pass_rate:.1%}",
            "results": self.results,
            "duration_ms": self.duration_ms,
            "run_mode": self.run_mode,
            "created_at": self.created_at.isoformat(),
        }


class AgentEval:
    """Handles batch testing and evaluation of Agent configurations."""

    def __init__(
        self,
        test_case_manager: Optional[TestCaseManager] = None,
        agent_executor: Optional[Callable] = None,
    ):
        self.test_case_manager = test_case_manager or TestCaseManager()
        self.agent_executor = agent_executor
        self._storage_dir = Path("data/eval_results")
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._batch_cache: Dict[str, BatchEvalResult] = {}

    def generate_batch_id(self) -> str:
        """Generate a unique batch ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:6]
        return f"batch_{timestamp}_{short_uuid}"

    async def run_batch(
        self,
        agent_id: str,
        case_ids: List[str],
        run_mode: str = "sequential",
        agent_executor: Optional[Callable] = None,
    ) -> BatchEvalResult:
        """Run a batch of test cases.

        Args:
            agent_id: The agent ID to test
            case_ids: List of test case IDs to run
            run_mode: "sequential" or "parallel"
            agent_executor: Function to execute agent chat (async function)

        Returns:
            BatchEvalResult with all test results
        """
        batch_id = self.generate_batch_id()
        start_time = time.time()

        cases = []
        for case_id in case_ids:
            case = self.test_case_manager.get_case(agent_id, case_id)
            if case:
                cases.append(case)

        if not cases:
            return BatchEvalResult(
                batch_id=batch_id,
                total=0,
                passed=0,
                failed=0,
                results=[],
                duration_ms=int((time.time() - start_time) * 1000),
                run_mode=run_mode,
            )

        results: List[Dict[str, Any]] = []
        passed = 0
        failed = 0

        if run_mode == "parallel" and agent_executor:
            tasks = []
            for case in cases:
                tasks.append(self._run_single_case(agent_id, case, agent_executor))
            case_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(case_results):
                if isinstance(result, Exception):
                    results.append({
                        "case_id": cases[i].id,
                        "passed": False,
                        "run_id": None,
                        "error": str(result),
                    })
                    failed += 1
                else:
                    results.append(result)
                    if result.get("passed"):
                        passed += 1
                    else:
                        failed += 1
        else:
            for case in cases:
                result = await self._run_single_case(
                    agent_id, case, agent_executor or self.agent_executor
                )
                results.append(result)
                if result.get("passed"):
                    passed += 1
                else:
                    failed += 1

        duration_ms = int((time.time() - start_time) * 1000)
        batch_result = BatchEvalResult(
            batch_id=batch_id,
            total=len(cases),
            passed=passed,
            failed=failed,
            results=results,
            duration_ms=duration_ms,
            run_mode=run_mode,
        )

        self._batch_cache[batch_id] = batch_result
        self._save_batch_result(batch_result)

        return batch_result

    async def _run_single_case(
        self,
        agent_id: str,
        case: TestCase,
        executor: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run a single test case."""
        run_id = generate_run_id()

        try:
            response = ""
            if executor:
                response = await executor(case.input)
            else:
                from ..agent import get_agent
                agent = get_agent()
                response = await agent.chat(case.input)

            passed = self._evaluate_case(case, response)

            result = TestCaseResult(
                test_case_id=case.id,
                test_run_id=run_id,
                passed=passed,
                intent_match=True,
                skill_match=case.expected_skill in response if case.expected_skill else True,
                response_match=any(
                    kw in response for kw in case.expected_response_keywords
                ) if case.expected_response_keywords else True,
                details={
                    "input": case.input,
                    "response": response[:500],
                    "expected_intent": case.expected_intent,
                },
            )
            self.test_case_manager.save_run_result(result)

            return {
                "case_id": case.id,
                "case_name": case.name,
                "passed": passed,
                "run_id": run_id,
                "response_preview": response[:200] if response else "",
            }

        except Exception as e:
            return {
                "case_id": case.id,
                "case_name": case.name,
                "passed": False,
                "run_id": run_id,
                "error": str(e),
            }

    def _evaluate_case(self, case: TestCase, response: str) -> bool:
        """Evaluate if a test case passed."""
        if case.expected_response_keywords:
            return any(kw in response for kw in case.expected_response_keywords)
        return len(response) > 0

    def _save_batch_result(self, result: BatchEvalResult) -> None:
        """Save batch result to disk."""
        file_path = self._storage_dir / f"{result.batch_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    def get_batch_result(self, batch_id: str) -> Optional[BatchEvalResult]:
        """Get a batch result by ID."""
        if batch_id in self._batch_cache:
            return self._batch_cache[batch_id]

        file_path = self._storage_dir / f"{batch_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    result = BatchEvalResult(
                        batch_id=data["batch_id"],
                        total=data["total"],
                        passed=data["passed"],
                        failed=data["failed"],
                        results=data["results"],
                        duration_ms=data["duration_ms"],
                        run_mode=data.get("run_mode", "sequential"),
                    )
                    self._batch_cache[batch_id] = result
                    return result
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def compare_versions(
        self,
        baseline_results: List[Dict[str, Any]],
        new_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare two sets of test results (version comparison).

        Args:
            baseline_results: Results from baseline version
            new_results: Results from new version

        Returns:
            Comparison report with new failures and fixes
        """
        baseline_map = {r["case_id"]: r for r in baseline_results}
        new_map = {r["case_id"]: r for r in new_results}

        all_case_ids = set(baseline_map.keys()) | set(new_map.keys())

        newly_failed = []
        newly_passed = []
        unchanged_failed = []
        unchanged_passed = []

        for case_id in all_case_ids:
            baseline_passed = baseline_map.get(case_id, {}).get("passed", False)
            new_passed = new_map.get(case_id, {}).get("passed", False)

            if baseline_passed and not new_passed:
                newly_failed.append({
                    "case_id": case_id,
                    "case_name": new_map.get(case_id, {}).get("case_name", ""),
                })
            elif not baseline_passed and new_passed:
                newly_passed.append({
                    "case_id": case_id,
                    "case_name": new_map.get(case_id, {}).get("case_name", ""),
                })
            elif not new_passed:
                unchanged_failed.append(case_id)
            else:
                unchanged_passed.append(case_id)

        baseline_pass_rate = sum(1 for r in baseline_results if r.get("passed")) / max(len(baseline_results), 1)
        new_pass_rate = sum(1 for r in new_results if r.get("passed")) / max(len(new_results), 1)

        return {
            "summary": {
                "baseline_pass_rate": f"{baseline_pass_rate:.1%}",
                "new_pass_rate": f"{new_pass_rate:.1%}",
                "rate_change": f"{(new_pass_rate - baseline_pass_rate):+.1%}",
            },
            "newly_failed": newly_failed,
            "newly_passed": newly_passed,
            "unchanged_failed_count": len(unchanged_failed),
            "unchanged_passed_count": len(unchanged_passed),
        }

    def generate_report(
        self,
        batch_results: List[BatchEvalResult],
        title: str = "Agent Evaluation Report",
    ) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        total_runs = sum(br.total for br in batch_results)
        total_passed = sum(br.passed for br in batch_results)
        total_failed = sum(br.failed for br in batch_results)
        total_duration = sum(br.duration_ms for br in batch_results)

        avg_latency = total_duration / max(total_runs, 1)

        return {
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_runs": total_runs,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "pass_rate": f"{total_passed / max(total_runs, 1):.1%}",
                "avg_latency_ms": f"{avg_latency:.0f}",
            },
            "batches": [
                {
                    "batch_id": br.batch_id,
                    "created_at": br.created_at.isoformat(),
                    "total": br.total,
                    "passed": br.passed,
                    "failed": br.failed,
                    "pass_rate": f"{br.pass_rate:.1%}",
                    "duration_ms": br.duration_ms,
                }
                for br in batch_results
            ],
        }
