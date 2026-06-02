"""Test case management for Agent diagnostics."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import TestCase, TestCaseResult


class TestCaseManager:
    """Manages test cases for batch testing and evaluation."""

    def __init__(self, storage_dir: str = "data/test_cases"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[TestCase]] = {}

    def _get_file_path(self, agent_id: str) -> Path:
        """Get the storage file path for an agent."""
        return self.storage_dir / f"{agent_id}.json"

    def _load_cases(self, agent_id: str) -> List[TestCase]:
        """Load test cases for an agent from storage."""
        if agent_id in self._cache:
            return self._cache[agent_id]

        file_path = self._get_file_path(agent_id)
        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cases = [TestCase.from_dict(c) for c in data.get("cases", [])]
                self._cache[agent_id] = cases
                return cases
        except (json.JSONDecodeError, KeyError):
            return []

    def _save_cases(self, agent_id: str, cases: List[TestCase]) -> None:
        """Save test cases for an agent to storage."""
        file_path = self._get_file_path(agent_id)
        data = {
            "agent_id": agent_id,
            "updated_at": datetime.now().isoformat(),
            "cases": [c.to_dict() for c in cases],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._cache[agent_id] = cases

    def generate_id(self) -> str:
        """Generate a unique test case ID."""
        timestamp = datetime.now().strftime("%Y%m%d")
        short_uuid = str(uuid.uuid4())[:8]
        return f"case_{timestamp}_{short_uuid}"

    def list_cases(
        self,
        agent_id: str,
        scenario_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[TestCase]:
        """List all test cases for an agent, optionally filtered."""
        cases = self._load_cases(agent_id)

        if scenario_id:
            cases = [c for c in cases if c.scenario_id == scenario_id]
        if status:
            cases = [c for c in cases if c.status == status]

        return cases

    def get_case(self, agent_id: str, case_id: str) -> Optional[TestCase]:
        """Get a specific test case by ID."""
        cases = self._load_cases(agent_id)
        for case in cases:
            if case.id == case_id:
                return case
        return None

    def create_case(
        self,
        agent_id: str,
        name: str,
        input_text: str,
        description: str = "",
        scenario_id: str = "",
        user_profile: str = "",
        expected_intent: str = "",
        expected_skill: str = "",
        expected_keywords: List[str] = None,
        created_by: str = "system",
    ) -> TestCase:
        """Create a new test case."""
        case = TestCase(
            id=self.generate_id(),
            agent_id=agent_id,
            name=name,
            description=description,
            input=input_text,
            scenario_id=scenario_id,
            user_profile=user_profile,
            expected_intent=expected_intent,
            expected_skill=expected_skill,
            expected_response_keywords=expected_keywords or [],
            created_by=created_by,
        )

        cases = self._load_cases(agent_id)
        cases.append(case)
        self._save_cases(agent_id, cases)

        return case

    def update_case(self, agent_id: str, case_id: str, updates: Dict[str, Any]) -> Optional[TestCase]:
        """Update an existing test case."""
        cases = self._load_cases(agent_id)
        for i, case in enumerate(cases):
            if case.id == case_id:
                for key, value in updates.items():
                    if hasattr(case, key):
                        setattr(case, key, value)
                case.updated_at = datetime.now()
                self._save_cases(agent_id, cases)
                return case
        return None

    def delete_case(self, agent_id: str, case_id: str) -> bool:
        """Delete a test case."""
        cases = self._load_cases(agent_id)
        original_len = len(cases)
        cases = [c for c in cases if c.id != case_id]

        if len(cases) < original_len:
            self._save_cases(agent_id, cases)
            return True
        return False

    def archive_case(self, agent_id: str, case_id: str) -> bool:
        """Archive a test case (set status to archived)."""
        return self.update_case(agent_id, case_id, {"status": "archived"}) is not None

    def save_run_result(self, result: TestCaseResult) -> None:
        """Save a test case execution result."""
        results_dir = self.storage_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        file_path = results_dir / f"{result.test_case_id}_results.json"
        try:
            existing = []
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(result.to_dict())
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([result.to_dict()], f, indent=2, ensure_ascii=False)

    def get_case_results(self, case_id: str) -> List[TestCaseResult]:
        """Get all execution results for a test case."""
        results_dir = self.storage_dir / "results"
        file_path = results_dir / f"{case_id}_results.json"

        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [TestCaseResult(**r) for r in data]
        except (json.JSONDecodeError, TypeError):
            return []

    def create_from_test_run(
        self,
        agent_id: str,
        test_run_id: str,
        input_text: str,
        expected_intent: str = "",
        name: str = "",
    ) -> TestCase:
        """Create a test case from a test run result."""
        if not name:
            name = f"用例_{datetime.now().strftime('%m%d_%H%M')}"
        return self.create_case(
            agent_id=agent_id,
            name=name,
            input_text=input_text,
            expected_intent=expected_intent,
            description=f"从测试运行 {test_run_id} 创建",
        )
