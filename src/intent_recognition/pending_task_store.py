"""Pending HITL Task Store.

Phase 4.1: Persistent task store for multi-round HITL slot filling.

This module implements the Pending Task Store that survives across HTTP requests.
It stores HITL tasks that are waiting for slot filling, keyed by task_id.

Protocol:
1. When pipeline generates HITL (missing slots), store the pending task
2. Frontend submits slot-fill via: POST /chat with [slot-fill] protocol
3. Backend loads the pending task by task_id
4. Backend validates and merges the filled slots
5. Backend checks if all slots are complete -> execute or ask for more

Task States:
- waiting_slots: task created, waiting for frontend to collect slots
- slots_partial: some slots filled, more needed
- slots_complete: all required slots filled, ready to execute
- cancelled: user cancelled the task
- expired: task expired (TTL exceeded)
"""

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default TTL for pending tasks: 30 minutes
DEFAULT_TTL_SECONDS = 30 * 60


@dataclass
class PendingTask:
    """A pending HITL task waiting for slot filling."""

    task_id: str
    action: str  # e.g., "create_purchase_requests"
    object: str  # e.g., "purchase_requests"
    missing_slots: List[str]
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    status: str = "waiting_slots"  # waiting_slots | slots_partial | slots_complete | cancelled | expired
    created_at: str = ""  # ISO timestamp
    updated_at: str = ""  # ISO timestamp
    expires_at: str = ""  # ISO timestamp
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.expires_at:
            self.expires_at = (datetime.now() + timedelta(seconds=DEFAULT_TTL_SECONDS)).isoformat()

    def is_expired(self) -> bool:
        """Check if the task has expired."""
        try:
            expires = datetime.fromisoformat(self.expires_at)
            return datetime.now() > expires
        except (ValueError, TypeError):
            return True

    def is_complete(self) -> bool:
        """Check if all required slots are filled."""
        return all(
            self.filled_slots.get(slot) not in (None, "", [])
            for slot in self.missing_slots
        )

    def remaining_slots(self) -> List[str]:
        """Return list of slots that are still missing."""
        return [
            slot for slot in self.missing_slots
            if self.filled_slots.get(slot) in (None, "", [])
        ]

    def fill_slots(self, new_slots: Dict[str, Any]) -> None:
        """Fill slots and update status."""
        self.filled_slots.update(new_slots)
        self.updated_at = datetime.now().isoformat()
        remaining = self.remaining_slots()
        if not remaining:
            self.status = "slots_complete"
        else:
            self.status = "slots_partial"
            self.missing_slots = remaining

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingTask":
        """Create from dictionary."""
        return cls(**data)


class PendingTaskStore:
    """Thread-safe pending HITL task store.

    Provides:
    - save(): Store a new pending task
    - load(): Load a task by task_id
    - update(): Update a task's slots
    - delete(): Remove a task
    - cleanup(): Remove expired tasks
    """

    _instance: Optional["PendingTaskStore"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for task store."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, persist_path: Optional[str] = None):
        """Initialize the task store.

        Parameters
        ----------
        ttl_seconds:
            Time-to-live for tasks in seconds. Default 30 minutes.
        persist_path:
            Optional path for JSON file persistence. If None, only in-memory.
        """
        if self._initialized:
            return
        self._initialized = True
        self._ttl = ttl_seconds
        self._persist_path = persist_path
        self._tasks: Dict[str, PendingTask] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load tasks from disk if persistence is configured."""
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for task_id, task_data in data.items():
                        task = PendingTask.from_dict(task_data)
                        if not task.is_expired():
                            self._tasks[task_id] = task
                    print(f"[PendingTaskStore] Loaded {len(self._tasks)} tasks from disk")
        except Exception as e:
            print(f"[PendingTaskStore] Failed to load from disk: {e}")

    def _save_to_disk(self) -> None:
        """Save tasks to disk if persistence is configured."""
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {task_id: task.to_dict() for task_id, task in self._tasks.items()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[PendingTaskStore] Failed to save to disk: {e}")

    def save(self, task: PendingTask) -> None:
        """Save a new pending task.

        Parameters
        ----------
        task:
            The pending task to store.
        """
        with self._lock:
            self._tasks[task.task_id] = task
            self._save_to_disk()
            print(f"[PendingTaskStore] Saved task {task.task_id} | action={task.action} | 缺槽={task.missing_slots}")

    def load(self, task_id: str) -> Optional[PendingTask]:
        """Load a pending task by task_id.

        Parameters
        ----------
        task_id:
            The task ID to look up.

        Returns
        -------
        PendingTask or None
            The task if found and not expired, None otherwise.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                print(f"[PendingTaskStore] Task {task_id} not found")
                return None
            if task.is_expired():
                print(f"[PendingTaskStore] Task {task_id} has expired, removing")
                del self._tasks[task_id]
                self._save_to_disk()
                return None
            print(f"[PendingTaskStore] Loaded task {task_id} | action={task.action} | status={task.status} | 已填={list(task.filled_slots.keys())} | 缺槽={task.missing_slots}")
            return task

    def update(self, task_id: str, **updates) -> Optional[PendingTask]:
        """Update a pending task.

        Parameters
        ----------
        task_id:
            The task ID to update.
        **updates:
            Fields to update (filled_slots, missing_slots, status, etc.).

        Returns
        -------
        PendingTask or None
            The updated task if found, None otherwise.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = datetime.now().isoformat()
            self._save_to_disk()
            return task

    def delete(self, task_id: str) -> bool:
        """Delete a pending task.

        Parameters
        ----------
        task_id:
            The task ID to delete.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self._save_to_disk()
                print(f"[PendingTaskStore] Deleted task {task_id}")
                return True
            return False

    def cleanup(self) -> int:
        """Remove all expired tasks.

        Returns
        -------
        int
            Number of tasks removed.
        """
        with self._lock:
            expired = [tid for tid, task in self._tasks.items() if task.is_expired()]
            for tid in expired:
                del self._tasks[tid]
            if expired:
                self._save_to_disk()
                print(f"[PendingTaskStore] Cleaned up {len(expired)} expired tasks")
            return len(expired)

    def get_all(self) -> List[PendingTask]:
        """Get all non-expired pending tasks."""
        with self._lock:
            self.cleanup()
            return list(self._tasks.values())

    def clear(self) -> None:
        """Clear all tasks (for testing)."""
        with self._lock:
            self._tasks.clear()
            self._save_to_disk()


# Global instance
_task_store: Optional[PendingTaskStore] = None


def get_task_store(persist_path: str = "data/pending_tasks.json") -> PendingTaskStore:
    """Get the global PendingTaskStore instance.

    Parameters
    ----------
    persist_path:
        Path for JSON file persistence.

    Returns
    -------
    PendingTaskStore
        The global task store instance.
    """
    global _task_store
    if _task_store is None:
        _task_store = PendingTaskStore(persist_path=persist_path)
    return _task_store


def save_pending_task(
    task_id: str,
    action: str,
    object: str,
    missing_slots: List[str],
    filled_slots: Optional[Dict[str, Any]] = None,
    confidence: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> PendingTask:
    """Convenience function to save a pending task.

    Parameters
    ----------
    task_id:
        Unique task identifier (e.g., "slot-fill-task-20260603-06849f").
    action:
        The action being performed (e.g., "create_purchase_requests").
    object:
        The business object (e.g., "purchase_requests").
    missing_slots:
        List of slot names that still need to be filled.
    filled_slots:
        Already-filled slots (optional).
    confidence:
        Confidence score from intent recognition.
    metadata:
        Additional metadata (optional).

    Returns
    -------
    PendingTask
        The saved pending task.
    """
    store = get_task_store()
    task = PendingTask(
        task_id=task_id,
        action=action,
        object=object,
        missing_slots=missing_slots,
        filled_slots=filled_slots or {},
        confidence=confidence,
        metadata=metadata or {},
    )
    store.save(task)
    return task


def load_pending_task(task_id: str) -> Optional[PendingTask]:
    """Convenience function to load a pending task.

    Parameters
    ----------
    task_id:
        The task ID to load.

    Returns
    -------
    PendingTask or None
        The task if found, None otherwise.
    """
    return get_task_store().load(task_id)


def update_pending_task(task_id: str, **updates) -> Optional[PendingTask]:
    """Convenience function to update a pending task."""
    return get_task_store().update(task_id, **updates)


def delete_pending_task(task_id: str) -> bool:
    """Convenience function to delete a pending task."""
    return get_task_store().delete(task_id)
