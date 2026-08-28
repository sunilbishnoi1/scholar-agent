"""
Cancellation Manager for Scholar Agent Research Workflows.

Provides thread-safe and distributed cancellation tracking, Celery task revocation,
and state polling for long-running multi-agent pipelines.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class TaskCancelledException(Exception):
    """Raised when an in-flight research project task is cancelled/stopped by the user."""

    def __init__(self, project_id: str, message: str = "Research task was stopped by user.") -> None:
        super().__init__(f"Project '{project_id}': {message}")
        self.project_id = project_id
        self.message = message


class CancellationManager:
    """
    Centralized manager tracking active task cancellations across background threads
    and distributed Celery workers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled_projects: set[str] = set()
        self._active_tasks: dict[str, str] = {}  # project_id -> celery_task_id
        self._redis_client = None
        self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection if available for distributed cancellation."""
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis

            self._redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis_client.ping()
            logger.info("CancellationManager initialized with active Redis backend.")
        except Exception as e:
            logger.info(f"CancellationManager using in-memory tracking (Redis unavailable: {e}).")
            self._redis_client = None

    def cancel_project(self, project_id: str) -> None:
        """Mark a project as cancelled."""
        with self._lock:
            self._cancelled_projects.add(project_id)

        if self._redis_client:
            try:
                # Set with 2-hour expiration
                self._redis_client.setex(f"scholar:cancelled:{project_id}", 7200, "1")
            except Exception as e:
                logger.warning(f"Failed to record cancellation in Redis for project {project_id}: {e}")

        logger.info(f"[CancellationManager] Project '{project_id}' marked as CANCELLED.")

    def clear_cancellation(self, project_id: str) -> None:
        """Clear the cancellation flag when restarting or starting a new review for a project."""
        with self._lock:
            self._cancelled_projects.discard(project_id)
            self._active_tasks.pop(project_id, None)

        if self._redis_client:
            try:
                self._redis_client.delete(f"scholar:cancelled:{project_id}")
            except Exception as e:
                logger.warning(f"Failed to clear Redis cancellation for project {project_id}: {e}")

        logger.info(f"[CancellationManager] Project '{project_id}' cancellation cleared.")

    def is_cancelled(self, project_id: str) -> bool:
        """Check if a project execution has been cancelled."""
        if not project_id:
            return False

        with self._lock:
            if project_id in self._cancelled_projects:
                return True

        if self._redis_client:
            try:
                val = self._redis_client.get(f"scholar:cancelled:{project_id}")
                if val:
                    # Also update local set
                    with self._lock:
                        self._cancelled_projects.add(project_id)
                    return True
            except Exception as e:
                logger.debug(f"Redis cancellation check error for {project_id}: {e}")

        return False

    def check_and_raise_if_cancelled(self, project_id: str) -> None:
        """Convenience helper to raise TaskCancelledException if cancelled."""
        if self.is_cancelled(project_id):
            raise TaskCancelledException(project_id)

    def register_task(self, project_id: str, task_id: str) -> None:
        """Register a Celery task ID associated with a project."""
        with self._lock:
            self._active_tasks[project_id] = task_id
        if self._redis_client:
            try:
                self._redis_client.setex(f"scholar:task:{project_id}", 7200, task_id)
            except Exception as e:
                logger.debug(f"Failed to cache task_id in Redis: {e}")

    def unregister_task(self, project_id: str) -> None:
        """Unregister a Celery task ID upon completion."""
        with self._lock:
            self._active_tasks.pop(project_id, None)
        if self._redis_client:
            try:
                self._redis_client.delete(f"scholar:task:{project_id}")
            except Exception as e:
                logger.debug(f"Failed to delete task_id in Redis: {e}")

    def get_task_id(self, project_id: str) -> Optional[str]:
        """Get the active task ID associated with a project if registered."""
        with self._lock:
            task_id = self._active_tasks.get(project_id)
        if not task_id and self._redis_client:
            try:
                task_id = self._redis_client.get(f"scholar:task:{project_id}")
            except Exception:
                pass
        return task_id

    def revoke_task(self, project_id: str, celery_app: Optional[object] = None) -> bool:
        """Revoke and terminate any in-flight Celery task for the project."""
        task_id = None
        with self._lock:
            task_id = self._active_tasks.get(project_id)

        if not task_id and self._redis_client:
            try:
                task_id = self._redis_client.get(f"scholar:task:{project_id}")
            except Exception:
                pass

        if task_id and celery_app:
            try:
                logger.info(f"[CancellationManager] Revoking Celery task '{task_id}' for project '{project_id}'")
                celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
                return True
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task {task_id}: {e}")

        return False


# Global singleton instance
cancellation_manager = CancellationManager()
