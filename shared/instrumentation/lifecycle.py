from __future__ import annotations

from typing import Any
from uuid import UUID

from shared.domain.identifiers import utc_now

from .models import ActionRun, ActionStatus


class ActionRunManager:
    """
    Manager for the lifecycle of a high-level ActionRun.
    Ensures consistent status transitions and timestamping.
    """

    def __init__(self, action_run: ActionRun):
        self.action_run = action_run

    @classmethod
    def start(
        cls,
        action_name: str,
        decision_id: UUID,
        parent_action_run_id: UUID | None = None,
        meme_id: UUID | None = None,
        source_event_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActionRunManager:
        """Initialize and start a new action run."""
        run = ActionRun(
            action_name=action_name,
            decision_id=decision_id,
            parent_action_run_id=parent_action_run_id,
            meme_id=meme_id,
            source_event_id=source_event_id,
            status=ActionStatus.RUNNING,
            started_at=utc_now(),
            metadata=metadata or {},
        )
        return cls(run)

    def finish(self, status: ActionStatus = ActionStatus.SUCCEEDED) -> ActionRun:
        """Mark the action run as finished."""
        self.action_run.status = status
        self.action_run.finished_at = utc_now()
        return self.action_run

    def fail(self, metadata: dict[str, Any] | None = None) -> ActionRun:
        """Mark the action run as failed."""
        if metadata:
            self.action_run.metadata.update(metadata)
        return self.finish(ActionStatus.FAILED)

    def cancel(self) -> ActionRun:
        """Mark the action run as cancelled."""
        return self.finish(ActionStatus.CANCELLED)

    def update_metadata(self, **kwargs: Any) -> None:
        """Update metadata on the active run."""
        self.action_run.metadata.update(kwargs)
