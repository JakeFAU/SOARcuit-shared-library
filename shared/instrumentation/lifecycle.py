from __future__ import annotations

from typing import Any
from uuid import UUID

from shared.domain.identifiers import utc_now

from .models import ActionRun, ActionStatus


class ActionRunManager:
    """Mutable lifecycle helper around a single action run."""

    def __init__(self, action_run: ActionRun):
        self.action_run = action_run

    @classmethod
    def start(
        cls,
        action_name: str,
        decision_id: UUID,
        *,
        meme_id: UUID | None = None,
        parent_action_run_id: UUID | None = None,
        triggered_by_action_run_id: UUID | None = None,
        source_event_id: UUID | None = None,
        actor_system: str | None = None,
        actor_id: str | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        prompt_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActionRunManager:
        run = ActionRun(
            action_name=action_name,
            decision_id=decision_id,
            meme_id=meme_id,
            parent_action_run_id=parent_action_run_id,
            triggered_by_action_run_id=triggered_by_action_run_id,
            source_event_id=source_event_id,
            actor_system=actor_system,
            actor_id=actor_id,
            provider=provider,
            model_name=model_name,
            model_version=model_version,
            prompt_version=prompt_version,
            status=ActionStatus.RUNNING,
            started_at=utc_now(),
            metadata=metadata or {},
        )
        return cls(run)

    def finish(self, status: ActionStatus = ActionStatus.SUCCEEDED) -> ActionRun:
        self.action_run.status = status
        self.action_run.finished_at = utc_now()
        return self.action_run

    def succeed(self) -> ActionRun:
        return self.finish(ActionStatus.SUCCEEDED)

    def fail(
        self,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActionRun:
        if reason is not None:
            self.action_run.reason = reason
        if metadata:
            self.action_run.metadata.update(metadata)
        return self.finish(ActionStatus.FAILED)

    def cancel(self, *, metadata: dict[str, Any] | None = None) -> ActionRun:
        if metadata:
            self.action_run.metadata.update(metadata)
        return self.finish(ActionStatus.CANCELLED)

    def set_meme_id(self, meme_id: UUID) -> None:
        self.action_run.meme_id = meme_id

    def set_source_event_id(self, source_event_id: UUID) -> None:
        self.action_run.source_event_id = source_event_id

    def update_metadata(self, **kwargs: Any) -> None:
        self.action_run.metadata.update(kwargs)
