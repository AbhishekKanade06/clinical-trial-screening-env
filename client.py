"""OpenEnv client for the clinical trial screening environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import ClinicalTrialScreeningAction, ClinicalTrialScreeningObservation


class ClinicalTrialScreeningEnvClient(
    EnvClient[ClinicalTrialScreeningAction, ClinicalTrialScreeningObservation, State]
):
    """Typed OpenEnv client for the clinical trial screening benchmark."""

    def _step_payload(self, action: ClinicalTrialScreeningAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self,
        payload: Dict[str, Any],
    ) -> StepResult[ClinicalTrialScreeningObservation]:
        obs_data = payload.get("observation", {})
        observation = ClinicalTrialScreeningObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            title=obs_data.get("title", ""),
            brief=obs_data.get("brief", ""),
            prompt=obs_data.get("prompt", ""),
            extracted_points=obs_data.get("extracted_points", {}),
            missing_targets=obs_data.get("missing_targets", []),
            available_actions=obs_data.get("available_actions", []),
            grader_score=obs_data.get("grader_score", 0.0),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            feedback=obs_data.get("feedback", ""),
            trial_metadata=obs_data.get("trial_metadata", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
