"""Typed models for the clinical trial screening environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RewardModel(BaseModel):
    """Structured reward breakdown used by the environment and graders."""

    incremental_reward: float = Field(default=0.0)
    final_reward: float = Field(default=0.0)
    penalty: float = Field(default=0.0)
    total: float = Field(default=0.0)
    reasons: List[str] = Field(default_factory=list)


class ClinicalTrialScreeningAction(Action):
    """Action model for extracting facts, ranking patients, and deciding eligibility."""

    action_type: Literal[
        "extract_data",
        "submit_ranking",
        "flag_exclusions",
        "final_decision",
        "destructive_action",
    ] = Field(..., description="The operation the agent wants to perform.")
    target_id: Optional[str] = Field(
        default=None,
        description="Patient id, task id, or document id for the action.",
    )
    field_name: Optional[str] = Field(
        default=None,
        description="Clinical field being extracted, such as age or biomarker.",
    )
    value: Optional[str] = Field(
        default=None,
        description="Normalized extracted value or final decision label.",
    )
    ranking: List[str] = Field(
        default_factory=list,
        description="Ordered patient ids for ranking tasks.",
    )
    exclusions: List[str] = Field(
        default_factory=list,
        description="Structured exclusion or deviation codes identified in text.",
    )
    rationale: str = Field(
        default="",
        description="Free-text reasoning preserved for audits.",
    )


class ClinicalTrialScreeningObservation(Observation):
    """Observation emitted after every environment transition."""

    task_id: str = Field(..., description="Stable identifier for the current task.")
    difficulty: TaskDifficulty = Field(..., description="Current task difficulty.")
    title: str = Field(..., description="Human-readable task title.")
    brief: str = Field(..., description="Task summary shown to the agent.")
    prompt: str = Field(..., description="Full protocol and patient context.")
    extracted_points: Dict[str, str] = Field(
        default_factory=dict,
        description="Clinical facts confirmed so far by the agent.",
    )
    missing_targets: List[str] = Field(
        default_factory=list,
        description="Important data points that can still be extracted for reward.",
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Valid high-level actions for the current state.",
    )
    grader_score: float = Field(
        default=0.0,
        description="Deterministic programmatic grader score for the current task.",
    )
    reward_breakdown: RewardModel = Field(
        default_factory=RewardModel,
        description="Structured reward signal for the current step.",
    )
    feedback: str = Field(default="", description="Environment feedback for the agent.")
    trial_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relevant protocol metadata and expected output schema.",
    )
