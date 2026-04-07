"""Core RL logic for clinical trial patient screening."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set
from uuid import uuid4

from openenv.core.env_server.types import State

from models import (
    ClinicalTrialScreeningAction,
    ClinicalTrialScreeningObservation,
    RewardModel,
    TaskDifficulty,
)

INCREMENTAL_REWARD = 0.20
FINAL_REWARD = 1.00
HALLUCINATION_PENALTY = -0.50


@dataclass(frozen=True)
class ScreeningTask:
    task_id: str
    difficulty: TaskDifficulty
    title: str
    brief: str
    prompt: str
    extraction_targets: Dict[str, str]
    expected_decision: str
    trial_metadata: Dict[str, Any]
    ranking_ground_truth: List[str] = field(default_factory=list)
    exclusion_ground_truth: Set[str] = field(default_factory=set)


@dataclass
class TaskRunState:
    extracted_points: Dict[str, str] = field(default_factory=dict)
    granted_rewards: Set[str] = field(default_factory=set)
    final_submitted: bool = False
    latest_grader_score: float = 0.0


def _build_tasks() -> List[ScreeningTask]:
    medium_scores = {
        "P-M101": 0.88,
        "P-M102": 0.71,
        "P-M103": 0.54,
    }
    return [
        ScreeningTask(
            task_id="easy_eligibility",
            difficulty=TaskDifficulty.EASY,
            title="Phase II EGFR-Mutated NSCLC Eligibility Check",
            brief="Determine whether the candidate meets five binary enrollment criteria.",
            prompt=(
                "Trial CT-NSCLC-201 enrolls adults with metastatic EGFR exon 19 or L858R "
                "non-small cell lung cancer after first-line osimertinib. Candidate E-001 is "
                "47 years old with biopsy-proven metastatic lung adenocarcinoma, EGFR exon 19 "
                "deletion, ECOG 1, no active brain metastases, and adequate hepatic function. "
                "Binary criteria: age >=18, confirmed metastatic NSCLC, sensitizing EGFR "
                "mutation present, ECOG 0-1, no active CNS disease."
            ),
            extraction_targets={
                "age": "47",
                "diagnosis": "metastatic nsclc",
                "biomarker": "egfr exon 19 deletion",
                "ecog": "1",
                "active_cns_disease": "no",
            },
            expected_decision="eligible",
            trial_metadata={
                "trial_id": "CT-NSCLC-201",
                "specialty": "thoracic oncology",
                "binary_criteria": [
                    "adult patient",
                    "metastatic NSCLC confirmed",
                    "sensitizing EGFR mutation",
                    "ECOG 0-1",
                    "no active CNS disease",
                ],
            },
        ),
        ScreeningTask(
            task_id="medium_patient_ranking",
            difficulty=TaskDifficulty.MEDIUM,
            title="Rank Patients for TROP2 ADC Expansion Cohort",
            brief="Rank three real-world candidates by protocol fit-score for an EGFR-mutated NSCLC study.",
            prompt=(
                "Trial CT-LUNG-312 is an antibody-drug conjugate study for metastatic EGFR-mutated "
                "NSCLC after progression on osimertinib. Rank candidates by expected screening fit. "
                "P-M101: 56 years, EGFR exon 19 deletion, post-osimertinib only, ECOG 0, stable "
                "treated brain metastases, CrCl 82 mL/min, AST/ALT normal. P-M102: 63 years, EGFR "
                "L858R, post-osimertinib and platinum, ECOG 1, mild AST elevation 1.4x ULN, no "
                "brain metastases, CrCl 68. P-M103: 59 years, exon 20 insertion, ECOG 1, chronic "
                "prednisone 15 mg, recent palliative radiation 5 days ago, CrCl 61. Internal fit "
                f"scores are predetermined as {medium_scores}."
            ),
            extraction_targets={
                "P-M101_fit_score": "0.88",
                "P-M102_fit_score": "0.71",
                "P-M103_fit_score": "0.54",
                "best_candidate": "P-M101",
                "lowest_candidate": "P-M103",
            },
            expected_decision="P-M101>P-M102>P-M103",
            ranking_ground_truth=["P-M101", "P-M102", "P-M103"],
            trial_metadata={
                "trial_id": "CT-LUNG-312",
                "specialty": "thoracic oncology",
                "ranking_rule": "higher fit-score ranks earlier",
                "fit_scores": medium_scores,
            },
        ),
        ScreeningTask(
            task_id="hard_protocol_deviations",
            difficulty=TaskDifficulty.HARD,
            title="Identify Protocol Deviations from Unstructured Screening Note",
            brief="Extract exclusions and protocol deviations from a realistic unstructured chart note.",
            prompt=(
                "Trial CT-LYMPH-440 is a CD19 bispecific study for relapsed diffuse large B-cell "
                "lymphoma. Exclusions include prednisone >10 mg/day within 7 days, live vaccine "
                "within 30 days, active hepatitis B viremia, ANC <1.0 x10^9/L, and major surgery "
                "within 14 days. Screening note: 'Mr. R is a 68-year-old man with relapsed DLBCL. "
                "He received a shingles live-attenuated vaccine 12 days ago at his PCP visit. He "
                "remains on prednisone 20 mg daily for COPD flare and underwent laparoscopic "
                "cholecystectomy 9 days ago. Labs today: ANC 0.9, HBV DNA undetectable on entecavir, "
                "bilirubin normal. Team asks whether any items trigger screen failure or protocol "
                "deviation before scheduling first dose.'"
            ),
            extraction_targets={
                "age": "68",
                "live_vaccine_days": "12",
                "prednisone_mg": "20",
                "surgery_days": "9",
                "anc": "0.9",
            },
            expected_decision="exclude",
            exclusion_ground_truth={
                "live_vaccine_within_30_days",
                "prednisone_over_10mg",
                "major_surgery_within_14_days",
                "anc_below_1.0",
            },
            trial_metadata={
                "trial_id": "CT-LYMPH-440",
                "specialty": "hematologic malignancy",
                "expected_exclusion_schema": [
                    "live_vaccine_within_30_days",
                    "prednisone_over_10mg",
                    "major_surgery_within_14_days",
                    "anc_below_1.0",
                    "active_hbv_viremia",
                ],
            },
        ),
    ]


def _normalize(value: str | None) -> str:
    return "" if value is None else value.strip().lower()


class EasyEligibilityGrader:
    @staticmethod
    def grade(task: ScreeningTask, task_state: TaskRunState, decision: str | None = None) -> float:
        extracted = sum(
            1
            for field_name, expected in task.extraction_targets.items()
            if _normalize(task_state.extracted_points.get(field_name)) == _normalize(expected)
        )
        extraction_score = extracted / len(task.extraction_targets)
        decision_score = 1.0 if _normalize(decision) == _normalize(task.expected_decision) else 0.0
        return round((0.5 * extraction_score) + (0.5 * decision_score), 4)


class MediumRankingGrader:
    @staticmethod
    def grade(task: ScreeningTask, task_state: TaskRunState, ranking: Sequence[str] | None = None) -> float:
        extracted = sum(
            1
            for field_name, expected in task.extraction_targets.items()
            if _normalize(task_state.extracted_points.get(field_name)) == _normalize(expected)
        )
        extraction_score = extracted / len(task.extraction_targets)
        ranking = list(ranking or [])
        if len(ranking) != len(task.ranking_ground_truth):
            ranking_score = 0.0
        else:
            correct_positions = sum(
                1
                for observed, expected in zip(ranking, task.ranking_ground_truth)
                if observed == expected
            )
            ranking_score = correct_positions / len(task.ranking_ground_truth)
        return round((0.4 * extraction_score) + (0.6 * ranking_score), 4)


class HardDeviationGrader:
    @staticmethod
    def grade(
        task: ScreeningTask,
        task_state: TaskRunState,
        exclusions: Sequence[str] | None = None,
        decision: str | None = None,
    ) -> float:
        extracted = sum(
            1
            for field_name, expected in task.extraction_targets.items()
            if _normalize(task_state.extracted_points.get(field_name)) == _normalize(expected)
        )
        extraction_score = extracted / len(task.extraction_targets)
        predicted = {_normalize(item) for item in exclusions or [] if item}
        if task.exclusion_ground_truth:
            exclusion_score = len(predicted & task.exclusion_ground_truth) / len(task.exclusion_ground_truth)
        else:
            exclusion_score = 0.0
        decision_score = 1.0 if _normalize(decision) == _normalize(task.expected_decision) else 0.0
        return round((0.3 * extraction_score) + (0.4 * exclusion_score) + (0.3 * decision_score), 4)


class ClinicalTrialScreeningEnv:
    """Stateful environment covering easy, medium, and hard screening tasks."""

    def __init__(self) -> None:
        self._tasks = _build_tasks()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._index = 0
        self._task_runs: Dict[str, TaskRunState] = {}
        self._episode_complete = False

    def reset(self) -> ClinicalTrialScreeningObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._index = 0
        self._episode_complete = False
        self._task_runs = {task.task_id: TaskRunState() for task in self._tasks}
        return self._build_observation(
            reward_model=RewardModel(total=0.0),
            feedback="Episode reset. Start with the easy eligibility assessment.",
        )

    def step(self, action: ClinicalTrialScreeningAction) -> ClinicalTrialScreeningObservation:
        self._state.step_count += 1
        task = self._current_task()
        task_state = self._task_runs[task.task_id]
        reward = RewardModel()
        feedback_parts: List[str] = []

        if self._episode_complete:
            reward.penalty += HALLUCINATION_PENALTY
            reward.reasons.append("episode_already_complete")
            reward.total = reward.penalty
            return self._build_observation(reward, "Episode already completed. Reset to start a new run.")

        if action.action_type == "destructive_action":
            reward.penalty += HALLUCINATION_PENALTY
            reward.reasons.append("destructive_action")
            feedback_parts.append("Destructive action blocked in screening workflow.")
        elif action.action_type == "extract_data":
            feedback_parts.append(self._handle_extract_data(task, task_state, action, reward))
        elif action.action_type == "submit_ranking":
            feedback_parts.append(self._handle_submit_ranking(task, task_state, action, reward))
        elif action.action_type == "flag_exclusions":
            feedback_parts.append(self._handle_flag_exclusions(task, task_state, action, reward))
        elif action.action_type == "final_decision":
            feedback_parts.append(self._handle_final_decision(task, task_state, action, reward))
        else:
            reward.penalty += HALLUCINATION_PENALTY
            reward.reasons.append("unsupported_action")
            feedback_parts.append("Unsupported action type for this environment.")

        reward.total = round(
            reward.incremental_reward + reward.final_reward + reward.penalty,
            4,
        )
        return self._build_observation(reward, " ".join(part for part in feedback_parts if part))

    def state(self) -> State:
        return self._state

    def _current_task(self) -> ScreeningTask:
        return self._tasks[self._index]

    def _build_observation(
        self,
        reward_model: RewardModel,
        feedback: str,
    ) -> ClinicalTrialScreeningObservation:
        task = self._current_task()
        task_state = self._task_runs.get(task.task_id, TaskRunState())
        missing_targets = [
            field_name
            for field_name in task.extraction_targets
            if field_name not in task_state.granted_rewards
        ]
        return ClinicalTrialScreeningObservation(
            task_id=task.task_id,
            difficulty=task.difficulty,
            title=task.title,
            brief=task.brief,
            prompt=task.prompt,
            extracted_points=deepcopy(task_state.extracted_points),
            missing_targets=missing_targets,
            available_actions=[
                "extract_data",
                "submit_ranking" if task.difficulty is TaskDifficulty.MEDIUM else "flag_exclusions",
                "final_decision",
                "destructive_action",
            ],
            grader_score=task_state.latest_grader_score,
            reward_breakdown=reward_model,
            feedback=feedback,
            done=self._episode_complete,
            reward=reward_model.total,
            trial_metadata=deepcopy(task.trial_metadata),
        )

    def _handle_extract_data(
        self,
        task: ScreeningTask,
        task_state: TaskRunState,
        action: ClinicalTrialScreeningAction,
        reward: RewardModel,
    ) -> str:
        field_name = action.field_name or ""
        if field_name not in task.extraction_targets:
            reward.penalty += HALLUCINATION_PENALTY
            reward.reasons.append("hallucinated_field")
            return f"Field '{field_name}' is not part of the protocol data model."

        expected = task.extraction_targets[field_name]
        observed = _normalize(action.value)
        if observed == _normalize(expected):
            task_state.extracted_points[field_name] = action.value or ""
            if field_name not in task_state.granted_rewards:
                task_state.granted_rewards.add(field_name)
                reward.incremental_reward += INCREMENTAL_REWARD
                reward.reasons.append(f"correct_extraction:{field_name}")
            task_state.latest_grader_score = self._current_grader_score(task, task_state, action)
            return f"Captured {field_name} correctly."

        reward.penalty += HALLUCINATION_PENALTY
        reward.reasons.append(f"incorrect_extraction:{field_name}")
        return f"Extracted value for {field_name} does not match the chart."

    def _handle_submit_ranking(
        self,
        task: ScreeningTask,
        task_state: TaskRunState,
        action: ClinicalTrialScreeningAction,
        reward: RewardModel,
    ) -> str:
        if task.difficulty is not TaskDifficulty.MEDIUM:
            reward.penalty += HALLUCINATION_PENALTY
            reward.reasons.append("ranking_on_non_medium_task")
            return "Ranking is only valid on the medium task."

        task_state.final_submitted = True
        is_correct = list(action.ranking) == task.ranking_ground_truth
        if is_correct:
            reward.final_reward += FINAL_REWARD
            reward.reasons.append("correct_ranking")
        else:
            reward.penalty += HALLUCINATION_PENALTY
            reward.reasons.append("incorrect_ranking")
        task_state.latest_grader_score = MediumRankingGrader.grade(task, task_state, action.ranking)
        self._advance_task()
        return "Ranking accepted." if is_correct else "Ranking accepted but does not match the deterministic fit ordering."

    def _handle_flag_exclusions(
        self,
        task: ScreeningTask,
        task_state: TaskRunState,
        action: ClinicalTrialScreeningAction,
        reward: RewardModel,
    ) -> str:
        if task.difficulty is not TaskDifficulty.HARD:
            reward.penalty += HALLUCINATION_PENALTY
            reward.reasons.append("exclusion_flag_on_non_hard_task")
            return "Exclusion flagging is reserved for the hard chart-review task."

        predicted = {_normalize(item) for item in action.exclusions if item}
        invalid = predicted - task.exclusion_ground_truth
        if invalid:
            reward.penalty += HALLUCINATION_PENALTY
            reward.reasons.append("hallucinated_exclusion")
            task_state.latest_grader_score = HardDeviationGrader.grade(
                task,
                task_state,
                exclusions=action.exclusions,
            )
            return f"Unsupported exclusion codes submitted: {sorted(invalid)}."

        task_state.latest_grader_score = HardDeviationGrader.grade(
            task,
            task_state,
            exclusions=action.exclusions,
        )
        return "Exclusion codes recorded."

    def _handle_final_decision(
        self,
        task: ScreeningTask,
        task_state: TaskRunState,
        action: ClinicalTrialScreeningAction,
        reward: RewardModel,
    ) -> str:
        decision = action.value or ""
        task_state.final_submitted = True
        if task.difficulty is TaskDifficulty.MEDIUM:
            reward.penalty += HALLUCINATION_PENALTY
            reward.reasons.append("decision_instead_of_ranking")
            task_state.latest_grader_score = MediumRankingGrader.grade(task, task_state)
            return "Medium task requires submit_ranking rather than final_decision."

        is_correct = _normalize(decision) == _normalize(task.expected_decision)
        if is_correct:
            reward.final_reward += FINAL_REWARD
            reward.reasons.append("correct_final_decision")
        else:
            reward.penalty += HALLUCINATION_PENALTY
            reward.reasons.append("incorrect_final_decision")

        if task.difficulty is TaskDifficulty.EASY:
            task_state.latest_grader_score = EasyEligibilityGrader.grade(task, task_state, decision=decision)
        else:
            exclusions = sorted(task.exclusion_ground_truth)
            task_state.latest_grader_score = HardDeviationGrader.grade(
                task,
                task_state,
                exclusions=exclusions,
                decision=decision,
            )

        self._advance_task()
        return "Final screening decision accepted." if is_correct else "Final decision conflicts with protocol evidence."

    def _current_grader_score(
        self,
        task: ScreeningTask,
        task_state: TaskRunState,
        action: ClinicalTrialScreeningAction,
    ) -> float:
        if task.difficulty is TaskDifficulty.EASY:
            return EasyEligibilityGrader.grade(task, task_state)
        if task.difficulty is TaskDifficulty.MEDIUM:
            return MediumRankingGrader.grade(task, task_state)
        return HardDeviationGrader.grade(task, task_state, exclusions=action.exclusions)

    def _advance_task(self) -> None:
        if self._index == len(self._tasks) - 1:
            self._episode_complete = True
            return
        self._index += 1

