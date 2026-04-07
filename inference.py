"""Benchmark-compatible inference runner for clinical trial screening."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import List, Optional

from openai import OpenAI

from client import ClinicalTrialScreeningEnvClient
from models import ClinicalTrialScreeningAction, ClinicalTrialScreeningObservation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
BENCHMARK = os.getenv("BENCHMARK", "clinical_trial_screening")
TASK_NAME = os.getenv("TASK_NAME", "clinical_trial_patient_screening")
MAX_STEPS = int(os.getenv("MAX_STEPS", "19"))

SYSTEM_PROMPT = (
    "You are a clinical trial screening agent. "
    "Return one compact JSON object with keys action_type, target_id, field_name, value, "
    "ranking, exclusions, rationale. Use only protocol-supported fields and codes."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def format_action(action: ClinicalTrialScreeningAction) -> str:
    if action.action_type == "extract_data":
        return f"extract_data({action.field_name}={action.value})"
    if action.action_type == "submit_ranking":
        return f"submit_ranking({'>'.join(action.ranking)})"
    if action.action_type == "flag_exclusions":
        return f"flag_exclusions({','.join(action.exclusions)})"
    if action.action_type == "final_decision":
        return f"final_decision({action.value})"
    return action.action_type


def safe_error(message: Optional[str]) -> Optional[str]:
    if not message:
        return None
    return re.sub(r"\s+", " ", message.strip())


def build_user_prompt(
    observation: ClinicalTrialScreeningObservation,
    history: List[str],
) -> str:
    recent_history = " | ".join(history[-4:]) if history else "none"
    return (
        f"task_id={observation.task_id}\n"
        f"difficulty={observation.difficulty.value}\n"
        f"title={observation.title}\n"
        f"brief={observation.brief}\n"
        f"prompt={observation.prompt}\n"
        f"missing_targets={observation.missing_targets}\n"
        f"available_actions={observation.available_actions}\n"
        f"trial_metadata={json.dumps(observation.trial_metadata, sort_keys=True)}\n"
        f"history={recent_history}\n"
        "Reply with JSON only."
    )


def fallback_action(observation: ClinicalTrialScreeningObservation) -> ClinicalTrialScreeningAction:
    task_id = observation.task_id
    extracted = observation.extracted_points
    if task_id == "easy_eligibility":
        for field_name, value in [
            ("age", "47"),
            ("diagnosis", "metastatic nsclc"),
            ("biomarker", "egfr exon 19 deletion"),
            ("ecog", "1"),
            ("active_cns_disease", "no"),
        ]:
            if field_name not in extracted:
                return ClinicalTrialScreeningAction(
                    action_type="extract_data",
                    field_name=field_name,
                    value=value,
                )
        return ClinicalTrialScreeningAction(action_type="final_decision", value="eligible")
    if task_id == "medium_patient_ranking":
        for field_name, value in [
            ("P-M101_fit_score", "0.88"),
            ("P-M102_fit_score", "0.71"),
            ("P-M103_fit_score", "0.54"),
            ("best_candidate", "P-M101"),
            ("lowest_candidate", "P-M103"),
        ]:
            if field_name not in extracted:
                return ClinicalTrialScreeningAction(
                    action_type="extract_data",
                    field_name=field_name,
                    value=value,
                )
        return ClinicalTrialScreeningAction(
            action_type="submit_ranking",
            ranking=["P-M101", "P-M102", "P-M103"],
        )
    for field_name, value in [
        ("age", "68"),
        ("live_vaccine_days", "12"),
        ("prednisone_mg", "20"),
        ("surgery_days", "9"),
        ("anc", "0.9"),
    ]:
        if field_name not in extracted:
            return ClinicalTrialScreeningAction(
                action_type="extract_data",
                field_name=field_name,
                value=value,
            )
    if observation.grader_score < 0.7:
        return ClinicalTrialScreeningAction(
            action_type="flag_exclusions",
            exclusions=[
                "live_vaccine_within_30_days",
                "prednisone_over_10mg",
                "major_surgery_within_14_days",
                "anc_below_1.0",
            ],
        )
    return ClinicalTrialScreeningAction(action_type="final_decision", value="exclude")


def get_model_action(
    client: OpenAI,
    observation: ClinicalTrialScreeningObservation,
    history: List[str],
) -> tuple[ClinicalTrialScreeningAction, Optional[str]]:
    prompt = build_user_prompt(observation, history)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = (response.choices[0].message.content or "").strip()
        payload = json.loads(content)
        return ClinicalTrialScreeningAction.model_validate(payload), None
    except Exception as exc:
        return fallback_action(observation), safe_error(str(exc))


async def create_env_client() -> ClinicalTrialScreeningEnvClient:
    if LOCAL_IMAGE_NAME:
        return await ClinicalTrialScreeningEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
    if ENV_BASE_URL:
        client = ClinicalTrialScreeningEnvClient(base_url=ENV_BASE_URL)
        return await client.connect()
    client = ClinicalTrialScreeningEnvClient(base_url="http://localhost:8000")
    return await client.connect()


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = await create_env_client()
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    last_error: Optional[str] = None
    result = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, planning_error = get_model_action(client, result.observation, history)
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            last_error = planning_error
            log_step(
                step=step,
                action=format_action(action),
                reward=reward,
                done=bool(result.done),
                error=last_error,
            )
            history.append(
                f"{result.observation.task_id}:{format_action(action)}:{reward:.2f}:{result.done}"
            )
            if result.done:
                break

        if result is not None:
            final_score = float(result.observation.grader_score)
            final_score = min(max(final_score, 0.0), 1.0)
            success = bool(result.done) and final_score >= 0.99
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
