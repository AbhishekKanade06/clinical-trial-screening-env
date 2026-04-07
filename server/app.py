"""FastAPI app exposing the clinical trial screening environment."""

from __future__ import annotations

import argparse
import json

import uvicorn
from fastapi.responses import HTMLResponse, RedirectResponse
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from env import ClinicalTrialScreeningEnv
from models import ClinicalTrialScreeningAction, ClinicalTrialScreeningObservation


class OpenEnvClinicalTrialScreeningEnvironment(Environment):
    """OpenEnv-compatible wrapper around the reusable simulator."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._runtime = ClinicalTrialScreeningEnv()

    def reset(self) -> ClinicalTrialScreeningObservation:
        return self._runtime.reset()

    def step(self, action: ClinicalTrialScreeningAction) -> ClinicalTrialScreeningObservation:  # type: ignore[override]
        return self._runtime.step(action)

    @property
    def state(self) -> State:
        return self._runtime.state()


app = create_app(
    OpenEnvClinicalTrialScreeningEnvironment,
    ClinicalTrialScreeningAction,
    ClinicalTrialScreeningObservation,
    env_name="clinical_trial_screening",
    max_concurrent_envs=4,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/web", status_code=307)


@app.get("/web", include_in_schema=False)
def web_ui() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Clinical Trial Screening</title>
    <style>
      :root {
        --bg: #f5f1e8;
        --panel: #fffdf8;
        --ink: #1e1f1c;
        --muted: #666a61;
        --line: #d9d0c1;
        --accent: #0f766e;
        --accent-2: #c2410c;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 30%),
          radial-gradient(circle at bottom right, rgba(194, 65, 12, 0.10), transparent 28%),
          var(--bg);
      }
      .wrap {
        max-width: 1180px;
        margin: 0 auto;
        padding: 32px 20px 48px;
      }
      .hero {
        display: grid;
        gap: 18px;
        margin-bottom: 24px;
      }
      .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 12px;
        color: var(--accent);
        font-weight: 700;
      }
      h1 {
        margin: 0;
        font-size: clamp(2.2rem, 5vw, 4.6rem);
        line-height: 0.95;
        max-width: 10ch;
      }
      .lead {
        max-width: 72ch;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.6;
      }
      .actions {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }
      button {
        border: 1px solid var(--line);
        background: var(--panel);
        color: var(--ink);
        padding: 12px 16px;
        border-radius: 999px;
        cursor: pointer;
        font: inherit;
      }
      button.primary {
        background: var(--accent);
        color: white;
        border-color: var(--accent);
      }
      .grid {
        display: grid;
        gap: 18px;
        grid-template-columns: 1.1fr 0.9fr;
      }
      .card {
        background: rgba(255, 253, 248, 0.94);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 20px;
        box-shadow: 0 16px 40px rgba(30, 31, 28, 0.06);
      }
      .card h2 {
        margin: 0 0 12px;
        font-size: 1.1rem;
      }
      .meta {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 16px;
      }
      .meta div {
        padding: 12px;
        border-radius: 14px;
        background: #f7f3eb;
        border: 1px solid #ebe1d0;
      }
      .label {
        display: block;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 6px;
      }
      pre, textarea {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: #fcfaf5;
        color: var(--ink);
        padding: 14px;
        font-family: "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 13px;
        line-height: 1.5;
      }
      textarea {
        min-height: 220px;
        resize: vertical;
      }
      .status {
        color: var(--muted);
        margin-bottom: 10px;
        min-height: 24px;
      }
      .quick-actions {
        display: grid;
        gap: 10px;
        margin-top: 14px;
      }
      .quick-actions button {
        border-radius: 16px;
        text-align: left;
      }
      .footnote {
        margin-top: 18px;
        color: var(--muted);
        font-size: 0.92rem;
      }
      @media (max-width: 900px) {
        .grid { grid-template-columns: 1fr; }
        .meta { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <section class="hero">
        <div class="eyebrow">OpenEnv Local Interface</div>
        <h1>Clinical Trial Screening Simulator</h1>
        <div class="lead">
          Inspect the current task, reset the episode, and step the environment from the browser.
          This UI talks directly to the OpenEnv API endpoints exposed by the local container.
        </div>
        <div class="actions">
          <button class="primary" id="resetBtn">Reset Episode</button>
          <button id="stateBtn">Refresh State</button>
          <button id="schemaBtn">Load Schema</button>
          <button id="docsBtn">Open API Docs</button>
        </div>
      </section>

      <section class="grid">
        <div class="card">
          <h2>Current Observation</h2>
          <div class="meta">
            <div><span class="label">Task</span><strong id="taskId">-</strong></div>
            <div><span class="label">Difficulty</span><strong id="difficulty">-</strong></div>
            <div><span class="label">Reward</span><strong id="reward">0.00</strong></div>
          </div>
          <div class="status" id="status">Ready.</div>
          <pre id="observationBox">{}</pre>
        </div>

        <div class="card">
          <h2>Step The Environment</h2>
          <div class="label">Action JSON</div>
          <textarea id="actionBox">{
  "action_type": "extract_data",
  "field_name": "age",
  "value": "47"
}</textarea>
          <div class="quick-actions">
            <button data-preset="easy">Preset: Easy extraction</button>
            <button data-preset="medium">Preset: Medium ranking</button>
            <button data-preset="hard">Preset: Hard exclusions</button>
            <button class="primary" id="stepBtn">Send Step</button>
          </div>
          <div class="footnote">
            Presets are examples only. You can edit the JSON before submitting.
          </div>
        </div>
      </section>
    </div>

    <script>
      const observationBox = document.getElementById("observationBox");
      const actionBox = document.getElementById("actionBox");
      const statusBox = document.getElementById("status");
      const taskId = document.getElementById("taskId");
      const difficulty = document.getElementById("difficulty");
      const reward = document.getElementById("reward");

      const presets = {
        easy: {
          action_type: "extract_data",
          field_name: "diagnosis",
          value: "metastatic nsclc"
        },
        medium: {
          action_type: "submit_ranking",
          ranking: ["P-M101", "P-M102", "P-M103"]
        },
        hard: {
          action_type: "flag_exclusions",
          exclusions: [
            "live_vaccine_within_30_days",
            "prednisone_over_10mg",
            "major_surgery_within_14_days",
            "anc_below_1.0"
          ]
        }
      };

      function setStatus(message) {
        statusBox.textContent = message;
      }

      function render(payload) {
        const observation = payload.observation || payload;
        taskId.textContent = observation.task_id || "-";
        difficulty.textContent = observation.difficulty || "-";
        reward.textContent = Number(payload.reward || observation.reward || 0).toFixed(2);
        observationBox.textContent = JSON.stringify(payload, null, 2);
      }

      async function callApi(path, options = {}) {
        const response = await fetch(path, {
          headers: { "Content-Type": "application/json" },
          ...options
        });
        if (!response.ok) {
          const body = await response.text();
          throw new Error(`${response.status} ${response.statusText}: ${body}`);
        }
        return response.json();
      }

      async function resetEnv() {
        setStatus("Resetting episode...");
        const payload = await callApi("/reset", { method: "POST", body: "{}" });
        render(payload);
        setStatus("Episode reset.");
      }

      async function getState() {
        setStatus("Loading state...");
        const payload = await callApi("/state");
        observationBox.textContent = JSON.stringify(payload, null, 2);
        setStatus("State loaded.");
      }

      async function getSchema() {
        setStatus("Loading schema...");
        const payload = await callApi("/schema");
        observationBox.textContent = JSON.stringify(payload, null, 2);
        setStatus("Schema loaded.");
      }

      async function stepEnv() {
        setStatus("Sending step...");
        const action = JSON.parse(actionBox.value);
        const payload = await callApi("/step", {
          method: "POST",
          body: JSON.stringify(action)
        });
        render(payload);
        setStatus("Step complete.");
      }

      document.getElementById("resetBtn").addEventListener("click", () => resetEnv().catch(err => setStatus(err.message)));
      document.getElementById("stateBtn").addEventListener("click", () => getState().catch(err => setStatus(err.message)));
      document.getElementById("schemaBtn").addEventListener("click", () => getSchema().catch(err => setStatus(err.message)));
      document.getElementById("stepBtn").addEventListener("click", () => stepEnv().catch(err => setStatus(err.message)));
      document.getElementById("docsBtn").addEventListener("click", () => window.open("/docs", "_blank"));

      document.querySelectorAll("[data-preset]").forEach((button) => {
        button.addEventListener("click", () => {
          const preset = presets[button.dataset.preset];
          actionBox.value = JSON.stringify(preset, null, 2);
        });
      });

      resetEnv().catch(err => setStatus(err.message));
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
