"""
inference.py — DataWrangler Agent Inference Script
===================================================
Official submission entry point for the hackathon.

Required environment variables (set in .env or shell):
  API_BASE_URL  : The LLM API endpoint
  MODEL_NAME    : The model identifier
  HF_TOKEN      : Your API key (HuggingFace / LLM provider)
"""

import os
import re
import json
import sys
import time

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

from openai import OpenAI
from env import DataWranglerEnv
from models import Action
from tasks import grader_easy, grader_medium, grader_hard

# ─── Configuration & Client ────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Detect placeholder token
IS_PLACEHOLDER = HF_TOKEN == "" or not HF_TOKEN

# (Warning moved to inside main)

client = OpenAI(
    api_key=HF_TOKEN if not IS_PLACEHOLDER else "unused",
    base_url=API_BASE_URL,
)

# ─── Task Definitions ──────────────────────────────────────────────────────────
TASKS = {
    "easy": {
        "state": {
            "total_rows": 100,
            "current_accuracy": 0.5,
            "columns": {
                "bad_col": {
                    "type": "numeric", "original_dtype": "int",
                    "missing_values": 100, "unique_count": 0, "has_outliers": False
                }
            }
        },
        "grader": grader_easy,
        "description": "Drop a completely empty column and train."
    },
    "medium": {
        "state": {
            "total_rows": 100,
            "current_accuracy": 0.5,
            "columns": {
                "age": {
                    "type": "numeric", "original_dtype": "int",
                    "missing_values": 20, "unique_count": 40, "has_outliers": False
                }
            }
        },
        "grader": grader_medium,
        "description": "Fill missing numeric values and train."
    },
    "hard": {
        "state": {
            "total_rows": 100,
            "current_accuracy": 0.4,
            "columns": {
                "useless":  {"type": "categorical", "original_dtype": "categorical", "missing_values": 100, "unique_count": 100, "has_outliers": False},
                "salary":   {"type": "numeric",     "original_dtype": "int",         "missing_values": 30,  "unique_count": 70,  "has_outliers": True},
                "city":     {"type": "categorical", "original_dtype": "categorical", "missing_values": 5,   "unique_count": 10,  "has_outliers": False},
            }
        },
        "grader": grader_hard,
        "description": "Clean a messy dataset with mixed types and missing data, then train."
    }
}


# ─── JSON Extraction (robust, no response_format dependency) ───────────────────
def extract_json(text: str) -> dict:
    """Extract JSON from LLM response — handles markdown code blocks & raw JSON."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try markdown code block ```json ... ```
    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    # Try first JSON object in text
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError(f"Could not extract JSON from LLM response: {text[:300]}")


def get_heuristic_action(obs) -> Action:
    """Fallback logic: selects the best cleaning action based on dataset state."""
    cols = obs.columns
    threshold = int(obs.total_rows * 0.9)

    # 1. Drop columns with mostly missing values
    for col, info in cols.items():
        if info.get("missing_values", 0) >= threshold:
            return Action(command="drop_column", target_column=col)

    # 2. Fill missing values (Prioritize Median for outliers)
    for col, info in cols.items():
        missing = info.get("missing_values", 0)
        if missing > 0:
            if info.get("type") == "numeric":
                if info.get("has_outliers"):
                    return Action(command="fill_median", target_column=col)
                return Action(command="fill_mean", target_column=col)
            else:
                return Action(command="fill_mode", target_column=col)

    # 3. Clip outliers
    for col, info in cols.items():
        if info.get("type") == "numeric" and info.get("has_outliers"):
            return Action(command="clip_outliers", target_column=col)

    # 4. Default to training
    return Action(command="train_model", target_column=None)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def is_ready_to_train(obs) -> bool:
    """Local check — no LLM call needed if dataset is already clean."""
    for col, info in obs.columns.items():
        if info.get("missing_values", 0) > 0:
            return False
    return True  # Also True when 0 columns remain


# ─── Agent Loop ────────────────────────────────────────────────────────────────
def run_agent(env, max_steps: int = 15) -> list:
    """Run AI agent — chooses and executes cleaning actions until dataset is ready."""
    obs = env.state()
    trajectory = []
    done = False

    while not done and env.step_count < max_steps:
        obs = env.state()  # Refresh each iteration

        # ── Smart shortcut: if already clean, train without calling LLM ──────
        if is_ready_to_train(obs):
            print("Dataset is clean — calling train_model directly", file=sys.stderr)
            action = Action(command="train_model", target_column=None)
            obs, reward, done, info = env.step(action)
            trajectory.append({"action": action.model_dump(), "reward": reward.value, "info": info})
            print(f"[STEP] step={env.step_count} reward={reward.value:.4f}", flush=True)
            print(f"train_model(None) | reward={reward.value:+.2f} | acc={info['accuracy']:.2f}", file=sys.stderr)
            break

        history = [
            f"{t['action']['command']}({t['action'].get('target_column')})"
            for t in trajectory
        ]

        cols_summary = ", ".join(
            f"{c}(type={i.get('type')}, missing={i.get('missing_values')}, outliers={i.get('has_outliers')})"
            for c, i in obs.columns.items()
        ) or "NO COLUMNS LEFT"

        threshold = int(obs.total_rows * 0.9)
        prompt = f"""You are a strict data cleaning AI. Clean this dataset step by step.

CURRENT STATE:
- Total rows: {obs.total_rows}
- Columns: {cols_summary}

PAST ACTIONS: {history if history else "None"}

DECISION RULES (MUST FOLLOW IN ORDER):
1. If missing_values >= {threshold} → "drop_column"
2. If type=numeric AND missing_values > 0 AND has_outliers=True → "fill_median"
3. If type=numeric AND missing_values > 0 AND has_outliers=False → "fill_mean"
4. If type=categorical AND missing_values > 0 → "fill_mode"
5. If type=numeric AND missing_values=0 AND has_outliers=True → "clip_outliers"

FORBIDDEN ACTS:
- NEVER use 'drop_column' if missing_values < {threshold}!! (CRITICAL)
- NEVER repeat a command on the same target_column.
- NEVER use fill commands if missing_values=0.

VALID COMMANDS: drop_column, fill_mean, fill_median, fill_mode, clip_outliers

Respond with ONLY this JSON format:
{{"reasoning": "apply rule X because Y", "command": "<ONE OF THE VALID COMMANDS>", "target_column": "col_name_or_null"}}"""

        action = None
        for attempt in range(3):
            try:
                if IS_PLACEHOLDER:
                    raise ValueError("HF_TOKEN is a placeholder — skipping LLM call.")

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200
                )
                raw_text = response.choices[0].message.content
                decision = extract_json(raw_text)

                target = decision.get("target_column")
                if target in ("null", "", "None", None, "null"):
                    target = None

                valid_cmds = {"drop_column", "fill_mean", "fill_median", "fill_mode", "clip_outliers", "train_model"}
                cmd = decision.get("command", "")
                
                if not target or target not in obs.columns:
                    if cmd != "train_model":
                        print(f"  [Safety] Overriding invalid target '{target}', selecting valid one.")
                        valid_fill = [c for c, i in obs.columns.items() if i.get("missing_values", 0) > 0]
                        if valid_fill:
                            target = valid_fill[0]
                            cmd = "fill_mean" if obs.columns[target].get("type") == "numeric" else "fill_mode"
                        else:
                            cmd = "train_model"
                            target = None
                else:
                    target_info = obs.columns.get(target, {})
                    missing = target_info.get("missing_values", 0)
                    is_numeric = target_info.get("type") == "numeric"
                    
                    if cmd not in valid_cmds:
                        print(f"  [Safety] Overriding invalid/unknown command '{cmd}' on '{target}'")
                        cmd = "fill_mean" if is_numeric else "fill_mode"
                    
                    bad_fill = cmd in ("fill_mean", "fill_median", "fill_mode") and missing == 0
                    bad_drop = cmd == "drop_column" and missing < threshold
                    bad_clip = cmd == "clip_outliers" and not target_info.get("has_outliers")

                    if bad_fill or bad_drop or bad_clip:
                        print(f"  [Safety] Action '{cmd}' on '{target}' is logically invalid, overriding!")
                        needs_fill = [c for c, i in obs.columns.items() if i.get("missing_values", 0) > 0]
                        needs_clip = [c for c, i in obs.columns.items() if i.get("has_outliers")]
                        
                        if needs_fill:
                            target = needs_fill[0]
                            cmd = "fill_mean" if obs.columns[target].get("type") == "numeric" else "fill_mode"
                        elif needs_clip:
                            target = needs_clip[0]
                            cmd = "clip_outliers"
                        else:
                            cmd = "train_model"
                            target = None

                # Final fallback: ensure cmd is train_model if it's anything else and target still None
                if target is None:
                    cmd = "train_model"

                action = Action(command=cmd, target_column=target)
                print(f"[{env.step_count+1}] {decision.get('reasoning', '')[:80]}", file=sys.stderr)
                break # Success
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    print(f"Rate limit hit. Waiting 2s... (Attempt {attempt+1}/3)", file=sys.stderr)
                    time.sleep(2)
                else:
                    print(f"LLM error ({type(e).__name__}). Retrying... (Attempt {attempt+1}/3)", file=sys.stderr)
                    # Show more detail for first error
                    if attempt == 0:
                        print(f"Details: {str(e)[:100]}...", file=sys.stderr)
                    time.sleep(1)

        if action is None:
            print("Model unavailable (or failed) — using HEURISTIC fallback logic", file=sys.stderr)
            action = get_heuristic_action(obs)

        obs, reward, done, info = env.step(action)
        trajectory.append({"action": action.model_dump(), "reward": reward.value, "info": info})
        print(f"[STEP] step={env.step_count} reward={reward.value:.4f}", flush=True)
        print(f"{action.command}({action.target_column}) | reward={reward.value:+.2f} | acc={info['accuracy']:.2f}", file=sys.stderr)

    return trajectory


def run_task(task_id: str) -> float:
    print(f"[START] task={task_id}", flush=True)
    task = TASKS[task_id]
    env = DataWranglerEnv(task["state"])
    env.reset()
    trajectory = run_agent(env)
    score = task["grader"](trajectory)
    print(f"[END] task={task_id} score={score:.4f} steps={env.step_count}", flush=True)
    return score


# ─── Main ──────────────────────────────────────────────────────────────────────
def run_all_tasks():
    """Run the agent on all predefined tasks and return the results."""
    if IS_PLACEHOLDER:
        print("WARNING: HF_TOKEN is not configured. Heuristic Mode active.", file=sys.stderr)

    results = {
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "tasks": {}
    }
    
    for task_id in TASKS:
        score = run_task(task_id)
        results["tasks"][task_id] = score
    
    return results


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    results = run_all_tasks()
    
    print("\nFinal Scores Summary:", file=sys.stderr)
    for task_id, score in results["tasks"].items():
        print(f"  {task_id:<10}: {score:.4f}", file=sys.stderr)
    print("Inference complete.", file=sys.stderr)

if __name__ == "__main__":
    main()
