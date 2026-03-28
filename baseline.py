import os
import re
import json
import time
from openai import OpenAI
from env import DataWranglerEnv
from models import Action
from data_loader import load_task_state
from tasks import grader_easy, grader_medium, grader_hard

# Use the required submission environment variables
client = OpenAI(
    api_key=os.environ.get("HF_TOKEN"),
    base_url=os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1"),
)


def extract_json(text: str) -> dict:
    """Robust JSON extractor — works even if model wraps in markdown."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError(f"Could not parse JSON: {text[:200]}")


def is_ready_to_train(obs) -> bool:
    for col, info in obs.columns.items():
        if info.get("missing_values", 0) > 0:
            return False
    return True


def run_agent(env, max_steps=15):
    """AI agent loop — cleans dataset step by step."""
    trajectory = []
    done = False

    while not done and env.step_count < max_steps:
        obs = env.state()

        if is_ready_to_train(obs):
            action = Action(command="train_model", target_column=None)
            obs, reward, done, info = env.step(action)
            trajectory.append({"action": action.model_dump(), "reward": reward.value, "info": info})
            print(f"  Action: train_model(None) | Reward: {reward.value:+.2f} | {reward.reason[:50]}")
            break

        history = [f"{t['action']['command']}({t['action'].get('target_column')})" for t in trajectory]

        cols_summary = ", ".join(
            f"{c}(type={i.get('type')}, missing={i.get('missing_values')}, outliers={i.get('has_outliers')})"
            for c, i in obs.columns.items()
        ) or "NO COLUMNS LEFT"

        threshold = int(obs.total_rows * 0.9)
        prompt = f"""You are a strict data cleaning AI. Clean this dataset step by step.

CURRENT STATE:
- Columns: {cols_summary}

PAST ACTIONS: {history if history else "None"}

DECISION RULES:
1. If missing_values >= {threshold} → "drop_column"
2. numeric + missing_values > 0 + has_outliers=True → "fill_median"
3. numeric + missing_values > 0 + has_outliers=False → "fill_mean"
4. categorical + missing_values > 0 → "fill_mode"
5. numeric + missing_values=0 + has_outliers=True → "clip_outliers"

FORBIDDEN:
- NEVER drop_column if missing_values < {threshold}!! (CRITICAL)
- Never fill if missing_values=0.

VALID COMMANDS: drop_column, fill_mean, fill_median, fill_mode, clip_outliers

Respond with ONLY JSON: {{"reasoning": "why", "command": "<ONE OF THE VALID COMMANDS>", "target_column": "col or null"}}"""

        action = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=os.environ.get("MODEL_NAME", "llama-3.1-8b-instant"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200
                )
                decision = extract_json(response.choices[0].message.content)
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
                print(f"  🤖 [{env.step_count+1}] {decision.get('reasoning', '')[:80]}")
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    print(f"  ⚠️ Rate limit. Waiting 2s... ({attempt+1}/3)")
                    time.sleep(2)
                else:
                    print(f"  ⚠️ LLM err. Retrying... ({attempt+1}/3)")
                    time.sleep(1)
        
        if action is None:
            print(f"  ❌ Model failed 3 times, fallback to train_model")
            action = Action(command="train_model", target_column=None)

        obs, reward, done, info = env.step(action)
        trajectory.append({"action": action.model_dump(), "reward": reward.value, "info": info})
        print(f"  Action: {action.command}({action.target_column}) | Reward: {reward.value:+.2f} | {reward.reason[:60]}")

    return trajectory


def run_single_task(initial_state: dict, grader_func, max_steps=15) -> float:
    print("\n--- TASK START ---")
    env = DataWranglerEnv(initial_state)
    env.reset()
    trajectory = run_agent(env, max_steps)
    return grader_func(trajectory)


def run_all_baselines() -> dict:
    dummy_easy = {
        "total_rows": 100, "current_accuracy": 0.5,
        "columns": {
            "bad_col": {"type": "numeric", "original_dtype": "int", "missing_values": 100, "unique_count": 0, "has_outliers": False}
        }
    }
    dummy_medium = {
        "total_rows": 100, "current_accuracy": 0.5,
        "columns": {
            "age": {"type": "numeric", "original_dtype": "int", "missing_values": 20, "unique_count": 40, "has_outliers": False}
        }
    }
    dummy_hard = {
        "total_rows": 100, "current_accuracy": 0.4,
        "columns": {
            "useless": {"type": "categorical", "original_dtype": "categorical", "missing_values": 100, "unique_count": 100, "has_outliers": False},
            "salary":  {"type": "numeric",     "original_dtype": "int",         "missing_values": 30,  "unique_count": 70,  "has_outliers": True},
            "city":    {"type": "categorical", "original_dtype": "categorical", "missing_values": 5,   "unique_count": 10,  "has_outliers": False},
        }
    }
    return {
        "easy":   run_single_task(dummy_easy,   grader_easy),
        "medium": run_single_task(dummy_medium, grader_medium),
        "hard":   run_single_task(dummy_hard,   grader_hard),
    }


def clean_uploaded_file(custom_env):
    """Called from app.py /baseline to clean user-uploaded file."""
    print("\n--- AI IS CLEANING YOUR UPLOADED FILE ---")
    run_agent(custom_env)


if __name__ == "__main__":
    print(run_all_baselines())