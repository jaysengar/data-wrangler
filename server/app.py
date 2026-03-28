from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from models import Action, Observation
from env import DataWranglerEnv
import tasks
import pandas as pd
import io
import os
from data_loader import df_to_state

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="DataWrangler OpenEnv API")

# ── Default environment state ─────────────────────────────────────────────────
initial_state = {
    "total_rows": 1000,
    "current_accuracy": 0.5,
    "columns": {
        "age":       {"type": "numeric",     "original_dtype": "int",         "missing_values": 50,   "unique_count": 60,  "has_outliers": False},
        "useless_id":{"type": "categorical", "original_dtype": "categorical",  "missing_values": 1000, "unique_count": 1000,"has_outliers": False},
    }
}
env = DataWranglerEnv(initial_state, df=None)
current_trajectory = []


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok", "message": "DataWrangler is running. Upload datasets to test!"}


# ── Upload custom CSV / XLSX ──────────────────────────────────────────────────
@app.post("/upload")
async def upload_custom_data(file: UploadFile = File(...)):
    global env, current_trajectory
    contents = await file.read()
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Only .csv and .xlsx files are supported.")

        custom_state = df_to_state(df, base_accuracy=0.3)
        env = DataWranglerEnv(custom_state, df=df)
        current_trajectory = []
        return {
            "message": f"Successfully loaded '{file.filename}'!",
            "rows": custom_state["total_rows"],
            "columns": list(custom_state["columns"].keys()),
            "initial_observation": env.reset().model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


# ── Download cleaned data ─────────────────────────────────────────────────────
@app.get("/download")
def download_cleaned_data():
    if env.df is None:
        raise HTTPException(status_code=400, detail="Upload a dataset first via /upload!")
    stream = io.StringIO()
    env.df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=cleaned_data.csv"
    return response


# ── OpenEnv required endpoints ────────────────────────────────────────────────
@app.api_route("/reset", methods=["GET", "POST"], response_model=Observation)
def reset_env():
    global current_trajectory
    current_trajectory = []
    return env.reset()


@app.get("/state", response_model=Observation)
def get_state():
    return env.state()


@app.post("/step")
def step_env(action: Action):
    global current_trajectory
    obs, reward, done, info = env.step(action)
    current_trajectory.append({
        "action": action.model_dump(),
        "reward": reward.value,
        "reason": reward.reason,
        "info": info
    })
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }


# ── Tasks & Grader ────────────────────────────────────────────────────────────
@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {"id": "easy",   "description": "Drop a completely empty column and train."},
            {"id": "medium", "description": "Fill missing numeric values and train."},
            {"id": "hard",   "description": "Clean a messy dataset with mixed types and missing data before training."}
        ],
        "action_schema": Action.model_json_schema(),
        "available_commands": [
            "drop_column", "fill_mean", "fill_median", "fill_mode",
            "clip_outliers", "drop_duplicates", "encode_category", "train_model"
        ]
    }


@app.get("/grader")
def get_grader(task_id: str = "medium"):
    global current_trajectory
    if task_id == "easy":
        score = tasks.grader_easy(current_trajectory)
    elif task_id == "medium":
        score = tasks.grader_medium(current_trajectory)
    elif task_id == "hard":
        score = tasks.grader_hard(current_trajectory)
    else:
        raise HTTPException(status_code=400, detail=f"Invalid task_id '{task_id}'. Use: easy, medium, hard")
    return {"task_id": task_id, "score": score, "trajectory_length": len(current_trajectory)}


# ── Baseline (LAZY IMPORT — avoids startup crash if env vars not set) ─────────
@app.get("/baseline")
def run_baseline_endpoint():
    global env
    try:
        import baseline as _baseline  # Lazy import: only loaded when this endpoint is hit
        scores = _baseline.run_all_baselines()
        custom_msg = "Default hackathon tasks completed."

        if getattr(env, "df", None) is not None:
            _baseline.clean_uploaded_file(env)
            custom_msg = "AI cleaned your uploaded file too! Grab it via /download"

        return {
            "status": "success",
            "baseline_scores": scores,
            "custom_file_status": custom_msg
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n\nTraceback:\n{tb}")


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)