---
title: DataWrangler-v1
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🧹 DataWrangler — OpenEnv AI Agent Environment

An **AutoML agent environment** where an AI agent learns to clean messy datasets step-by-step to maximize model accuracy.

## 🚀 Quick Start

### Required Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | The API endpoint for the LLM (e.g. `https://api.groq.com/openai/v1`) |
| `MODEL_NAME` | The model identifier (e.g. `llama-3.1-8b-instant`) |
| `HF_TOKEN` | Your Hugging Face / LLM API key |

### Running Locally

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_api_key_here"

# Run the inference script
python Inference.py

# Or start the API server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Running with Docker

```bash
docker build -t datawrangler .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.groq.com/openai/v1" \
  -e MODEL_NAME="llama-3.1-8b-instant" \
  -e HF_TOKEN="your_api_key_here" \
  datawrangler
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET / POST | `/reset` | Reset environment, returns initial Observation |
| GET | `/state` | Get current environment state |
| POST | `/step` | Take an action, returns observation + reward |
| GET | `/tasks` | List all available tasks |
| GET | `/grader?task_id=<id>` | Score the current trajectory |
| POST | `/upload` | Upload a custom CSV/XLSX dataset |
| GET | `/download` | Download the cleaned dataset |
| GET | `/baseline` | Run the AI baseline agent on all tasks |

## 🎯 Tasks

| Task | Description |
|---|---|
| `easy` | Drop a completely empty column and train. |
| `medium` | Fill missing numeric values and train. |
| `hard` | Clean a messy dataset with mixed types and missing data before training. |

## ⚙️ Actions

```json
{
  "command": "drop_column | fill_mean | encode_category | train_model",
  "target_column": "column_name or null"
}
```

## 📦 Project Structure

```
data_wrangler_env/
├── Inference.py      # ✅ Official inference script (submission entry point)
├── app.py            # FastAPI server (OpenEnv endpoints)
├── env.py            # DataWranglerEnv logic
├── models.py         # Pydantic typed models
├── tasks.py          # Task graders (easy / medium / hard)
├── baseline.py       # AI baseline agent
├── data_loader.py    # CSV/XLSX → state converter
├── openenv.yaml      # OpenEnv spec
├── Dockerfile        # Container config
└── requirements.txt  # Dependencies
```
