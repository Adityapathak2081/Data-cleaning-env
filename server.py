from fastapi import FastAPI, HTTPException
from env import DataCleaningEnv, Action, Observation
import os

app = FastAPI(
    title="Data Cleaning Environment",
    description="An OpenEnv environment where AI agents learn to clean messy datasets.",
    version="1.0.0"
)

# Create all 3 task environments
envs = {
    "fix_nulls":  DataCleaningEnv(task="fix_nulls"),
    "fix_types":  DataCleaningEnv(task="fix_types"),
    "full_clean": DataCleaningEnv(task="full_clean")
}

# Auto-reset all on startup
for e in envs.values():
    e.reset()

TASK = os.getenv("TASK", "fix_nulls")
current_task = TASK


# ─────────────────────────────────────────
# STANDARD OPENENV ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health")
def health():
    """Required by OpenEnv validator"""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    """Required by OpenEnv validator"""
    return {
        "name": "data-cleaning-env",
        "description": "A real-world data cleaning environment where AI agents learn to clean messy datasets.",
        "version": "1.0.0",
        "tasks": ["fix_nulls", "fix_types", "full_clean"]
    }


@app.get("/schema")
def schema():
    """Required by OpenEnv validator"""
    return {
        "action": {
            "cleaned_data": {"type": "string", "description": "Cleaned dataset as JSON array"},
            "steps_taken": {"type": "string", "description": "Explanation of cleaning steps"}
        },
        "observation": {
            "dataset_name": {"type": "string"},
            "data": {"type": "string", "description": "Dirty dataset as JSON array"},
            "issues_hint": {"type": "string"},
            "task": {"type": "string"}
        },
        "state": {
            "task": {"type": "string"},
            "dataset": {"type": "string"},
            "done": {"type": "boolean"},
            "steps": {"type": "integer"}
        }
    }


@app.post("/mcp")
def mcp(payload: dict = {}):
    """Required by OpenEnv validator - JSON-RPC endpoint"""
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id", 1),
        "result": {
            "name": "data-cleaning-env",
            "version": "1.0.0"
        }
    }


# ─────────────────────────────────────────
# CORE ENVIRONMENT ENDPOINTS
# ─────────────────────────────────────────

@app.get("/")
def home():
    return {
        "status": "running",
        "environment": "data-cleaning-env",
        "tasks": list(envs.keys()),
        "current_task": current_task
    }


@app.post("/reset")
def reset(task: str = None):
    global current_task
    if task and task in envs:
        current_task = task
    elif task and task not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")
    env = envs[current_task]
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: Action, task: str = None):
    global current_task
    if task and task in envs:
        current_task = task
    env = envs[current_task]
    if env.dataset is None:
        env.reset()
    obs, reward, done, info = env.step(action)
    safe_score = round(min(max(float(reward.score), 0.01), 0.99), 2)
    return {
        "observation": obs.dict(),
        "reward": {"score": safe_score, "feedback": reward.feedback},
        "done": done,
        "info": info
    }


@app.get("/state")
def state(task: str = None):
    global current_task
    if task and task in envs:
        current_task = task
    return envs[current_task].state()


# ─────────────────────────────────────────
# TASK SPECIFIC ENDPOINTS
# ─────────────────────────────────────────

@app.post("/reset/{task_name}")
def reset_task(task_name: str):
    if task_name not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_name}")
    obs = envs[task_name].reset()
    return obs.dict()


@app.post("/step/{task_name}")
def step_task(task_name: str, action: Action):
    if task_name not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_name}")
    env = envs[task_name]
    if env.dataset is None:
        env.reset()
    obs, reward, done, info = env.step(action)
    safe_score = round(min(max(float(reward.score), 0.01), 0.99), 2)
    return {
        "observation": obs.dict(),
        "reward": {"score": safe_score, "feedback": reward.feedback},
        "done": done,
        "info": info
    }


@app.get("/state/{task_name}")
def state_task(task_name: str):
    if task_name not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_name}")
    return envs[task_name].state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"name": "fix_nulls",  "difficulty": "easy"},
            {"name": "fix_types",  "difficulty": "medium"},
            {"name": "full_clean", "difficulty": "hard"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)