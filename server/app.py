from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys

# Add parent directory to path so we can import env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DataCleaningEnv, Action

# ─────────────────────────────────────────
# CREATE THE APP
# ─────────────────────────────────────────
app = FastAPI(
    title="Data Cleaning Environment",
    description="An OpenEnv environment where AI agents learn to clean messy datasets.",
    version="1.0.0"
)

TASK = os.getenv("TASK", "fix_nulls")
env = DataCleaningEnv(task=TASK)


@app.get("/")
def home():
    return {
        "status": "running",
        "environment": "data-cleaning-env",
        "current_task": TASK
    }


@app.post("/reset")
def reset():
    try:
        obs = env.reset()
        return obs.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# MAIN FUNCTION — required by OpenEnv spec
# ─────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )


if __name__ == "__main__":
    main()