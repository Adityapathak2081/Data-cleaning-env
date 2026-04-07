from fastapi import FastAPI, HTTPException
from env import DataCleaningEnv, Action
import os

# ─────────────────────────────────────────
# CREATE THE APP
# ─────────────────────────────────────────
app = FastAPI(
    title="Data Cleaning Environment",
    description="An OpenEnv environment where AI agents learn to clean messy datasets.",
    version="1.0.0"
)

# Read which task to run from environment variable
# Default is fix_nulls if nothing is set
TASK = os.getenv("TASK", "fix_nulls")

# Create one instance of the environment
env = DataCleaningEnv(task=TASK)


# ─────────────────────────────────────────
# ROUTES (API Endpoints)
# ─────────────────────────────────────────

@app.get("/")
def home():
    """Health check — just to confirm server is running"""
    return {
        "status": "running",
        "environment": "data-cleaning-env",
        "current_task": TASK
    }


@app.post("/reset")
def reset():
    """
    Resets the environment to a fresh start.
    The hackathon validator pings this first.
    Returns the initial observation.
    """
    try:
        obs = env.reset()
        return obs.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: Action):
    """
    Agent submits a cleaned dataset here.
    Returns observation, reward, done, info.
    """
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
    """
    Returns the current state of the environment.
    """
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# RUN THE SERVER
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)