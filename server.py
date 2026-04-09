from fastapi import FastAPI, HTTPException
from env import DataCleaningEnv, Action
import os

app = FastAPI(
    title="Data Cleaning Environment",
    description="An OpenEnv environment where AI agents learn to clean messy datasets.",
    version="1.0.0"
)

TASK = os.getenv("TASK", "fix_nulls")
env = DataCleaningEnv(task=TASK)

# Auto-reset on startup so dataset is never None
env.reset()


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
        # Auto-reset if dataset is None
        if env.dataset is None:
            env.reset()

        obs, reward, done, info = env.step(action)

        # Force score strictly between 0 and 1
        safe_score = round(min(max(float(reward.score), 0.01), 0.99), 2)

        return {
            "observation": obs.dict(),
            "reward": {
                "score": safe_score,
                "feedback": reward.feedback
            },
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)