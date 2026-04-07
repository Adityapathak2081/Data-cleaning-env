import os
import json
from openai import OpenAI
from env import DataCleaningEnv, Action

# ─────────────────────────────────────────
# CONFIGURATION
# Read from environment variables
# ─────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN")

# ─────────────────────────────────────────
# SETUP OpenAI CLIENT
# (points to HuggingFace router, not OpenAI)
# ─────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# The 3 tasks we will run
TASKS = ["fix_nulls", "fix_types", "full_clean"]
MAX_STEPS = 1  # one action per episode for now


def run_task(task: str):
    """
    Runs the AI agent on a single task.
    Prints logs in the required [START][STEP][END] format.
    """

    # Step 1: Create environment and reset it
    env = DataCleaningEnv(task=task)
    obs = env.reset()

    # ── MANDATORY LOG FORMAT ──
    print(f"[START] task={task} env=data-cleaning-env model={MODEL_NAME}")

    # Step 2: Build the prompt for the AI
    prompt = f"""You are a data cleaning expert.

You are given a dirty dataset in JSON format.
Your job is to clean it based on the task instructions.

Dataset:
{obs.data}

Task Instructions:
{obs.issues_hint}

Rules:
- Return ONLY a valid JSON object, no extra text, no markdown, no backticks.
- The JSON must have exactly these two fields:
  1. "cleaned_data": the cleaned rows as a JSON array (list of objects)
  2. "steps_taken": a short string explaining what you fixed

Example format:
{{
    "cleaned_data": [{{"name": "Alice", "age": 30, "salary": 50000, "department": "HR"}}],
    "steps_taken": "Filled missing ages with median, removed duplicates"
}}
"""

    # Step 3: Call the AI model
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2   # low temperature = more consistent output
        )

        # Get the raw text response
        raw = response.choices[0].message.content

        # Clean up in case model wraps in markdown backticks
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Parse the JSON
        parsed = json.loads(raw)

        # Convert cleaned_data back to JSON string (env expects string)
        if isinstance(parsed["cleaned_data"], list):
            parsed["cleaned_data"] = json.dumps(parsed["cleaned_data"])

        # Create Action object
        action = Action(
            cleaned_data=parsed["cleaned_data"],
            steps_taken=parsed.get("steps_taken", "No explanation given")
        )

        # Step 4: Submit action to environment
        obs, reward, done, info = env.step(action)

        # ── MANDATORY LOG FORMAT ──
        print(f"[STEP] step=1 action=clean_dataset reward={reward.score:.2f} done={str(done).lower()} error=null")
        print(f"[END] success={'true' if reward.score >= 0.5 else 'false'} steps=1 rewards={reward.score:.2f}")

        # Extra info (not required but helpful to see)
        print(f"       feedback: {reward.feedback}")
        print(f"       steps_taken: {parsed.get('steps_taken', '')}")
        print()

    except json.JSONDecodeError as e:
        print(f"[STEP] step=1 action=clean_dataset reward=0.00 done=true error=JSON parse failed: {e}")
        print(f"[END] success=false steps=1 rewards=0.00")
        print()

    except Exception as e:
        print(f"[STEP] step=1 action=clean_dataset reward=0.00 done=true error={str(e)}")
        print(f"[END] success=false steps=1 rewards=0.00")
        print()


# ─────────────────────────────────────────
# MAIN — Run all 3 tasks
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Data Cleaning Environment — Baseline Inference")
    print("=" * 60)
    print()

    if not API_KEY:
        print("⚠️  WARNING: HF_TOKEN not set!")
        print("   Set it with: $env:HF_TOKEN='your_token_here'")
        print("   Get a free token at: https://huggingface.co/settings/tokens")
        print()
    
    for task in TASKS:
        print(f"─── Running task: {task} ───")
        run_task(task)
    
    print("=" * 60)
    print("  All tasks complete!")
    print("=" * 60)