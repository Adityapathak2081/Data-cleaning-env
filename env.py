from pydantic import BaseModel
from typing import Optional
import pandas as pd
import json
from datasets import get_dataset

# ─────────────────────────────────────────
# TYPED MODELS
# These define what the agent sees, does,
# and gets back as feedback
# ─────────────────────────────────────────

class Observation(BaseModel):
    """What the agent SEES at each step"""
    dataset_name: str
    data: str           # the dataset as a JSON string
    issues_hint: str    # a hint about what's wrong
    task: str           # which task is being run

class Action(BaseModel):
    """What the agent DOES"""
    cleaned_data: str   # the cleaned dataset as JSON string
    steps_taken: str    # explanation of what the agent fixed

class Reward(BaseModel):
    """Feedback the agent gets after acting"""
    score: float        # 0.0 to 1.0
    feedback: str       # human readable explanation of score


# ─────────────────────────────────────────
# THE ENVIRONMENT
# ─────────────────────────────────────────

class DataCleaningEnv:
    """
    A real-world data cleaning environment.
    
    3 Tasks:
      - fix_nulls  (easy)   : fill missing values
      - fix_types  (medium) : fix types + remove duplicates
      - full_clean (hard)   : fix everything
    """

    def __init__(self, task="fix_nulls"):
        self.task = task
        self.dataset = None
        self.done = False
        self.step_count = 0

    def reset(self):
        """
        Resets the environment to a fresh state.
        Returns the first Observation.
        """
        self.dataset = get_dataset("employee_data")
        self.done = False
        self.step_count = 0

        # Different hints for different tasks
        hints = {
            "fix_nulls":  "Some columns have missing (None) values. Fill them appropriately.",
            "fix_types":  "Some columns have wrong data types and there are duplicate rows. Fix them.",
            "full_clean": "Dataset has nulls, type errors, duplicates, outliers, and formatting issues. Fix all of them."
        }

        return Observation(
            dataset_name=self.dataset["name"],
            data=json.dumps(self.dataset["data"]),
            issues_hint=hints[self.task],
            task=self.task
        )

    def state(self):
        """
        Returns the current state of the environment.
        """
        return {
            "task": self.task,
            "dataset": self.dataset["name"] if self.dataset else None,
            "done": self.done,
            "steps": self.step_count
        }

    def step(self, action: Action):
        """
        Agent submits a cleaned dataset.
        We grade it and return (observation, reward, done, info).
        """
        self.step_count += 1

        # Grade the action
        score, feedback = self._grade(action)

        self.done = True  # one action per episode

        obs = Observation(
            dataset_name=self.dataset["name"],
            data=action.cleaned_data,
            issues_hint="Grading complete.",
            task=self.task
        )
        reward = Reward(score=round(score, 2), feedback=feedback)

        return obs, reward, self.done, {"steps": self.step_count}


    def _grade(self, action: Action):
        """
        The grader - scores the agent's cleaned data.
        Returns (score strictly between 0.01 and 0.99, feedback string)
        """

        # Try to parse the cleaned data the agent returned
        try:
            cleaned = pd.DataFrame(json.loads(action.cleaned_data))
        except Exception as e:
            return 0.05, f"Could not parse cleaned_data as JSON ❌ Error: {e}"

        score = 0.0
        feedback = []

        # ── EASY: fix_nulls ──────────────────────────
        # Check if all null/None values are gone
        null_count = cleaned.isnull().sum().sum()
        if null_count == 0:
            score += 0.38
            feedback.append("No nulls remaining ✅ (+0.38)")
        else:
            score += 0.05
            feedback.append(f"{null_count} null(s) still remaining ❌ (+0.05)")

        # ── MEDIUM: fix_types + duplicates ───────────
        if self.task in ("fix_types", "full_clean"):

            # Check age column is all numeric
            try:
                age_numeric = pd.to_numeric(cleaned["age"], errors="coerce")
                bad_types = age_numeric.isnull().sum()
                if bad_types == 0:
                    score += 0.19
                    feedback.append("Age column types correct ✅ (+0.19)")
                else:
                    score += 0.05
                    feedback.append(f"Age column has {bad_types} non-numeric value(s) ❌ (+0.05)")
            except KeyError:
                score += 0.02
                feedback.append("Age column missing ❌ (+0.02)")

            # Check no duplicate rows
            dup_count = cleaned.duplicated().sum()
            if dup_count == 0:
                score += 0.19
                feedback.append("No duplicate rows ✅ (+0.19)")
            else:
                score += 0.05
                feedback.append(f"{dup_count} duplicate row(s) remaining ❌ (+0.05)")

        # ── HARD: outliers + formatting ───────────────
        if self.task == "full_clean":

            # Check no age outliers (valid range: 0-100)
            try:
                age_col = pd.to_numeric(cleaned["age"], errors="coerce")
                outliers = cleaned[(age_col < 0) | (age_col > 100)]
                if len(outliers) == 0:
                    score += 0.09
                    feedback.append("No age outliers ✅ (+0.09)")
                else:
                    score += 0.02
                    feedback.append(f"{len(outliers)} outlier(s) in age column ❌ (+0.02)")
            except KeyError:
                score += 0.02
                feedback.append("Age column missing ❌ (+0.02)")

            # Check department names are Title Case
            try:
                badly_formatted = cleaned[
                    cleaned["department"] != cleaned["department"].str.title()
                ]
                if len(badly_formatted) == 0:
                    score += 0.09
                    feedback.append("Department formatting correct ✅ (+0.09)")
                else:
                    score += 0.02
                    feedback.append(f"{len(badly_formatted)} department(s) not Title Case ❌ (+0.02)")
            except KeyError:
                score += 0.02
                feedback.append("Department column missing ❌ (+0.02)")

        # ── ENSURE SCORE IS STRICTLY BETWEEN 0 and 1 ──
        score = round(min(max(score, 0.01), 0.99), 2)

        return score, " | ".join(feedback)


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    for task in ["fix_nulls", "fix_types", "full_clean"]:
        env = DataCleaningEnv(task=task)
        obs = env.reset()
        print(f"\n📋 Task: {task}")
        print(f"   Hint : {obs.issues_hint}")
        print(f"   State: {env.state()}")
    print("\n✅ env.py loaded successfully!")