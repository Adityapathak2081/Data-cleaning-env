from pydantic import BaseModel
from typing import Optional
import pandas as pd
import json
from datasets import get_dataset

# ─────────────────────────────────────────
# TYPED MODELS
# ─────────────────────────────────────────

class Observation(BaseModel):
    """What the agent SEES at each step"""
    dataset_name: str
    data: str
    issues_hint: str
    task: str

class Action(BaseModel):
    """What the agent DOES"""
    cleaned_data: str
    steps_taken: str

class Reward(BaseModel):
    """Feedback the agent gets after acting"""
    score: float
    feedback: str


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
        """Resets the environment to a fresh state."""
        self.dataset = get_dataset("employee_data")
        self.done = False
        self.step_count = 0

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
        """Returns the current state of the environment."""
        return {
            "task": self.task,
            "dataset": self.dataset["name"] if self.dataset else None,
            "done": self.done,
            "steps": self.step_count
        }

    def step(self, action: Action):
        """Agent submits a cleaned dataset."""
        self.step_count += 1
        score, feedback = self._grade(action)
        self.done = True

        # Force strictly between 0 and 1
        score = float(score)
        score = round(min(max(score, 0.01), 0.99), 2)

        obs = Observation(
            dataset_name=self.dataset["name"],
            data=action.cleaned_data,
            issues_hint="Grading complete.",
            task=self.task
        )
        reward = Reward(score=score, feedback=feedback)

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

        feedback = []

        # ── EASY: fix_nulls ──────────────────────────
        null_count = cleaned.isnull().sum().sum()
        if null_count == 0:
            null_score = 0.38
            feedback.append("No nulls remaining ✅")
        else:
            null_score = 0.06
            feedback.append(f"{null_count} null(s) still remaining ❌")

        # ── MEDIUM: fix_types + duplicates ───────────
        type_score = 0.0
        dup_score = 0.0

        if self.task in ("fix_types", "full_clean"):
            try:
                age_numeric = pd.to_numeric(cleaned["age"], errors="coerce")
                bad_types = age_numeric.isnull().sum()
                if bad_types == 0:
                    type_score = 0.18
                    feedback.append("Age column types correct ✅")
                else:
                    type_score = 0.06
                    feedback.append(f"Age has {bad_types} non-numeric value(s) ❌")
            except KeyError:
                type_score = 0.03
                feedback.append("Age column missing ❌")

            dup_count = cleaned.duplicated().sum()
            if dup_count == 0:
                dup_score = 0.18
                feedback.append("No duplicate rows ✅")
            else:
                dup_score = 0.06
                feedback.append(f"{dup_count} duplicate(s) remaining ❌")

        # ── HARD: outliers + formatting ───────────────
        outlier_score = 0.0
        format_score = 0.0

        if self.task == "full_clean":
            try:
                age_col = pd.to_numeric(cleaned["age"], errors="coerce")
                outliers = cleaned[(age_col < 0) | (age_col > 100)]
                if len(outliers) == 0:
                    outlier_score = 0.09
                    feedback.append("No age outliers ✅")
                else:
                    outlier_score = 0.03
                    feedback.append(f"{len(outliers)} outlier(s) in age ❌")
            except KeyError:
                outlier_score = 0.03
                feedback.append("Age column missing ❌")

            try:
                badly_formatted = cleaned[
                    cleaned["department"] != cleaned["department"].str.title()
                ]
                if len(badly_formatted) == 0:
                    format_score = 0.09
                    feedback.append("Department formatting correct ✅")
                else:
                    format_score = 0.03
                    feedback.append(f"{len(badly_formatted)} department(s) wrong format ❌")
            except KeyError:
                format_score = 0.03
                feedback.append("Department column missing ❌")

        # ── CALCULATE FINAL SCORE BY TASK ──
        if self.task == "fix_nulls":
            score = null_score

        elif self.task == "fix_types":
            score = null_score + type_score + dup_score

        elif self.task == "full_clean":
            score = null_score + type_score + dup_score + outlier_score + format_score

        else:
            score = 0.05

        # ── HARD CLAMP ──
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