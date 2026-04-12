import json
import pandas as pd
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.rubrics import Rubric
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..models import DataCleaningAction, DataCleaningObservation
except ImportError:
    from models import DataCleaningAction, DataCleaningObservation

DIRTY_DATA = [
    {"name": "Alice",  "age": 30,    "salary": 50000, "department": "HR"},
    {"name": "Bob",    "age": None,  "salary": 60000, "department": "IT"},
    {"name": "Carol",  "age": 25,    "salary": None,  "department": "hr"},
    {"name": "Bob",    "age": None,  "salary": 60000, "department": "IT"},
    {"name": "Dave",   "age": 999,   "salary": 55000, "department": "Finance"},
    {"name": "",       "age": 28,    "salary": 52000, "department": "IT"},
    {"name": "Eve",    "age": "abc", "salary": 48000, "department": "HR"},
]

TASKS = ["fix_nulls", "fix_types", "full_clean"]

HINTS = {
    "fix_nulls":  "Some columns have missing (None) values. Fill them appropriately.",
    "fix_types":  "Some columns have wrong data types and there are duplicate rows. Fix them.",
    "full_clean": "Dataset has nulls, type errors, duplicates, outliers, and formatting issues. Fix all of them."
}


def safe_score(score: float) -> float:
    return round(min(max(float(score), 0.01), 0.99), 2)


def parse_data(data):
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list):
        return pd.DataFrame(data) if data else pd.DataFrame()
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return pd.DataFrame(parsed) if parsed else pd.DataFrame()
        except Exception:
            pass
    return pd.DataFrame()


def compute_grade(cleaned_data, task: str) -> float:
    cleaned = parse_data(cleaned_data)

    if cleaned.empty:
        if task == "fix_nulls":
            return safe_score(0.06)
        elif task == "fix_types":
            return safe_score(0.18)
        else:
            return safe_score(0.24)

    null_score = 0.38 if cleaned.isnull().sum().sum() == 0 else 0.06
    type_score = dup_score = outlier_score = format_score = 0.0

    if task in ("fix_types", "full_clean"):
        try:
            type_score = 0.18 if pd.to_numeric(cleaned["age"], errors="coerce").isnull().sum() == 0 else 0.06
        except Exception:
            type_score = 0.03
        try:
            dup_score = 0.18 if cleaned.duplicated().sum() == 0 else 0.06
        except Exception:
            dup_score = 0.03

    if task == "full_clean":
        try:
            age_col = pd.to_numeric(cleaned["age"], errors="coerce")
            outlier_score = 0.09 if len(cleaned[(age_col < 0) | (age_col > 100)]) == 0 else 0.03
        except Exception:
            outlier_score = 0.03
        try:
            format_score = 0.09 if len(cleaned[cleaned["department"] != cleaned["department"].str.title()]) == 0 else 0.03
        except Exception:
            format_score = 0.03

    if task == "fix_nulls":
        total = null_score
    elif task == "fix_types":
        total = null_score + type_score + dup_score
    else:
        total = null_score + type_score + dup_score + outlier_score + format_score

    return safe_score(total)


# ─────────────────────────────────────────
# RUBRICS — attached to environment
# ─────────────────────────────────────────

class DataCleaningRubric(Rubric):
    """Main rubric that grades based on current task"""

    def __init__(self, get_task_fn):
        super().__init__()
        self._get_task = get_task_fn

    def forward(self, action=None, observation=None) -> float:
        try:
            task = self._get_task()
            if action is None:
                return safe_score(0.06)
            if hasattr(action, 'cleaned_data'):
                return compute_grade(action.cleaned_data, task)
            return safe_score(0.06)
        except Exception:
            return safe_score(0.06)


# ─────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────

class DataCleaningEnvironment(Environment):
    """Data Cleaning Environment — AI agents learn to clean messy datasets."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._current_task = "fix_nulls"
        self._task_index = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # Attach rubric
        rubric = DataCleaningRubric(get_task_fn=lambda: self._current_task)
        super().__init__(rubric=rubric)

    def reset(self, seed=None, episode_id=None, **kwargs) -> DataCleaningObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = TASKS[self._task_index % len(TASKS)]
        self._reset_rubric()

        return DataCleaningObservation(
            dataset_name="employee_data",
            data=json.dumps(DIRTY_DATA),
            issues_hint=HINTS[self._current_task],
            task=self._current_task,
            done=False,
            reward=safe_score(0.06),
        )

    def step(self, action: DataCleaningAction, timeout_s=None, **kwargs) -> DataCleaningObservation:
        self._state.step_count += 1
        task = self._current_task

        # Use rubric to compute score
        score = self._apply_rubric(action, None)
        if score == 0.0:
            score = compute_grade(action.cleaned_data, task)
        score = safe_score(score)

        self._task_index += 1

        return DataCleaningObservation(
            dataset_name="employee_data",
            data=action.cleaned_data,
            issues_hint="Grading complete.",
            task=task,
            done=True,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._state