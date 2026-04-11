import pandas as pd
import json
from openenv.core.rubrics import Rubric


def safe_score(score: float) -> float:
    """ALWAYS returns a score strictly between 0 and 1"""
    return round(min(max(float(score), 0.01), 0.99), 2)


def parse_data(cleaned_data):
    """Parse cleaned_data safely into a DataFrame"""
    if cleaned_data is None:
        return pd.DataFrame()
    if isinstance(cleaned_data, pd.DataFrame):
        return cleaned_data
    if isinstance(cleaned_data, list):
        return pd.DataFrame(cleaned_data) if cleaned_data else pd.DataFrame()
    if isinstance(cleaned_data, str):
        try:
            parsed = json.loads(cleaned_data)
            if isinstance(parsed, list):
                return pd.DataFrame(parsed) if parsed else pd.DataFrame()
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


class FixNullsGrader(Rubric):
    """
    Easy task grader — checks if all nulls are removed.
    Score range: 0.06 to 0.38 (never 0.0 or 1.0)
    """

    def forward(self, action=None, observation=None) -> float:
        try:
            # Try to get cleaned_data from action
            if action is None:
                return safe_score(0.06)

            # Handle different action types
            if hasattr(action, 'cleaned_data'):
                cleaned = parse_data(action.cleaned_data)
            elif isinstance(action, dict):
                cleaned = parse_data(action.get('cleaned_data'))
            elif isinstance(action, str):
                cleaned = parse_data(action)
            else:
                cleaned = parse_data(action)

            if cleaned.empty:
                return safe_score(0.06)

            null_count = cleaned.isnull().sum().sum()
            score = 0.38 if null_count == 0 else 0.06
            return safe_score(score)

        except Exception:
            return safe_score(0.06)


class FixTypesGrader(Rubric):
    """
    Medium task grader — checks nulls + types + duplicates.
    Score range: 0.18 to 0.74 (never 0.0 or 1.0)
    """

    def forward(self, action=None, observation=None) -> float:
        try:
            if action is None:
                return safe_score(0.18)

            if hasattr(action, 'cleaned_data'):
                cleaned = parse_data(action.cleaned_data)
            elif isinstance(action, dict):
                cleaned = parse_data(action.get('cleaned_data'))
            else:
                cleaned = parse_data(action)

            if cleaned.empty:
                return safe_score(0.18)

            # Check nulls
            null_count = cleaned.isnull().sum().sum()
            null_score = 0.38 if null_count == 0 else 0.06

            # Check age types
            try:
                age_numeric = pd.to_numeric(cleaned["age"], errors="coerce")
                bad_types = age_numeric.isnull().sum()
                type_score = 0.18 if bad_types == 0 else 0.06
            except Exception:
                type_score = 0.03

            # Check duplicates
            try:
                dup_count = cleaned.duplicated().sum()
                dup_score = 0.18 if dup_count == 0 else 0.06
            except Exception:
                dup_score = 0.03

            return safe_score(null_score + type_score + dup_score)

        except Exception:
            return safe_score(0.18)


class FullCleanGrader(Rubric):
    """
    Hard task grader — checks everything.
    Score range: 0.24 to 0.92 (never 0.0 or 1.0)
    """

    def forward(self, action=None, observation=None) -> float:
        try:
            if action is None:
                return safe_score(0.24)

            if hasattr(action, 'cleaned_data'):
                cleaned = parse_data(action.cleaned_data)
            elif isinstance(action, dict):
                cleaned = parse_data(action.get('cleaned_data'))
            else:
                cleaned = parse_data(action)

            if cleaned.empty:
                return safe_score(0.24)

            # Check nulls
            null_count = cleaned.isnull().sum().sum()
            null_score = 0.38 if null_count == 0 else 0.06

            # Check age types
            try:
                age_numeric = pd.to_numeric(cleaned["age"], errors="coerce")
                bad_types = age_numeric.isnull().sum()
                type_score = 0.18 if bad_types == 0 else 0.06
            except Exception:
                type_score = 0.03

            # Check duplicates
            try:
                dup_count = cleaned.duplicated().sum()
                dup_score = 0.18 if dup_count == 0 else 0.06
            except Exception:
                dup_score = 0.03

            # Check outliers
            try:
                age_col = pd.to_numeric(cleaned["age"], errors="coerce")
                outliers = cleaned[(age_col < 0) | (age_col > 100)]
                outlier_score = 0.09 if len(outliers) == 0 else 0.03
            except Exception:
                outlier_score = 0.03

            # Check formatting
            try:
                badly_formatted = cleaned[
                    cleaned["department"] != cleaned["department"].str.title()
                ]
                format_score = 0.09 if len(badly_formatted) == 0 else 0.03
            except Exception:
                format_score = 0.03

            total = null_score + type_score + dup_score + outlier_score + format_score
            return safe_score(total)

        except Exception:
            return safe_score(0.24)


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Testing with no arguments:")
    print(f"FixNullsGrader()(None, None):  {FixNullsGrader()(None, None)}")
    print(f"FixTypesGrader()(None, None):  {FixTypesGrader()(None, None)}")
    print(f"FullCleanGrader()(None, None): {FullCleanGrader()(None, None)}")

    print("\nTesting with empty action:")
    print(f"FixNullsGrader()('', None):  {FixNullsGrader()('', None)}")
    print(f"FixTypesGrader()('', None):  {FixTypesGrader()('', None)}")
    print(f"FullCleanGrader()('', None): {FullCleanGrader()('', None)}")

    print("\nAll scores strictly between 0 and 1 ✅")