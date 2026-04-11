import pandas as pd
import json


def safe_score(score: float) -> float:
    """ALWAYS returns a score strictly between 0 and 1"""
    return round(min(max(float(score), 0.01), 0.99), 2)


class BaseGrader:
    """Base grader class — always returns safe score"""

    def __call__(self, cleaned_data=None, steps_taken=None, **kwargs):
        try:
            return self.grade(cleaned_data, steps_taken)
        except Exception:
            return 0.05

    def grade(self, cleaned_data=None, steps_taken=None):
        return 0.05

    def parse(self, cleaned_data):
        """Parse cleaned_data safely"""
        if cleaned_data is None:
            return pd.DataFrame()
        if isinstance(cleaned_data, list):
            return pd.DataFrame(cleaned_data)
        if isinstance(cleaned_data, str):
            try:
                return pd.DataFrame(json.loads(cleaned_data))
            except Exception:
                return pd.DataFrame()
        if isinstance(cleaned_data, pd.DataFrame):
            return cleaned_data
        return pd.DataFrame()


class FixNullsGrader(BaseGrader):
    """
    Easy task grader — checks if all nulls are removed.
    Score range: 0.06 to 0.38 (never 0.0 or 1.0)
    """

    def grade(self, cleaned_data=None, steps_taken=None):
        cleaned = self.parse(cleaned_data)

        if cleaned.empty:
            return safe_score(0.06)

        null_count = cleaned.isnull().sum().sum()
        if null_count == 0:
            score = 0.38
        else:
            score = 0.06

        return safe_score(score)


class FixTypesGrader(BaseGrader):
    """
    Medium task grader — checks nulls + types + duplicates.
    Score range: 0.18 to 0.74 (never 0.0 or 1.0)
    """

    def grade(self, cleaned_data=None, steps_taken=None):
        cleaned = self.parse(cleaned_data)

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


class FullCleanGrader(BaseGrader):
    """
    Hard task grader — checks everything.
    Score range: 0.24 to 0.92 (never 0.0 or 1.0)
    """

    def grade(self, cleaned_data=None, steps_taken=None):
        cleaned = self.parse(cleaned_data)

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


# ─────────────────────────────────────────
# TEST — call graders with no arguments
# (simulates what the validator does)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Testing graders with no arguments:")
    print(f"FixNullsGrader(): {FixNullsGrader()()}")
    print(f"FixTypesGrader(): {FixTypesGrader()()}")
    print(f"FullCleanGrader(): {FullCleanGrader()()}")

    print("\nTesting graders with empty data:")
    print(f"FixNullsGrader()(cleaned_data=[]): {FixNullsGrader()(cleaned_data=[])}")
    print(f"FixTypesGrader()(cleaned_data=[]): {FixTypesGrader()(cleaned_data=[])}")
    print(f"FullCleanGrader()(cleaned_data=[]): {FullCleanGrader()(cleaned_data=[])}")

    print("\nAll scores strictly between 0 and 1 ✅")