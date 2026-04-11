from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional

class DataCleaningAction(Action):
    """Action for the Data Cleaning environment."""
    cleaned_data: str = Field(..., description="Cleaned dataset as JSON array string")
    steps_taken: str = Field(default="", description="Explanation of cleaning steps")

class DataCleaningObservation(Observation):
    """Observation from the Data Cleaning environment."""
    dataset_name: str = Field(default="", description="Name of the dataset")
    data: str = Field(default="", description="The dirty dataset as JSON string")
    issues_hint: str = Field(default="", description="Hint about what needs fixing")
    task: str = Field(default="fix_nulls", description="Current task name")