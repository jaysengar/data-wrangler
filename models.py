from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class Observation(BaseModel):
    total_rows: int = Field(..., description="Total rows in the dataset")
    columns: Dict[str, Dict[str, Any]] = Field(
        ...,
        description=(
            "Column info dict. Each column has: "
            "type ('numeric'/'categorical'/'datetime'), "
            "original_dtype ('int'/'float'/'categorical'/'datetime'), "
            "missing_values (int), "
            "unique_count (int), "
            "has_outliers (bool)"
        )
    )
    current_accuracy: float = Field(..., description="Current ML model accuracy (0.0 to 1.0)")
    step_count: int = Field(..., description="Steps taken so far")


class Action(BaseModel):
    command: str = Field(
        ...,
        description=(
            "One of: "
            "'drop_column' — remove a useless/high-missing column; "
            "'fill_mean' — fill numeric NaN with mean (integer-aware); "
            "'fill_median' — fill numeric NaN with median (use when has_outliers=True); "
            "'fill_mode' — fill any column NaN with most frequent value; "
            "'clip_outliers' — IQR-based outlier clipping on numeric column; "
            "'drop_duplicates' — remove duplicate rows (target_column=null); "
            "'encode_category' — label-encode a categorical column to numeric; "
            "'train_model' — train model (only when all columns are numeric & 0 missing)"
        )
    )
    target_column: Optional[str] = Field(
        None,
        description="Column to apply the command to. Use null for drop_duplicates and train_model."
    )


class Reward(BaseModel):
    value: float = Field(..., description="Reward value (-1.0 to 1.0)")
    reason: str = Field(..., description="Human-readable explanation of the reward")