import copy
import pandas as pd
import numpy as np
from models import Observation, Action, Reward
from typing import Tuple, Dict, Any


class DataWranglerEnv:
    """
    Ultimate DataWrangler Environment
    ==================================
    Supports 8 cleaning commands with smart, type-aware logic:
      drop_column     — Remove useless / high-missing columns
      fill_mean       — Fill numeric NaN with mean (INTEGER-AWARE ✨)
      fill_median     — Fill numeric NaN with median (outlier-robust ✨)
      fill_mode       — Fill any column NaN with mode (most frequent value ✨)
      clip_outliers   — IQR-based outlier clipping on numeric columns ✨
      drop_duplicates — Remove exact duplicate rows ✨
      drop_duplicates — Remove exact duplicate rows ✨
      train_model     — Train model (only works on clean data)
    """

    def __init__(self, initial_state: dict, df: pd.DataFrame = None):
        self.initial_state = copy.deepcopy(initial_state)
        self.initial_df = df.copy() if df is not None else None
        self.reset()

    def reset(self) -> Observation:
        self.state_data = copy.deepcopy(self.initial_state)
        self.df = self.initial_df.copy() if self.initial_df is not None else None
        self.step_count = 0
        self.max_steps = 15  # Increased for complex multi-step cleaning
        self.model_trained = False
        return self.state()

    def state(self) -> Observation:
        return Observation(
            total_rows=self.state_data["total_rows"],
            columns=self.state_data["columns"],
            current_accuracy=self.state_data["current_accuracy"],
            step_count=self.step_count
        )

    def _is_clean_for_training(self) -> bool:
        """Check if dataset is fully ready for model training."""
        for col, info in self.state_data["columns"].items():
            if info["missing_values"] > 0:
                return False
        return True

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        reward_val = 0.0
        reason = ""
        cols = self.state_data["columns"]
        target = action.target_column

        # ── train_model ──────────────────────────────────────────────────────────
        if action.command == "train_model":
            if self._is_clean_for_training():
                self.state_data["current_accuracy"] += 0.2
                reward_val = 1.0
                reason = "✅ Model trained successfully on clean data!"
                self.model_trained = True
            else:
                # Tell the agent exactly what's wrong
                problems = []
                for c, info in cols.items():
                    if info["missing_values"] > 0:
                        problems.append(f"'{c}' has {info['missing_values']} missing values")
                    if info["type"] in ("categorical", "datetime"):
                        problems.append(f"'{c}' is still {info['type']} (needs encoding)")
                reward_val = -1.0
                reason = f"❌ Training failed! Issues: {'; '.join(problems)}"

        # ── drop_column ──────────────────────────────────────────────────────────
        elif action.command == "drop_column":
            if target in cols:
                total = self.state_data["total_rows"]
                miss_pct = cols[target]["missing_values"] / max(total, 1)
                if miss_pct >= 0.9:
                    reward_val = 0.8
                    reason = f"✅ Perfect drop: '{target}' was {miss_pct*100:.0f}% empty — garbage column removed!"
                    self.state_data["current_accuracy"] += 0.05
                elif miss_pct >= 0.5:
                    reward_val = 0.2
                    reason = f"⚠️ Risky drop: '{target}' had {miss_pct*100:.0f}% missing. Consider filling instead."
                else:
                    reward_val = -0.5
                    reason = f"❌ Bad drop: '{target}' only had {miss_pct*100:.0f}% missing — lost valuable data!"
                    self.state_data["current_accuracy"] -= 0.1

                del cols[target]
                if self.df is not None and target in self.df.columns:
                    self.df.drop(columns=[target], inplace=True)
            else:
                reward_val = -0.1
                reason = f"❌ Column '{target}' does not exist."

        # ── fill_mean ────────────────────────────────────────────────────────────
        elif action.command == "fill_mean":
            if target in cols and cols[target]["type"] == "numeric":
                if cols[target]["missing_values"] > 0:
                    cols[target]["missing_values"] = 0
                    if self.df is not None and target in self.df.columns:
                        mean_val = self.df[target].mean()
                        original_dtype = cols[target].get("original_dtype", "float")
                        if original_dtype == "int":
                            fill_val = int(round(mean_val))
                            reason = f"✅ Filled '{target}' with mean={fill_val} (integer — no ugly decimals!)"
                            self.df[target] = self.df[target].fillna(fill_val).astype(int)
                        else:
                            fill_val = round(mean_val, 4)
                            reason = f"✅ Filled '{target}' with mean={fill_val:.4f}"
                            self.df[target] = self.df[target].fillna(fill_val)
                    else:
                        reason = f"✅ Filled '{target}' with mean."
                    reward_val = 0.5
                    self.state_data["current_accuracy"] += 0.1
                else:
                    reward_val = -0.1
                    reason = f"⚠️ '{target}' already has no missing values."
            else:
                reward_val = -0.5
                reason = "❌ fill_mean only works on numeric columns."

        # ── fill_median (NEW ✨) ──────────────────────────────────────────────────
        elif action.command == "fill_median":
            if target in cols and cols[target]["type"] == "numeric":
                if cols[target]["missing_values"] > 0:
                    cols[target]["missing_values"] = 0
                    has_outliers = cols[target].get("has_outliers", False)
                    if self.df is not None and target in self.df.columns:
                        median_val = self.df[target].median()
                        original_dtype = cols[target].get("original_dtype", "float")
                        if original_dtype == "int":
                            fill_val = int(round(median_val))
                            reason = f"✅ Filled '{target}' with median={fill_val} (integer-aware!)"
                            self.df[target] = self.df[target].fillna(fill_val).astype(int)
                        else:
                            fill_val = round(median_val, 4)
                            reason = f"✅ Filled '{target}' with median={fill_val:.4f}"
                            self.df[target] = self.df[target].fillna(fill_val)
                    else:
                        reason = f"✅ Filled '{target}' with median."

                    if has_outliers:
                        reason += " 🎯 Bonus: Median is outlier-robust — better choice than mean here!"
                        reward_val = 0.7  # Extra reward for smart choice
                        self.state_data["current_accuracy"] += 0.13
                    else:
                        reward_val = 0.5
                        self.state_data["current_accuracy"] += 0.1
                else:
                    reward_val = -0.1
                    reason = f"⚠️ '{target}' already has no missing values."
            else:
                reward_val = -0.5
                reason = "❌ fill_median only works on numeric columns."

        # ── fill_mode (NEW ✨) ────────────────────────────────────────────────────
        elif action.command == "fill_mode":
            if target in cols:
                if cols[target]["missing_values"] > 0:
                    cols[target]["missing_values"] = 0
                    if self.df is not None and target in self.df.columns:
                        modes = self.df[target].mode()
                        if len(modes) > 0:
                            self.df[target] = self.df[target].fillna(modes[0])
                            reason = f"✅ Filled '{target}' with mode='{modes[0]}' (most frequent value)."
                        else:
                            reason = f"✅ Filled '{target}' missing values with mode."
                    else:
                        reason = f"✅ Filled '{target}' with mode."
                    reward_val = 0.5
                    self.state_data["current_accuracy"] += 0.08
                else:
                    reward_val = -0.1
                    reason = f"⚠️ '{target}' already has no missing values."
            else:
                reward_val = -0.5
                reason = f"❌ Column '{target}' does not exist."

        # ── clip_outliers (NEW ✨) ────────────────────────────────────────────────
        elif action.command == "clip_outliers":
            if target in cols and cols[target]["type"] == "numeric":
                if cols[target].get("has_outliers", False):
                    if self.df is not None and target in self.df.columns:
                        series = self.df[target].dropna()
                        Q1 = series.quantile(0.25)
                        Q3 = series.quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        n_clipped = int(((self.df[target] < lower) | (self.df[target] > upper)).sum())
                        self.df[target] = self.df[target].clip(lower=lower, upper=upper)
                        # Restore integer dtype if needed
                        if cols[target].get("original_dtype") == "int":
                            self.df[target] = (
                                self.df[target].round().astype("Int64")
                                if self.df[target].isnull().any()
                                else self.df[target].round().astype(int)
                            )
                        reason = f"✅ Clipped {n_clipped} outliers in '{target}' using IQR method. Data quality UP!"
                    else:
                        reason = f"✅ Clipped outliers in '{target}'."
                    cols[target]["has_outliers"] = False
                    reward_val = 0.7
                    self.state_data["current_accuracy"] += 0.08
                else:
                    reward_val = -0.1
                    reason = f"ℹ️ No outliers detected in '{target}' — nothing to clip."
            else:
                reward_val = -0.3
                reason = "❌ clip_outliers only works on numeric columns."

        # ── drop_duplicates (NEW ✨) ──────────────────────────────────────────────
        elif action.command == "drop_duplicates":
            if self.df is not None:
                before = len(self.df)
                self.df.drop_duplicates(inplace=True)
                self.df.reset_index(drop=True, inplace=True)
                after = len(self.df)
                removed = before - after
                self.state_data["total_rows"] = after
                if removed > 0:
                    reward_val = 0.4
                    reason = f"✅ Removed {removed} duplicate rows ({before} → {after}). Dataset is unique now!"
                    self.state_data["current_accuracy"] += 0.05
                else:
                    reward_val = 0.0
                    reason = "ℹ️ No duplicate rows found — already clean."
            else:
                reward_val = 0.0
                reason = "ℹ️ No dataframe loaded, skipping duplicate check."


        else:
            reward_val = -0.5
            reason = f"❌ Unknown command: '{action.command}'. Valid: drop_column, fill_mean, fill_median, fill_mode, clip_outliers, drop_duplicates, train_model"

        # Clamp accuracy
        self.state_data["current_accuracy"] = min(1.0, max(0.0, self.state_data["current_accuracy"]))
        done = self.model_trained or self.step_count >= self.max_steps
        info = {"accuracy": self.state_data["current_accuracy"], "success": self.model_trained}

        return self.state(), Reward(value=reward_val, reason=reason), done, info

# Version: NO_ENCODE_V2

