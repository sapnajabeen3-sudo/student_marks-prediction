"""
generate_dataset.py
-------------------
Generates a realistic synthetic student performance dataset.

Developer: Sapna Jabeen
Year: 2026
Project: Student Performance Predictor
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_student_data(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset for student performance prediction.

    Parameters
    ----------
    n_samples : int
        Number of student records to generate (default: 1200).
    random_state : int
        Seed for reproducibility (default: 42).

    Returns
    -------
    pd.DataFrame
        DataFrame containing student features and target variable.
    """
    rng = np.random.default_rng(random_state)

    # ── Core features ──────────────────────────────────────────────────────────
    study_hours = np.clip(rng.normal(loc=5.0, scale=2.0, size=n_samples), 0.5, 12.0)
    attendance = np.clip(rng.normal(loc=78.0, scale=12.0, size=n_samples), 30.0, 100.0)
    assignment_score = np.clip(rng.normal(loc=72.0, scale=13.0, size=n_samples), 20.0, 100.0)
    quiz_score = np.clip(rng.normal(loc=70.0, scale=14.0, size=n_samples), 15.0, 100.0)
    previous_gpa = np.clip(rng.normal(loc=2.8, scale=0.6, size=n_samples), 0.5, 4.0)
    participation_score = np.clip(rng.normal(loc=65.0, scale=15.0, size=n_samples), 10.0, 100.0)

    # ── Realistic target variable (Final Exam Score) ────────────────────────
    # Weighted combination that mirrors real academic relationships
    base_score = (
        study_hours        * 2.5
        + attendance       * 0.30
        + assignment_score * 0.25
        + quiz_score       * 0.20
        + previous_gpa     * 6.0
        + participation_score * 0.10
    )

    # Normalise to a 0–100 range, then add noise
    min_b, max_b = base_score.min(), base_score.max()
    base_score_norm = (base_score - min_b) / (max_b - min_b) * 70 + 20
    noise = rng.normal(loc=0, scale=5.0, size=n_samples)
    final_exam_score = np.clip(base_score_norm + noise, 0.0, 100.0).round(2)

    # ── Assemble DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "study_hours_per_day":    study_hours.round(2),
        "attendance_percentage":  attendance.round(2),
        "assignment_score":       assignment_score.round(2),
        "quiz_score":             quiz_score.round(2),
        "previous_gpa":           previous_gpa.round(2),
        "participation_score":    participation_score.round(2),
        "final_exam_score":       final_exam_score,
    })

    # ── Inject a small amount of realistic missingness / noise ─────────────
    for col in ["assignment_score", "quiz_score", "participation_score"]:
        mask = rng.random(n_samples) < 0.015          # ~1.5 % missing
        df.loc[mask, col] = np.nan

    # A handful of realistic duplicates
    dup_idx = rng.choice(df.index, size=10, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


def save_dataset(df: pd.DataFrame, output_path: str = "data/student_performance.csv") -> None:
    """
    Save the generated DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to persist.
    output_path : str
        Relative or absolute file path for the CSV.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅  Dataset saved → {output_path}  ({len(df)} rows, {df.shape[1]} columns)")


def main() -> None:
    """Entry point: generate and save the student performance dataset."""
    print("📊  Generating student performance dataset …")
    df = generate_student_data(n_samples=1200)
    save_dataset(df)
    print(df.describe().round(2))


if __name__ == "__main__":
    main()
