"""
train_model.py
--------------
Cleans data, performs EDA, trains multiple regression models,
evaluates them, and persists the best one.

Developer: Sapna Jabeen
Year: 2026
Project: Student Performance Predictor
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/student_performance.csv"
MODEL_PATH  = "models/student_model.pkl"
VISUALS_DIR = Path("visuals")
FEATURES    = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignment_score",
    "quiz_score",
    "previous_gpa",
    "participation_score",
]
TARGET = "final_exam_score"
PALETTE = "#4F46E5"


# ── 1. Data Loading ────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"📥  Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── 2. Data Cleaning ───────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, remove duplicates, and clip outliers.

    Parameters
    ----------
    df : pd.DataFrame  Raw dataset.

    Returns
    -------
    pd.DataFrame  Cleaned dataset.
    """
    original_len = len(df)

    # Drop duplicates
    df = df.drop_duplicates()
    print(f"🧹  Duplicates removed: {original_len - len(df)}")

    # Fill numeric NaNs with column median
    for col in FEATURES:
        n_missing = df[col].isna().sum()
        if n_missing:
            df[col] = df[col].fillna(df[col].median())
            print(f"    └─ {col}: {n_missing} missing values filled with median")

    # Clip extreme outliers to sane academic ranges
    df["study_hours_per_day"]   = df["study_hours_per_day"].clip(0.0, 16.0)
    df["attendance_percentage"] = df["attendance_percentage"].clip(0.0, 100.0)
    df["assignment_score"]      = df["assignment_score"].clip(0.0, 100.0)
    df["quiz_score"]            = df["quiz_score"].clip(0.0, 100.0)
    df["previous_gpa"]          = df["previous_gpa"].clip(0.0, 4.0)
    df["participation_score"]   = df["participation_score"].clip(0.0, 100.0)
    df[TARGET]                  = df[TARGET].clip(0.0, 100.0)

    print(f"✅  Cleaned dataset: {len(df)} rows remaining")
    return df.reset_index(drop=True)


# ── 3. Feature Engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to enrich the model.

    Parameters
    ----------
    df : pd.DataFrame  Cleaned dataset.

    Returns
    -------
    pd.DataFrame  Dataset with additional engineered columns.
    """
    df = df.copy()
    df["academic_engagement"] = (
        df["attendance_percentage"] * 0.4
        + df["participation_score"] * 0.3
        + df["study_hours_per_day"] * 3.0
    ).round(3)

    df["avg_score"] = df[["assignment_score", "quiz_score"]].mean(axis=1).round(3)
    df["gpa_study_interaction"] = (df["previous_gpa"] * df["study_hours_per_day"]).round(3)
    print("⚙️   Feature engineering complete. New features: academic_engagement, avg_score, gpa_study_interaction")
    return df


# ── 4. EDA Visualisations ──────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """Generate and save all EDA charts to the visuals/ directory."""
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = FEATURES + [TARGET]
    sns.set_style("whitegrid")

    # 4a. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(VISUALS_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    # 4b. Feature distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=30, color=PALETTE, alpha=0.8, edgecolor="white")
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
        axes[i].set_xlabel("")
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(VISUALS_DIR / "feature_distributions.png", dpi=120)
    plt.close(fig)

    # 4c. Box plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor=PALETTE, alpha=0.6),
                        medianprops=dict(color="white", linewidth=2))
        axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
    plt.suptitle("Feature Box Plots", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(VISUALS_DIR / "boxplots.png", dpi=120)
    plt.close(fig)

    # 4d. Final score distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[TARGET], bins=40, color=PALETTE, alpha=0.85, edgecolor="white")
    ax.axvline(df[TARGET].mean(), color="#EF4444", linestyle="--", linewidth=1.5,
               label=f"Mean = {df[TARGET].mean():.1f}")
    ax.set_title("Final Exam Score Distribution", fontsize=13)
    ax.set_xlabel("Final Exam Score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(VISUALS_DIR / "score_distribution.png", dpi=120)
    plt.close(fig)

    # 4e. Attendance vs Final Score
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["attendance_percentage"], df[TARGET],
               alpha=0.35, s=18, color=PALETTE)
    m, b = np.polyfit(df["attendance_percentage"], df[TARGET], 1)
    x_line = np.linspace(df["attendance_percentage"].min(),
                         df["attendance_percentage"].max(), 100)
    ax.plot(x_line, m * x_line + b, color="#EF4444", linewidth=2, label="Trend")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Final Exam Score")
    ax.set_title("Attendance vs Final Score", fontsize=13)
    ax.legend()
    plt.tight_layout()
    fig.savefig(VISUALS_DIR / "attendance_vs_score.png", dpi=120)
    plt.close(fig)

    # 4f. Study Hours vs Final Score
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["study_hours_per_day"], df[TARGET],
               alpha=0.35, s=18, color="#10B981")
    m, b = np.polyfit(df["study_hours_per_day"], df[TARGET], 1)
    x_line = np.linspace(df["study_hours_per_day"].min(),
                         df["study_hours_per_day"].max(), 100)
    ax.plot(x_line, m * x_line + b, color="#EF4444", linewidth=2, label="Trend")
    ax.set_xlabel("Study Hours / Day")
    ax.set_ylabel("Final Exam Score")
    ax.set_title("Study Hours vs Final Score", fontsize=13)
    ax.legend()
    plt.tight_layout()
    fig.savefig(VISUALS_DIR / "study_hours_vs_score.png", dpi=120)
    plt.close(fig)

    print(f"📊  EDA charts saved to {VISUALS_DIR}/")


# ── 5. Model Training & Evaluation ────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    extended_features: list,
) -> Tuple[object, pd.DataFrame, StandardScaler]:
    """
    Train three regression models, evaluate them, and return the best one.

    Parameters
    ----------
    df               : pd.DataFrame  Fully preprocessed dataset.
    extended_features: list          Feature column names to use.

    Returns
    -------
    best_model   : fitted sklearn estimator
    results_df   : DataFrame of model comparison metrics
    scaler       : fitted StandardScaler
    """
    X = df[extended_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models: Dict[str, object] = {
        "Linear Regression":       LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results = []
    trained_models: Dict[str, object] = {}

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE":   round(mae, 4),
            "MSE":   round(mse, 4),
            "RMSE":  round(rmse, 4),
            "R² Score": round(r2, 4),
        })
        trained_models[name] = model
        print(f"  {name:30s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False).reset_index(drop=True)

    best_name  = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\n🏆  Best model: {best_name}  (R² = {results_df.iloc[0]['R² Score']})")

    # Feature importance chart (Random Forest)
    rf_model = trained_models["Random Forest Regressor"]
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature":    extended_features,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [PALETTE if i >= len(feat_df) - 3 else "#A5B4FC"
              for i in range(len(feat_df))]
    ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="white")
    ax.set_title("Feature Importance (Random Forest)", fontsize=13)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(VISUALS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)

    return best_model, results_df, scaler, extended_features


# ── 6. Model Persistence ───────────────────────────────────────────────────────

def save_model(model: object, scaler: StandardScaler,
               feature_names: list, path: str = MODEL_PATH) -> None:
    """
    Persist the trained model, scaler, and feature list via joblib.

    Parameters
    ----------
    model         : Fitted sklearn estimator.
    scaler        : Fitted StandardScaler.
    feature_names : List of feature column names.
    path          : Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "scaler": scaler, "features": feature_names}
    joblib.dump(bundle, path)
    print(f"💾  Model bundle saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Full pipeline: load → clean → engineer → EDA → train → save."""
    print("\n" + "=" * 60)
    print("  Student Performance Predictor — Training Pipeline")
    print("  Developer: Sapna Jabeen  |  Year: 2026")
    print("=" * 60 + "\n")

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    run_eda(df)

    extended_features = FEATURES + [
        "academic_engagement",
        "avg_score",
        "gpa_study_interaction",
    ]

    print("\n🤖  Training models …")
    best_model, results_df, scaler, features = train_and_evaluate(df, extended_features)

    print("\n📋  Model Comparison:")
    print(results_df.to_string(index=False))

    save_model(best_model, scaler, features)
    print("\n✅  Pipeline complete.\n")


if __name__ == "__main__":
    main()
