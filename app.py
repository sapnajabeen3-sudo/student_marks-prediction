"""
app.py
------
Streamlit dashboard for the Student Performance Predictor.

Developer: Sapna Jabeen
Year: 2026
Project: Student Performance Predictor
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

    /* ── Reset & Base ── */
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .block-container { padding: 1.5rem 2rem 2rem; }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #4338ca 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        width: 320px; height: 320px;
        background: rgba(99,102,241,0.2);
        border-radius: 50%;
        top: -80px; right: -80px;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.4rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c7d2fe;
        margin: 0 0 1.2rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        color: #e0e7ff;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }

    /* ── Section Headers ── */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e1b4b;
        margin: 1.8rem 0 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #4F46E5, transparent);
        border-radius: 2px;
        margin: 0 0 1.5rem;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(79,70,229,0.12);
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #4F46E5;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Prediction Result Card ── */
    .result-card {
        background: linear-gradient(135deg, #312e81, #4F46E5);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 8px 30px rgba(79,70,229,0.35);
    }
    .result-score {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -2px;
    }
    .result-grade {
        font-size: 1.6rem;
        font-weight: 600;
        color: #c7d2fe;
        margin-top: 0.3rem;
    }
    .result-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.8rem;
    }

    /* ── Recommendation Cards ── */
    .rec-card {
        background: #f8fafc;
        border-left: 4px solid #4F46E5;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.88rem;
        color: #374151;
        line-height: 1.55;
    }
    .rec-card.success { border-left-color: #10B981; }
    .rec-card.warning { border-left-color: #F59E0B; }
    .rec-card.danger  { border-left-color: #EF4444; }

    /* ── Feature row ── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 0.5rem;
        font-size: 0.78rem;
        color: #9ca3af;
    }
    .footer a { color: #4F46E5; text-decoration: none; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
    section[data-testid="stSidebar"] .stSlider > div > div > div { background: #4F46E5 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/student_model.pkl"
BASE_FEATURES = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignment_score",
    "quiz_score",
    "previous_gpa",
    "participation_score",
]


# ── Utility Functions ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict | None:
    """Load the persisted model bundle from disk (cached)."""
    path = Path(MODEL_PATH)
    if not path.exists():
        return None
    return joblib.load(path)


def score_to_grade(score: float) -> tuple[str, str]:
    """Convert a numeric score to a letter grade and CSS colour class."""
    if score >= 90:
        return "A", "success"
    elif score >= 80:
        return "B", "success"
    elif score >= 70:
        return "C", "warning"
    elif score >= 60:
        return "D", "warning"
    else:
        return "F", "danger"


def engineer_input(raw: dict) -> pd.DataFrame:
    """
    Apply the same feature engineering used during training to user inputs.

    Parameters
    ----------
    raw : dict  Raw values keyed by BASE_FEATURES names.

    Returns
    -------
    pd.DataFrame  Single-row frame with all model features.
    """
    df = pd.DataFrame([raw])
    df["academic_engagement"] = (
        df["attendance_percentage"] * 0.4
        + df["participation_score"] * 0.3
        + df["study_hours_per_day"] * 3.0
    )
    df["avg_score"] = df[["assignment_score", "quiz_score"]].mean(axis=1)
    df["gpa_study_interaction"] = df["previous_gpa"] * df["study_hours_per_day"]
    return df


def build_recommendations(inputs: dict, score: float) -> list[tuple[str, str]]:
    """
    Generate personalised, actionable recommendations.

    Returns
    -------
    List of (message, css_class) tuples.
    """
    recs: list[tuple[str, str]] = []

    if inputs["study_hours_per_day"] < 3:
        recs.append(("📚 You're studying fewer than 3 hours daily. Aim for 4–6 focused hours "
                     "to meaningfully boost your final score.", "warning"))
    elif inputs["study_hours_per_day"] >= 6:
        recs.append(("🌟 Excellent study discipline! Your dedicated hours are a strong "
                     "predictor of success — keep the momentum.", "success"))

    if inputs["attendance_percentage"] < 70:
        recs.append(("⚠️ Attendance below 70% is strongly correlated with lower exam scores. "
                     "Prioritise attending all upcoming classes.", "danger"))
    elif inputs["attendance_percentage"] >= 90:
        recs.append(("✅ Outstanding attendance! Consistent presence in class reinforces "
                     "learning and signals commitment.", "success"))

    if inputs["assignment_score"] < 60:
        recs.append(("📝 Assignment scores below 60 suggest gaps in understanding. "
                     "Review feedback from your instructor and revisit key concepts.", "warning"))

    if inputs["quiz_score"] < 60:
        recs.append(("🧠 Quiz performance indicates recall challenges. "
                     "Try spaced-repetition techniques and practice past papers.", "warning"))

    if inputs["previous_gpa"] < 2.0:
        recs.append(("📈 A GPA under 2.0 suggests past academic challenges. "
                     "Talk to an academic advisor to build a structured recovery plan.", "danger"))
    elif inputs["previous_gpa"] >= 3.5:
        recs.append(("🏆 Strong prior GPA! Build on this foundation by deepening "
                     "understanding rather than just memorising.", "success"))

    if score >= 85:
        recs.append(("🎉 Great predicted outcome! Stay consistent in the final stretch "
                     "and you're on track for an excellent grade.", "success"))
    elif score < 60:
        recs.append(("🆘 This prediction signals risk. Act now: form a study group, "
                     "attend office hours, and reduce non-academic distractions.", "danger"))

    return recs if recs else [
        ("You're performing in a solid range. Keep your current habits and "
         "aim to slightly increase study hours and quiz preparation.", "success")
    ]


def gauge_chart(score: float) -> go.Figure:
    """Return a Plotly gauge chart for the predicted score."""
    grade, _ = score_to_grade(score)
    color = "#10B981" if score >= 80 else "#F59E0B" if score >= 60 else "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 30, "color": "#1e1b4b",
                                              "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  60], "color": "#fee2e2"},
                {"range": [60, 70], "color": "#fef3c7"},
                {"range": [70, 80], "color": "#dbeafe"},
                {"range": [80, 90], "color": "#d1fae5"},
                {"range": [90, 100], "color": "#a7f3d0"},
            ],
            "threshold": {
                "line": {"color": "#1e1b4b", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
        title={"text": f"Grade: <b>{grade}</b>",
               "font": {"size": 18, "color": "#4F46E5", "family": "Space Grotesk"}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="white",
        font={"family": "Inter"},
    )
    return fig


def feature_importance_chart(bundle: dict) -> go.Figure:
    """Return a horizontal bar chart of feature importances."""
    model    = bundle["model"]
    features = bundle["features"]

    friendly = {
        "study_hours_per_day":    "Study Hours / Day",
        "attendance_percentage":  "Attendance %",
        "assignment_score":       "Assignment Score",
        "quiz_score":             "Quiz Score",
        "previous_gpa":           "Previous GPA",
        "participation_score":    "Participation Score",
        "academic_engagement":    "Academic Engagement",
        "avg_score":              "Avg Score",
        "gpa_study_interaction":  "GPA × Study",
    }

    try:
        importances = model.feature_importances_
    except AttributeError:
        # Linear regression — use abs(coef)
        importances = np.abs(model.coef_)

    pairs = sorted(zip(features, importances), key=lambda x: x[1])
    labels = [friendly.get(f, f) for f, _ in pairs]
    values = [v for _, v in pairs]
    colors = ["#4F46E5" if v == max(values) else "#A5B4FC" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        height=360,
        margin=dict(l=0, r=30, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter"},
        xaxis={"showgrid": True, "gridcolor": "#f3f4f6"},
    )
    return fig


def score_comparison_chart(predicted: float) -> go.Figure:
    """Bar chart comparing student's predicted score to academic benchmarks."""
    categories = ["Fail Threshold", "Pass Threshold", "Your Score", "Good Score", "Excellence"]
    values     = [0, 60, predicted, 80, 95]
    colors     = ["#EF4444", "#F59E0B", "#4F46E5", "#10B981", "#059669"]

    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Score Benchmarks",
        yaxis={"range": [0, 110], "title": "Score"},
        height=300,
        margin=dict(l=0, r=0, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter"},
        yaxis_gridcolor="#f3f4f6",
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar inputs and return collected values."""
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 1rem 0 0.5rem;'>
                <div style='font-size:2.8rem;'>🎓</div>
                <div style='font-family:Space Grotesk; font-size:1.1rem; font-weight:700; color:#e0e7ff;'>
                    Student Performance<br>Predictor
                </div>
                <div style='font-size:0.72rem; color:#a5b4fc; margin-top:0.3rem;'>by Sapna Jabeen · 2026</div>
            </div>
            <hr style='border-color:rgba(255,255,255,0.15); margin:1rem 0;'>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📋 Student Inputs")

        study_hours = st.slider(
            "📖 Study Hours Per Day",
            min_value=0.0, max_value=12.0, value=5.0, step=0.5,
            help="Average daily study hours (0 – 12)",
        )
        attendance = st.slider(
            "🏫 Attendance (%)",
            min_value=0.0, max_value=100.0, value=80.0, step=1.0,
        )
        assignment = st.slider(
            "📝 Assignment Score",
            min_value=0.0, max_value=100.0, value=72.0, step=1.0,
        )
        quiz = st.slider(
            "🧠 Quiz Score",
            min_value=0.0, max_value=100.0, value=70.0, step=1.0,
        )
        prev_gpa = st.slider(
            "📊 Previous GPA",
            min_value=0.0, max_value=4.0, value=2.8, step=0.1,
        )
        participation = st.slider(
            "🙋 Participation Score",
            min_value=0.0, max_value=100.0, value=65.0, step=1.0,
        )

        predict_btn = st.button("🔮  Predict Performance", use_container_width=True, type="primary")

        st.markdown(
            """
            <hr style='border-color:rgba(255,255,255,0.15); margin:1.5rem 0 1rem;'>
            <div style='font-size:0.72rem; color:#a5b4fc; text-align:center; line-height:1.8;'>
                <b>Developer:</b> Sapna Jabeen<br>
                <b>Year:</b> 2026<br>
                <b>Model:</b> Random Forest<br>
                <b>Dataset:</b> 1 200 records
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "study_hours_per_day":   study_hours,
        "attendance_percentage": attendance,
        "assignment_score":      assignment,
        "quiz_score":            quiz,
        "previous_gpa":          prev_gpa,
        "participation_score":   participation,
        "_predict":              predict_btn,
    }


# ── Main App ───────────────────────────────────────────────────────────────────

def main() -> None:
    """Render the full Streamlit dashboard."""
    inputs = render_sidebar()
    bundle = load_model_bundle()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🎓 Student Performance Predictor</div>
            <div class="hero-subtitle">
                Machine Learning–powered academic outcome forecasting —<br>
                enter student metrics to predict the Final Exam Score.
            </div>
            <span class="hero-badge">Random Forest</span>
            <span class="hero-badge">R² ≈ 0.96</span>
            <span class="hero-badge">Streamlit · Plotly · scikit-learn</span>
            <span class="hero-badge">Sapna Jabeen · 2026</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bundle is None:
        st.error(
            "⚠️  Model not found. Run `python train_model.py` first to generate "
            "`models/student_model.pkl`."
        )
        st.stop()

    # ── Student Input Summary ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Student Input Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (c1, f"{inputs['study_hours_per_day']:.1f} h", "Study / Day"),
        (c2, f"{inputs['attendance_percentage']:.0f}%",  "Attendance"),
        (c3, f"{inputs['assignment_score']:.0f}",        "Assignment"),
        (c4, f"{inputs['quiz_score']:.0f}",              "Quiz Score"),
        (c5, f"{inputs['previous_gpa']:.1f}",            "Prev. GPA"),
        (c6, f"{inputs['participation_score']:.0f}",     "Participation"),
    ]
    for col, val, lbl in cards:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Prediction ────────────────────────────────────────────────────────────
    if inputs["_predict"] or True:   # always predict for UX smoothness
        raw = {k: v for k, v in inputs.items() if k != "_predict"}
        X   = engineer_input(raw)[bundle["features"]]
        X_sc   = bundle["scaler"].transform(X)
        score  = float(np.clip(bundle["model"].predict(X_sc)[0], 0, 100))
        grade, css_cls = score_to_grade(score)

        st.markdown("")
        st.markdown('<div class="section-header">🔮 Prediction Result</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        left, mid, right = st.columns([1, 1.2, 1])

        with left:
            st.plotly_chart(gauge_chart(score), use_container_width=True, config={"displayModeBar": False})

        with mid:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">PREDICTED FINAL SCORE</div>
                    <div class="result-score">{score:.1f}</div>
                    <div class="result-grade">Grade: {grade}</div>
                    <hr style='border-color:rgba(255,255,255,0.2); margin:1rem 0;'>
                    <div style='font-size:0.82rem; color:rgba(255,255,255,0.7);'>
                        {'🏆 Excellent performance' if score >= 90 else
                         '✅ Good performance'     if score >= 80 else
                         '📈 Average — room to grow' if score >= 70 else
                         '⚠️ Below average'         if score >= 60 else
                         '🆘 High risk — needs intervention'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.plotly_chart(score_comparison_chart(score),
                            use_container_width=True,
                            config={"displayModeBar": False})

        # ── Visual Dashboard ──────────────────────────────────────────────────
        st.markdown("")
        st.markdown('<div class="section-header">📈 Visual Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        left2, right2 = st.columns(2)
        with left2:
            st.plotly_chart(feature_importance_chart(bundle),
                            use_container_width=True,
                            config={"displayModeBar": False})
        with right2:
            # Radar / spider chart of student profile
            categories   = ["Study", "Attendance", "Assignment", "Quiz", "GPA×25", "Participation"]
            student_vals = [
                inputs["study_hours_per_day"] / 12 * 100,
                inputs["attendance_percentage"],
                inputs["assignment_score"],
                inputs["quiz_score"],
                inputs["previous_gpa"] / 4 * 100,
                inputs["participation_score"],
            ]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=student_vals + [student_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(79,70,229,0.25)",
                line=dict(color="#4F46E5", width=2),
                name="Student",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[75] * len(categories) + [75],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(16,185,129,0.08)",
                line=dict(color="#10B981", width=1.5, dash="dot"),
                name="Class Avg",
            ))
            fig_radar.update_layout(
                title="Student Profile Radar",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                    gridcolor="#f3f4f6", tickfont=dict(size=9)),
                    angularaxis=dict(tickfont=dict(size=10)),
                    bgcolor="white",
                ),
                showlegend=True,
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="white",
                font={"family": "Inter"},
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown("")
        st.markdown('<div class="section-header">💡 Performance Insights & Recommendations</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        recs = build_recommendations(raw, score)
        for msg, css in recs:
            st.markdown(f'<div class="rec-card {css}">{msg}</div>', unsafe_allow_html=True)

    # ── About Section ─────────────────────────────────────────────────────────
    with st.expander("ℹ️  About this Project", expanded=False):
        st.markdown(
            """
            ### Student Performance Predictor
            A production-quality ML application that predicts a student's **Final Exam Score**
            from six academic and behavioural features using an ensemble regression model.

            | Feature | Description |
            |---------|-------------|
            | Study Hours / Day | Average daily self-study time |
            | Attendance % | Percentage of classes attended |
            | Assignment Score | Average score across all assignments (0–100) |
            | Quiz Score | Average quiz performance (0–100) |
            | Previous GPA | Cumulative GPA from prior semesters (0–4.0) |
            | Participation Score | Classroom participation rating (0–100) |

            **ML Pipeline:** Data generation → Cleaning → Feature engineering →
            Train/test split → StandardScaler → Random Forest Regressor → Evaluation → joblib persistence.

            **Developer:** Sapna Jabeen &nbsp;|&nbsp; **Year:** 2026
            """,
            unsafe_allow_html=True,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="footer">
            Built with ❤️ by <strong>Sapna Jabeen</strong> · 2026 ·
            Student Performance Predictor · Powered by scikit-learn & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
