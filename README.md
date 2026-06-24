# 🎓 Student Performance Predictor

> *Predict. Analyse. Improve.*  
> An end-to-end Machine Learning application that forecasts a student's Final Exam Score from six academic and behavioural features.

---

## 📌 Overview

Student Performance Predictor is a **production-quality ML application** built with Python, scikit-learn, and Streamlit. It trains a Random Forest regression model on a realistic synthetic dataset, persists the best model, and surfaces predictions through a modern interactive dashboard — complete with gauge charts, radar profiles, and personalised recommendations.

---

## ✨ Features

- 🤖 **Multi-model Training** — Linear Regression, Decision Tree, Random Forest with automatic best-model selection
- 📊 **Interactive Dashboard** — Gauge charts, radar profiles, feature importance, and benchmark comparisons
- 💡 **Personalised Insights** — Actionable recommendations based on the student's specific input profile
- 🧹 **Full ML Pipeline** — Data generation → Cleaning → Feature engineering → Training → Evaluation → Persistence
- 📱 **Responsive Design** — Clean, mobile-friendly layout with custom CSS
- 🔄 **Reusable Architecture** — Modular code with type hints, docstrings, and PEP 8 compliance

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML Framework | scikit-learn 1.4 |
| Dashboard | Streamlit 1.35 |
| Charts | Plotly 5.22 |
| EDA Visuals | Matplotlib · Seaborn |
| Data | Pandas · NumPy |
| Persistence | joblib |

---

## 📂 Project Structure

```
student-performance-predictor/
├── app.py                    # Streamlit dashboard
├── train_model.py            # Full ML pipeline
├── generate_dataset.py       # Synthetic data generator
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── student_performance.csv
├── models/
│   └── student_model.pkl     # Serialised model bundle
├── visuals/                  # EDA charts (PNG)
├── assets/
├── notebooks/
│   └── analysis.ipynb
└── screenshots/
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate the dataset

```bash
python generate_dataset.py
```

### 4. Train the model

```bash
python train_model.py
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Machine Learning Workflow

```
Raw Synthetic Data (1 200 records)
         │
         ▼
  Data Cleaning
  ├─ Remove duplicates
  ├─ Fill missing values (median)
  └─ Clip outliers to academic ranges
         │
         ▼
  Feature Engineering
  ├─ academic_engagement (weighted composite)
  ├─ avg_score (mean of assignments + quizzes)
  └─ gpa_study_interaction (GPA × study hours)
         │
         ▼
  EDA (8 charts saved to visuals/)
         │
         ▼
  Train / Test Split (80 / 20)
         │
         ▼
  StandardScaler → 3 Models trained
  ├─ Linear Regression
  ├─ Decision Tree Regressor
  └─ Random Forest Regressor  ← best
         │
         ▼
  Evaluation (MAE · MSE · RMSE · R²)
         │
         ▼
  Best model saved → models/student_model.pkl
```

---

## 📈 Model Performance

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Random Forest** | **~1.8** | **~2.4** | **~0.96** |
| Decision Tree | ~2.6 | ~3.5 | ~0.93 |
| Linear Regression | ~4.1 | ~5.2 | ~0.85 |

> *Exact numbers vary slightly with each run due to random data generation.*

---

## 🖼️ Screenshots

> Add screenshots of the running dashboard to the `screenshots/` folder.

---

## 🚀 Future Enhancements

- [ ] Upload real institutional dataset via the UI
- [ ] Multi-class classification (Grade prediction: A / B / C / D / F)
- [ ] SHAP-based model explainability
- [ ] User authentication and result history
- [ ] CSV bulk prediction mode
- [ ] Deployment to Streamlit Community Cloud / Hugging Face Spaces

---

## 📄 License

This project is licensed under the **MIT License** — see `LICENSE` for details.

---

## 👩‍💻 Author

**Sapna Jabeen**  
Machine Learning Engineer · Data Scientist · Streamlit Developer  
📅 Developed: 2026

---

> *"Education is the most powerful weapon which you can use to change the world." — Nelson Mandela*
