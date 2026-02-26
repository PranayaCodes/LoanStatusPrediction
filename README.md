**README.md**

---

# 🏦 Loan Approval Prediction System

A machine learning web application that predicts loan approval outcomes based on applicant financial and demographic data. Built with Logistic Regression, SMOTE for class imbalance handling, and deployed via FastAPI with a clean web interface and CLI tool.

---

## 📊 Project Overview

This project trains a Logistic Regression model on a loan approval dataset to predict whether a loan application will be **Approved** or **Rejected**. The system addresses class imbalance using SMOTE and applies isotonic calibration to produce reliable confidence scores. The trained model is served through both a FastAPI web application and a command-line interface.

---

## ✨ Features

- Full data preprocessing pipeline with EDA visualizations
- SMOTE oversampling to handle class imbalance
- GridSearchCV hyperparameter tuning scored on F1
- Isotonic calibration for reliable probability outputs
- Confusion matrix, ROC curve, and feature coefficient visualizations
- FastAPI web application with live prediction and confidence score
- Command-line interface for quick terminal predictions

---

## 🗂️ Project Structure

```
LoanStatusPrediction/
│
├── data/
│   ├── loan_approval_dataset.csv       # Raw dataset
│   ├── processed_data.csv              # Cleaned and encoded dataset
│   └── eda_visualizations/             # EDA charts
│
├── feature/
│   ├── train.py                        # Model training script
│   ├── model.pkl                       # Trained model
│   ├── scaler.pkl                      # Fitted scaler
│   └── visualizations/                 # Training visualizations
│
├── fastapi/
│   ├── main.py                         # FastAPI application
│   ├── templates/                      # HTML templates
│   └── static/                         # CSS and JS files
│
├── data/preprocess.py                  # Data preprocessing script
└── Loan_Status_cli.py                  # CLI prediction tool
```

---

## ⚙️ Setup and Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/LoanStatusPrediction.git
cd LoanStatusPrediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run preprocessing**
```bash
python data/preprocess.py
```

**4. Train the model**
```bash
python feature/train.py
```

**5. Start the web application**
```bash
cd fastapi
uvicorn main:app --reload
```

**6. Or use the CLI**
```bash
python Loan_Status_cli.py
```

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.9133 | 0.9208 | 0.9416 | 0.9311 |
| Logistic Regression (SMOTE) | 0.9333 | 0.9759 | 0.9153 | 0.9446 |

**ROC AUC — Baseline: 0.973 | SMOTE: 0.974**

---

## 🔑 Key Features Used

- No. of Dependents
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential, Commercial, Luxury and Bank Asset Values
- Education
- Self Employed Status

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data processing
- **Scikit-learn** — Model training and evaluation
- **Imbalanced-learn** — SMOTE resampling
- **Matplotlib / Seaborn / Plotly** — Visualizations
- **FastAPI** — Web application
- **Joblib** — Model serialization

---

## 📄 License

This project is for educational purposes.

---

---

**GitHub Description (short, for the repo bio):**

> A loan approval prediction system built with Logistic Regression, SMOTE, and FastAPI. Predicts whether a loan will be approved or rejected based on financial and demographic data, with a web interface and CLI tool. Achieves 93.33% accuracy and 0.974 AUC.
