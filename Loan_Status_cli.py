# predict_cli.py
# Loan Status Prediction — Command Line Interface
# Run: python predict_cli.py

import os
import sys
import joblib
import pandas as pd

# ── Paths ───────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "feature", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "feature", "scaler.pkl")

FEATURE_COLUMNS = [
    "Age",
    "Income",
    "Credit_Score",
    "Loan_Amount",
    "Loan_Term",
    "Employment_Status_Self-Employed",
    "Employment_Status_Unemployed",
]

# ── Helpers ─────────────────────────────────────────────────
def divider(char="─", width=52):
    print(char * width)

def header():
    divider("═")
    print("  LOAN STATUS PREDICTION — CLI")
    print("  Logistic Regression + SMOTE")
    divider("═")
    print()

def get_float(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = float(input(f"  {prompt}: ").strip())
            if min_val is not None and val < min_val:
                print(f"  ⚠  Value must be ≥ {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  ⚠  Value must be ≤ {max_val}")
                continue
            return val
        except ValueError:
            print("  ⚠  Please enter a valid number.")

def get_employment():
    options = {"1": "Employed", "2": "Self-Employed", "3": "Unemployed"}
    print()
    print("  Employment Status:")
    for k, v in options.items():
        print(f"    [{k}] {v}")
    while True:
        choice = input("  Select (1/2/3): ").strip()
        if choice in options:
            return options[choice]
        print("  ⚠  Please enter 1, 2, or 3.")

def confidence_bar(pct, width=30):
    filled = int(pct / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:.1f}%"

def score_label(score):
    if score < 580: return "Poor"
    if score < 670: return "Fair"
    if score < 740: return "Good"
    if score < 800: return "Very Good"
    return "Exceptional"

#  Main
def main():
    header()

    # Load model & scaler
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("  ✗  Model or scaler not found.")
        print("     Run 'python feature/train.py' first.")
        sys.exit(1)

    print("  Enter applicant details below.")
    print()

    #  Section 01: Applicant Profile
    divider()
    print("  01 — APPLICANT PROFILE")
    divider()
    age               = get_float("Age (18–100)", min_val=18, max_val=100)
    income            = get_float("Annual Income (USD)", min_val=0)
    employment_status = get_employment()

    #  Section 02: Credit Profile
    print()
    divider()
    print("  02 — CREDIT PROFILE")
    divider()
    credit_score = get_float("Credit Score (300–850)", min_val=300, max_val=850)
    print(f"  → Credit rating: {score_label(int(credit_score))}")

    #  Section 03: Loan Details
    print()
    divider()
    print("  03 — LOAN DETAILS")
    divider()
    loan_amount = get_float("Loan Amount (USD)", min_val=0)
    loan_term   = get_float("Loan Term (months)", min_val=1)

    #  Build Feature Vector
    emp_self_employed = 1 if employment_status == "Self-Employed" else 0
    emp_unemployed    = 1 if employment_status == "Unemployed"    else 0

    raw = pd.DataFrame([[
        age,
        income,
        credit_score,
        loan_amount,
        loan_term,
        emp_self_employed,
        emp_unemployed,
    ]], columns=FEATURE_COLUMNS)

    #  Predict
    scaled      = scaler.transform(raw)
    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]
    confidence  = round(float(max(probability)) * 100, 2)

    # ── Display Result
    print()
    divider("═")
    print("  ASSESSMENT RESULT")
    divider("═")

    if prediction == 1:
        print("  ✓  LOAN APPROVED")
    else:
        print("  ✗  LOAN REJECTED")

    print()
    print(f"  Confidence:  {confidence_bar(confidence)}")
    print(f"  Probability: {confidence}%")
    print()

    # ── Summary Table
    divider()
    print("  INPUT SUMMARY")
    divider()
    print(f"  Age              : {int(age)}")
    print(f"  Income           : ${income:,.0f}")
    print(f"  Employment       : {employment_status}")
    print(f"  Credit Score     : {int(credit_score)} ({score_label(int(credit_score))})")
    print(f"  Loan Amount      : ${loan_amount:,.0f}")
    print(f"  Loan Term        : {int(loan_term)} months")
    divider("═")
    print()

    # ── Run another?
    again = input("  Run another prediction? (y/n): ").strip().lower()
    if again == "y":
        print()
        main()
    else:
        print()
        print("  Goodbye.")
        print()

if __name__ == "__main__":
    main()