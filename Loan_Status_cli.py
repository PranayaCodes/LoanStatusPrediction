import os
import sys
import joblib
import pandas as pd


# Resolve paths relative to this file
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "feature", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "feature", "scaler.pkl")

# Feature order must match exactly what the model was trained on
FEATURE_COLUMNS = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "education_Not Graduate",
    "self_employed_Yes",
]


def divider(char="─", width=52):
    print(char * width)


def header():
    divider("═")
    print("  LOAN STATUS PREDICTION — CLI")
    print("  Logistic Regression + SMOTE + GridSearchCV")
    divider("═")
    print()


def get_float(prompt, min_val=None, max_val=None):
    # Keep prompting until the user enters a valid number within the given range
    while True:
        try:
            val = float(input(f"  {prompt}: ").strip())
            if min_val is not None and val < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  Value must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("  Please enter a valid number.")


def get_choice(prompt, options):
    # Display labeled options and return the chosen value
    print(f"\n  {prompt}:")
    for key, label in options.items():
        print(f"    [{key}] {label}")
    while True:
        choice = input(f"  Select ({'/'.join(options.keys())}): ").strip()
        if choice in options:
            return options[choice]
        print(f"  Please enter one of: {', '.join(options.keys())}")


def cibil_label(score):
    # Return a human-readable credit rating for a CIBIL score
    if score < 500:
        return "Poor"
    if score < 600:
        return "Fair"
    if score < 700:
        return "Good"
    if score < 800:
        return "Very Good"
    return "Exceptional"


def confidence_bar(pct, width=30):
    # Build a simple ASCII progress bar representing confidence percentage
    filled = int(pct / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:.2f}%"


def main():
    header()

    # Load saved model and scaler — exit early if they haven't been trained yet
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("  Model or scaler not found.")
        print("  Run 'python feature/train.py' first.")
        sys.exit(1)

    print("  Enter applicant details below.")
    print()

    # Section 01 — Applicant background information
    divider()
    print("  01 — APPLICANT PROFILE")
    divider()
    no_of_dependents = int(get_float("No. of Dependents (0–10)", min_val=0, max_val=10))
    education        = get_choice("Education",    {"1": "Graduate", "2": "Not Graduate"})
    self_employed    = get_choice("Self Employed", {"1": "Yes",       "2": "No"})

    # Section 02 — Income and credit details
    print()
    divider()
    print("  02 — FINANCIAL PROFILE")
    divider()
    income_annum = get_float("Annual Income in Rs (e.g. 5000000)", min_val=0)
    cibil_score  = int(get_float("CIBIL Score (300–900)", min_val=300, max_val=900))
    print(f"  CIBIL Rating: {cibil_label(cibil_score)}")

    # Section 03 — Loan request details
    print()
    divider()
    print("  03 — LOAN DETAILS")
    divider()
    loan_amount = get_float("Loan Amount in Rs (e.g. 10000000)", min_val=0)
    loan_term   = int(get_float("Loan Term in years (e.g. 10)", min_val=1))

    # Section 04 — Asset holdings across different categories
    print()
    divider()
    print("  04 — ASSET DETAILS")
    divider()
    residential_assets_value = get_float("Residential Assets Value in Rs", min_val=0)
    commercial_assets_value  = get_float("Commercial Assets Value in Rs",   min_val=0)
    luxury_assets_value      = get_float("Luxury Assets Value in Rs",       min_val=0)
    bank_asset_value         = get_float("Bank Asset Value in Rs",          min_val=0)

    # Convert categorical inputs to one-hot encoded binary values
    edu_not_graduate = 1 if education     == "Not Graduate" else 0
    self_emp_yes     = 1 if self_employed == "Yes"          else 0

    # Build a single-row DataFrame in the exact column order the model expects
    raw = pd.DataFrame([[
        no_of_dependents,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        edu_not_graduate,
        self_emp_yes,
    ]], columns=FEATURE_COLUMNS)

    # Scale and run prediction
    scaled      = scaler.transform(raw)
    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]
    confidence  = round(float(max(probability)) * 100, 2)

    # Display the result
    print()
    divider("═")
    print("  ASSESSMENT RESULT")
    divider("═")

    if prediction == 1:
        print("  LOAN APPROVED")
    else:
        print("  LOAN REJECTED")

    print()
    print(f"  Confidence:  {confidence_bar(confidence)}")
    print(f"  Probability: {confidence}%")
    print()

    # Display a clean summary of everything the user entered
    divider()
    print("  INPUT SUMMARY")
    divider()
    print(f"  Dependents               : {no_of_dependents}")
    print(f"  Education                : {education}")
    print(f"  Self Employed            : {self_employed}")
    print(f"  Annual Income            : Rs {income_annum:,.0f}")
    print(f"  CIBIL Score              : {cibil_score} ({cibil_label(cibil_score)})")
    print(f"  Loan Amount              : Rs {loan_amount:,.0f}")
    print(f"  Loan Term                : {loan_term} years")
    print(f"  Residential Assets       : Rs {residential_assets_value:,.0f}")
    print(f"  Commercial Assets        : Rs {commercial_assets_value:,.0f}")
    print(f"  Luxury Assets            : Rs {luxury_assets_value:,.0f}")
    print(f"  Bank Assets              : Rs {bank_asset_value:,.0f}")
    divider("═")
    print()

    # Let the user run another prediction without restarting the script
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