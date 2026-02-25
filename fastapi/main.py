import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# Resolve paths relative to this file so the app works from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

app = FastAPI()

# Serve static files (CSS, JS) from the static folder
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)

# Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load trained model and scaler once at startup
model  = joblib.load(os.path.join(ROOT_DIR, "feature", "model.pkl"))
scaler = joblib.load(os.path.join(ROOT_DIR, "feature", "scaler.pkl"))

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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Render the main form page with no prediction result
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    no_of_dependents:          int   = Form(...),
    income_annum:              float = Form(...),
    loan_amount:               float = Form(...),
    loan_term:                 int   = Form(...),
    cibil_score:               int   = Form(...),
    residential_assets_value:  float = Form(...),
    commercial_assets_value:   float = Form(...),
    luxury_assets_value:       float = Form(...),
    bank_asset_value:          float = Form(...),
    education:                 str   = Form(...),
    self_employed:             str   = Form(...),
):
    # Convert categorical fields to one-hot encoded values
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

    # Scale and predict
    scaled     = scaler.transform(raw)
    prediction = model.predict(scaled)[0]
    proba      = model.predict_proba(scaled)[0]
    confidence = round(float(max(proba)) * 100, 2)

    result       = "Approved" if prediction == 1 else "Rejected"
    result_class = "approved" if prediction == 1 else "rejected"

    return templates.TemplateResponse("index.html", {
        "request":      request,
        "result":       result,
        "result_class": result_class,
        "confidence":   confidence,
    })