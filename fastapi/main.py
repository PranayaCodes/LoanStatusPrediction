# main.py
# Loan Status Prediction — FastAPI App

import sys
import os

# Allow imports from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

#  App Setup 

app = FastAPI(title="AI Loan Predictor")

BASE_DIR = os.path.dirname(__file__)

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

MODEL_PATH  = os.path.join(BASE_DIR, "..", "feature", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "feature", "scaler.pkl")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_COLUMNS = [
    "Age",
    "Income",
    "Credit_Score",
    "Loan_Amount",
    "Loan_Term",
    "Employment_Status_Self-Employed",
    "Employment_Status_Unemployed",
]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float             = Form(...),
    income: float          = Form(...),
    credit_score: float    = Form(...),
    loan_amount: float     = Form(...),
    loan_term: float       = Form(...),
    employment_status: str = Form(...),
):
    
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

    scaled = scaler.transform(raw)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    result = "Approved ✓" if prediction == 1 else "Rejected ✗"
    confidence = round(float(max(probability)) * 100, 2)
    result_class = "approved" if prediction == 1 else "rejected"

    return templates.TemplateResponse("index.html", {
        "request":      request,
        "result":       result,
        "confidence":   confidence,
        "result_class": result_class,
    })