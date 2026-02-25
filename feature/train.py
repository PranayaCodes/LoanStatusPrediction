import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from imblearn.over_sampling import SMOTE


os.makedirs("feature/visualizations", exist_ok=True)


# ── Load Processed Data ──────────────────────────────────────
# Reading the cleaned and encoded dataset produced by preprocess.py.
# The target column loan_status is separated from the input features.

data = pd.read_csv("data/processed_data.csv")

X = data.drop("loan_status", axis=1)
y = data["loan_status"]

print("\n MODEL TRAINING — LOAN APPROVAL PREDICTION\n")
print(f"Dataset loaded. Shape: {data.shape}")
print(f"Features: {X.columns.tolist()}")


# ── Class Distribution Before SMOTE ─────────────────────────
# Visualizing the class imbalance in the raw dataset before
# any resampling. This helps show why SMOTE is needed.

plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, palette="muted", legend=False)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Loan Status  (0 = Rejected,  1 = Approved)")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("feature/visualizations/class_before_smote.png")
plt.close()


# ── Train / Test Split ───────────────────────────────────────
# Splitting the data into 80% training and 20% testing.
# stratify=y ensures both splits have the same class ratio.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")


# ── Feature Scaling ──────────────────────────────────────────
# Fitting the scaler only on training data to prevent leakage.
# The same scaler is then applied to the test set for consistency.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ── SMOTE Resampling ─────────────────────────────────────────
# Applying SMOTE only on the training data to fix class imbalance
# by generating synthetic samples for the minority class.
# Applying it to test data would give misleading evaluation results.

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAfter SMOTE — Resampled training size: {X_train_resampled.shape[0]}")


# ── Class Distribution After SMOTE ──────────────────────────
# Confirming that both classes are now balanced after resampling.

plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_resampled, hue=y_train_resampled, palette="muted", legend=False)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Loan Status  (0 = Rejected,  1 = Approved)")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("feature/visualizations/class_after_smote.png")
plt.close()


# ── Baseline Model ───────────────────────────────────────────
# Training a simple logistic regression without SMOTE to use
# as a comparison baseline in the ROC curve later.

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_scaled, y_train)
y_prob_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]


# ── GridSearchCV ─────────────────────────────────────────────
# Using GridSearchCV to find the optimal regularization strength C.
# A lower C means more regularization, higher C means less.
# We score on F1 since the dataset has class imbalance.

print("\nRunning GridSearchCV to find best C...")

param_grid = {"C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10]}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, solver="lbfgs"),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
grid_search.fit(X_train_resampled, y_train_resampled)

best_C = grid_search.best_params_["C"]
print(f"Best C found     : {best_C}")
print(f"Best CV F1 Score : {round(grid_search.best_score_, 4)}")

results_df = pd.DataFrame(grid_search.cv_results_)[
    ["param_C", "mean_test_score", "std_test_score"]
]
results_df.columns = ["C Value", "Mean F1 Score", "Std F1 Score"]
results_df = results_df.sort_values("Mean F1 Score", ascending=False)

print("\nGridSearchCV Results:")
print(results_df.to_string(index=False))


# ── Final Model ──────────────────────────────────────────────
# Training the final model using the best C with isotonic calibration.
# Isotonic calibration produces more reliable confidence scores
# than the default sigmoid method.

base_model = LogisticRegression(max_iter=1000, C=best_C, solver="lbfgs")
model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
model.fit(X_train_resampled, y_train_resampled)


# ── Evaluation ───────────────────────────────────────────────
# Evaluating the model on the held-out test set using standard
# classification metrics to measure real-world performance.

y_pred       = model.predict(X_test_scaled)
y_prob_smote = model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n" + "=" * 55)
print("  MODEL PERFORMANCE")
print("=" * 55)
print(f"  Accuracy  : {round(accuracy, 4)}")
print(f"  Precision : {round(precision, 4)}")
print(f"  Recall    : {round(recall, 4)}")
print(f"  F1 Score  : {round(f1, 4)}")


# Confusion matrix showing true vs predicted outcomes
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Rejected", "Predicted Approved"],
    yticklabels=["Actual Rejected", "Actual Approved"],
)
plt.title("Confusion Matrix — Logistic Regression + SMOTE")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("feature/visualizations/confusion_matrix.png")
plt.close()


# ROC curve comparing baseline vs SMOTE model
fpr1, tpr1, _ = roc_curve(y_test, y_prob_baseline)
auc1          = auc(fpr1, tpr1)

fpr2, tpr2, _ = roc_curve(y_test, y_prob_smote)
auc2          = auc(fpr2, tpr2)

plt.figure(figsize=(7, 5))
plt.plot(fpr1, tpr1, label=f"Baseline  (AUC = {auc1:.3f})")
plt.plot(fpr2, tpr2, label=f"SMOTE     (AUC = {auc2:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.title("ROC Curve — Logistic Regression Performance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("feature/visualizations/roc_curve.png")
plt.close()

print(f"\n  ROC AUC — Baseline: {auc1:.4f}  |  SMOTE: {auc2:.4f}")


# Interactive ROC curve using Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=fpr1, y=tpr1,
    mode="lines",
    name=f"Baseline  (AUC = {auc1:.3f})",
))

fig.add_trace(go.Scatter(
    x=fpr2, y=tpr2,
    mode="lines",
    name=f"SMOTE  (AUC = {auc2:.3f})",
))

fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    line=dict(dash="dash", color="black"),
    name="Random Classifier",
))

fig.update_layout(
    title="Interactive ROC Curve — Logistic Regression Performance",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    legend=dict(x=0.6, y=0.1),
)

fig.write_html("feature/visualizations/roc_curve_interactive.html")


# GridSearchCV C value vs F1 Score chart
plt.figure(figsize=(7, 4))
plt.plot(
    results_df["C Value"].astype(float),
    results_df["Mean F1 Score"],
    marker="o",
    color="steelblue",
)
plt.xscale("log")
plt.axvline(x=best_C, color="red", linestyle="--", label=f"Best C = {best_C}")
plt.title("GridSearchCV — C Value vs F1 Score")
plt.xlabel("C Value (log scale)")
plt.ylabel("Mean F1 Score (CV = 5)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("feature/visualizations/gridsearch_results.png")
plt.close()


# Feature coefficients showing how much each feature pushes
# toward approval or rejection in the logistic regression model
lr_coef = base_model.fit(X_train_resampled, y_train_resampled)
coef_df = pd.DataFrame({
    "Feature":     X.columns,
    "Coefficient": lr_coef.coef_[0],
}).sort_values("Coefficient", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Coefficient", y="Feature", data=coef_df)
plt.title("Feature Coefficients — Logistic Regression + SMOTE")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.savefig("feature/visualizations/feature_coefficients.png")
plt.close()


# ── Save Model and Scaler ────────────────────────────────────
# Saving the trained model and scaler so they can be loaded
# by the FastAPI app and the CLI without retraining.

joblib.dump(model,  "feature/model.pkl")
joblib.dump(scaler, "feature/scaler.pkl")

print("\n  Model saved  →  feature/model.pkl")
print("  Scaler saved →  feature/scaler.pkl")

print("\nVisualizations saved in: feature/visualizations/")
print("  - class_before_smote.png")
print("  - class_after_smote.png")
print("  - confusion_matrix.png")
print("  - roc_curve.png")
print("  - roc_curve_interactive.html")
print("  - gridsearch_results.png")
print("  - feature_coefficients.png")
print("\nTraining complete.")