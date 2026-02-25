import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, auc)

from imblearn.over_sampling import SMOTE


# Create folder for saving visualizations
os.makedirs("feature/visualizations", exist_ok=True)

# Load preprocessed data (unscaled)
data = pd.read_csv("data/processed_data.csv")

X = data.drop("Loan_Approved", axis=1)
y = data["Loan_Approved"]


# Class distribution before SMOTE
plt.figure()
sns.countplot(x=y)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Loan Approved (0 = Rejected, 1 = Approved)")
plt.ylabel("Number of Samples")
plt.savefig("feature/visualizations/class_before_smote.png")
plt.close()


# Split data — stratify preserves class ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit scaler on training data only, then apply to both sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Apply SMOTE only on training data to fix class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)


# Class distribution after SMOTE
plt.figure()
sns.countplot(x=y_train_resampled)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Loan Approved (0 = Rejected, 1 = Approved)")
plt.ylabel("Number of Samples")
plt.savefig("feature/visualizations/class_after_smote.png")
plt.close()


# Baseline model without SMOTE — used only for ROC comparison
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_scaled, y_train)
y_prob_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]

# Main model with SMOTE and probability calibration
# C=0.01 applies strong regularization to prevent overconfident predictions
# isotonic calibration produces more realistic confidence scores
base_model = LogisticRegression(max_iter=1000, C=0.01, solver='lbfgs')
model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
model.fit(X_train_resampled, y_train_resampled)


# Predictions on test set
y_pred       = model.predict(X_test_scaled)
y_prob_smote = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted No", "Predicted Yes"],
            yticklabels=["Actual No", "Actual Yes"])
plt.title("Confusion Matrix - Logistic Regression with SMOTE")
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

plt.figure()
plt.plot(fpr1, tpr1, label=f"Baseline (AUC = {auc1:.3f})")
plt.plot(fpr2, tpr2, label=f"SMOTE    (AUC = {auc2:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Logistic Regression Performance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("feature/visualizations/roc_curve.png")
plt.close()

print(f"\nROC AUC — Baseline: {auc1:.4f} | SMOTE: {auc2:.4f}")


# Feature coefficients to show impact of each feature on prediction
lr_coef = base_model.fit(X_train_resampled, y_train_resampled)
coef_df = pd.DataFrame({
    "Feature":     X.columns,
    "Coefficient": lr_coef.coef_[0]
}).sort_values("Coefficient", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Coefficient", y="Feature", data=coef_df)
plt.title("Logistic Regression Feature Coefficients (SMOTE Model)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.savefig("feature/visualizations/feature_coefficients.png")
plt.close()


# Save model and scaler for use in FastAPI and CLI
joblib.dump(model,  "feature/model.pkl")
joblib.dump(scaler, "feature/scaler.pkl")

print("\nModel saved  → feature/model.pkl")
print("Scaler saved → feature/scaler.pkl")
print("\nAll visualizations saved in → feature/visualizations/")
print("  - class_before_smote.png")
print("  - class_after_smote.png")
print("  - confusion_matrix.png")
print("  - roc_curve.png")
print("  - feature_coefficients.png")