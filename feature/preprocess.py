import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


os.makedirs("data/eda_visualizations", exist_ok=True)


# Loading the raw loan dataset to understand what we are working with.
# We check the structure, column types and missing values before
# making any changes to the data.

data = pd.read_csv("data/loan_approval_dataset.csv")

# The CSV has leading spaces baked into column names and string values
data.columns = data.columns.str.strip()
for col in data.select_dtypes(include=["object", "string"]).columns:
    data[col] = data[col].str.strip()

print("\n DATASET OVERVIEW (BEFORE CLEANING)\n")

print("First 5 rows:")
print(data.head())

print("\nDataset shape (rows, columns):")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values in each column:")
print(data.isnull().sum())

print("\nStatistical summary:")
print(data.describe())

print("\nLoan status distribution:")
print(data["loan_status"].value_counts())


# Checking for duplicate rows and null entries before proceeding.
# The dataset is clean but this step ensures nothing slips through.

print("\n DATA CLEANING\n")

duplicates = data.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")

if duplicates > 0:
    data = data.drop_duplicates()
    print("Duplicates removed.")
else:
    print("No duplicates found.")

print("\nMissing values after cleaning:")
print(data.isnull().sum())

# loan_id is just a row identifier and adds no predictive value
data.drop(columns=["loan_id"], inplace=True)
print("\nDropped column: loan_id")


# After removing duplicates and loan_id we do a second overview
# to confirm the dataset is clean and ready for EDA and encoding.

print("\n DATASET OVERVIEW (AFTER CLEANING)\n")

print("Dataset shape (rows, columns):")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nStatistical summary:")
print(data.describe())

print("\nLoan status distribution:")
print(data["loan_status"].value_counts())


# Generating charts to understand the distribution of features
# and how they relate to the loan approval outcome.
# These are saved to disk and not shown interactively.

print("\n GENERATING EDA VISUALIZATIONS\n")

# Target class balance
plt.figure(figsize=(6, 4))
sns.countplot(x="loan_status", hue="loan_status", data=data, palette="muted", legend=False)
plt.title("Target Distribution — Loan Status (Raw)")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_visualizations/target_distribution.png")
plt.close()

# Distribution of all numeric features
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(data[col], bins=30, color="steelblue", edgecolor="white")
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Numeric Feature Distributions", y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig("data/eda_visualizations/numeric_distributions.png")
plt.close()

# Boxplots to spot outliers in the financial columns
financial_cols = [
    "income_annum",
    "loan_amount",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(financial_cols):
    axes[i].boxplot(data[col], patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    axes[i].set_title(col)
    axes[i].set_ylabel("Value (Rs)")

plt.suptitle("Boxplots — Financial Features (Outlier Check)", fontsize=13)
plt.tight_layout()
plt.savefig("data/eda_visualizations/financial_boxplots.png")
plt.close()

# Categorical feature counts
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(x="education",     hue="education",     data=data, ax=axes[0], palette="Set2", legend=False)
sns.countplot(x="self_employed", hue="self_employed", data=data, ax=axes[1], palette="Set2", legend=False)
axes[0].set_title("Education Distribution")
axes[1].set_title("Self Employed Distribution")
plt.tight_layout()
plt.savefig("data/eda_visualizations/categorical_distributions.png")
plt.close()

# Loan outcome split by education and employment type
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x="education",     hue="loan_status", data=data, ax=axes[0], palette="Set1")
sns.countplot(x="self_employed", hue="loan_status", data=data, ax=axes[1], palette="Set1")
axes[0].set_title("Loan Status by Education")
axes[1].set_title("Loan Status by Employment Type")
for ax in axes:
    ax.legend(title="Loan Status")
plt.tight_layout()
plt.savefig("data/eda_visualizations/status_by_category.png")
plt.close()

# Correlation heatmap across all numeric features
plt.figure(figsize=(10, 7))
sns.heatmap(data[numeric_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, square=True)
plt.title("Correlation Heatmap — Numeric Features")
plt.tight_layout()
plt.savefig("data/eda_visualizations/correlation_heatmap.png")
plt.close()

print("Visualizations saved to: data/eda_visualizations/")


# The model only works with numbers so we convert education
# and self_employed into binary columns using one-hot encoding.
# drop_first=True drops the reference category to avoid
# multicollinearity, which is standard for logistic regression.

print("\n DATA PREPROCESSING\n")

data["loan_status"] = data["loan_status"].map({"Approved": 1, "Rejected": 0})

data = pd.get_dummies(data, columns=["education", "self_employed"], drop_first=True)
data.columns = data.columns.str.strip().str.replace(r"\s+", " ", regex=True)

print("Categorical features encoded: education, self_employed")
print("Columns after encoding:", data.columns.tolist())


# Splitting into input features (X) and the target (y) to
# verify everything looks correct before saving.

X = data.drop("loan_status", axis=1)
y = data["loan_status"]

print("\nFeatures and target separated.")
print("Feature columns:", X.columns.tolist())


# StandardScaler is applied in train.py after the train/test split.
# Scaling here would cause data leakage because the scaler would
# learn statistics from the full dataset including the test portion.


# Writing the cleaned and encoded dataset to disk so train.py
# can load it directly without repeating these steps.

processed_data = pd.concat([X, y.reset_index(drop=True)], axis=1)
processed_data.to_csv("data/processed_data.csv", index=False)

print("\nProcessed dataset saved as 'data/processed_data.csv'")
print(f"Final shape: {processed_data.shape}")
print("\n PREPROCESSING COMPLETED SUCCESSFULLY \n")