# preprocess.py
# Loan Status Prediction Project
# Data Cleaning and Preprocessing

import pandas as pd
from sklearn.preprocessing import StandardScaler



# STEP 1: LOAD DATASET


# Load dataset
data = pd.read_csv("data/train.csv")

print("\n DATASET OVERVIEW (BEFORE CLEANING)\n")

# Show first 5 rows
print("First 5 rows:")
print(data.head())

# Show dataset size
print("\nDataset shape (rows, columns):")
print(data.shape)

# Show column names
print("\nColumns:")
print(data.columns.tolist())

# Show data types
print("\nData types:")
print(data.dtypes)

# Check missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Statistical summary
print("\nStatistical summary:")
print(data.describe())

# Target distribution
print("\nLoan Approved distribution:")
print(data["Loan_Approved"].value_counts())



# STEP 2: DATA CLEANING


print("\nDATA CLEANING \n")

# Check duplicate rows
duplicates = data.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")

# Remove duplicates if present
if duplicates > 0:
    data = data.drop_duplicates()
    print("Duplicates removed.")
else:
    print("No duplicates found.")

# Confirm missing values again
print("\nMissing values after cleaning:")
print(data.isnull().sum())



# STEP 3: DATA PREPROCESSING

print("\n DATA PREPROCESSING\n")

# Convert categorical variable into numeric
data = pd.get_dummies(data, columns=["Employment_Status"], drop_first=True)

print("Categorical feature 'Employment_Status' encoded.")


# Separate features and target
X = data.drop("Loan_Approved", axis=1)
y = data["Loan_Approved"]

print("Features and target separated.")


# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature scaling applied.")


# Convert scaled data back to dataframe
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Combine features and target
processed_data = pd.concat([X_scaled, y], axis=1)


# STEP 4: SAVE PROCESSED DATA

processed_data.to_csv("data/processed_data.csv", index=False)

print("\nProcessed dataset saved as 'processed_data.csv'")
print("\n PREPROCESSING COMPLETED SUCCESSFULLY ")