import pandas as pd

# Load raw dataset
data = pd.read_csv("data/train.csv")

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
print("\nLoan Approved distribution:")
print(data["Loan_Approved"].value_counts())

#  Data Cleaning 
print("\nDATA CLEANING \n")

duplicates = data.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")

if duplicates > 0:
    data = data.drop_duplicates()
    print("Duplicates removed.")
else:
    print("No duplicates found.")

print("\nMissing values after cleaning:")
print(data.isnull().sum())

#  Preprocessing 
print("\n DATA PREPROCESSING\n")

# One-hot encode Employment_Status (drop_first removes "Employed" as baseline)
data = pd.get_dummies(data, columns=["Employment_Status"], drop_first=True)
print("Categorical feature 'Employment_Status' encoded.")

# Separate features and target
X = data.drop("Loan_Approved", axis=1)
y = data["Loan_Approved"]
print("Features and target separated.")

# No scaling here — StandardScaler is applied in train.py
processed_data = pd.concat([X, y.reset_index(drop=True)], axis=1)

# Save processed data
processed_data.to_csv("data/processed_data.csv", index=False)
print("\nProcessed dataset saved as 'processed_data.csv'")
print("\n PREPROCESSING COMPLETED SUCCESSFULLY \n")