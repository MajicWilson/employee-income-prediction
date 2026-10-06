"""Data cleaning and preprocessing.

Reads the original file, applies the documented cleaning rules, and writes data/employee_income_clean.csv.
Missing predictor values are left blank here; they are filled inside the modelling pipeline using the
training set only (see 04_models.py), so that no information from the test set leaks into training.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "data" / "employee_income_original.csv")
df = raw.copy()
log = []

def note(rule, count, detail=""):
    log.append({"rule": rule, "rows_or_cells_affected": int(count), "detail": detail})

# 1. Duplicate observations: identical in every column (same Employee_ID, same details).
n_dup = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
note("1. Remove identical duplicate rows", n_dup, f"{len(raw)} -> {len(df)} rows")

# 2. Inconsistent categorical values.
gender_map = {"F": "Female", "female": "Female", "M": "Male"}
dept_map = {"Human Resources": "HR", "I.T.": "IT", "Mktg": "Marketing", "finance": "Finance"}
n_g = df["Gender"].isin(gender_map).sum()
n_d = df["Department"].isin(dept_map).sum()
df["Gender"] = df["Gender"].replace(gender_map)
df["Department"] = df["Department"].replace(dept_map)
note("2a. Standardise Gender spellings", n_g, str(gender_map))
note("2b. Standardise Department spellings", n_d, str(dept_map))

# 3. Impossible or unrealistic predictor values -> set to missing (filled later in the pipeline).
bad_age = (df["Age"] < 18) | (df["Age"] > 75)
note("3a. Impossible Age (<18 or >75) set to missing", bad_age.sum(),
     f"values: {sorted(df.loc[bad_age, 'Age'].tolist())}")
df.loc[bad_age, "Age"] = np.nan

bad_hours = (df["Hours_Per_Week"] < 10) | (df["Hours_Per_Week"] > 80)
note("3b. Impossible Hours_Per_Week (<10 or >80) set to missing", bad_hours.sum(),
     f"values: {sorted(df.loc[bad_hours, 'Hours_Per_Week'].tolist())}")
df["Hours_Per_Week"] = df["Hours_Per_Week"].astype("float")
df.loc[bad_hours, "Hours_Per_Week"] = np.nan

# 4. Target variable: rows without a usable Monthly_Income cannot be used to train or test a model.
no_target = df["Monthly_Income"].isna()
note("4a. Drop rows with missing Monthly_Income", no_target.sum(),
     f"IDs: {df.loc[no_target, 'Employee_ID'].tolist()}")
df = df[~no_target]

# Extreme incomes: every other employee earns at most about GHS 20,600; these five are 3.5x to 7.5x that,
# include a level-1 employee with 2 years' experience on GHS 73,003, and do not match their job level even
# after dividing by 10. Treated as unreliable records and removed rather than corrected by guesswork.
extreme = df["Monthly_Income"] > 30000
note("4b. Drop rows with unreliable extreme Monthly_Income (> GHS 30,000)", extreme.sum(),
     f"IDs and values: {df.loc[extreme, ['Employee_ID', 'Monthly_Income']].values.tolist()}")
df = df[~extreme].reset_index(drop=True)

# 5. Data types: whole-number columns stored as decimals because of blanks -> nullable integers.
for col in ["Age", "Years_Experience", "Hours_Per_Week", "Training_Hours"]:
    df[col] = df[col].round().astype("Int64")
df["Job_Level"] = df["Job_Level"].astype("Int64")
for col in ["Gender", "Education", "Department"]:
    df[col] = df[col].astype("string")
note("5. Convert Age, Years_Experience, Hours_Per_Week, Training_Hours to whole numbers", 0,
     "type change only; categorical columns stored as text")

# 6. Consistency check after cleaning: experience should not imply starting work before age 16.
chk = df.dropna(subset=["Age", "Years_Experience"])
note("6. Check: Age - Years_Experience < 16 after cleaning", (chk["Age"] - chk["Years_Experience"] < 16).sum(),
     "should be 0")

df.to_csv(ROOT / "data" / "employee_income_clean.csv", index=False)
log_df = pd.DataFrame(log)
log_df.to_csv(ROOT / "outputs" / "02_cleaning_log.csv", index=False)

print(log_df.to_string(index=False))
print(f"\nFinal cleaned dataset: {len(df)} rows, {df.shape[1]} columns")
print("\nMissing predictor values left for pipeline imputation:")
print(df.isna().sum()[df.isna().sum() > 0].to_string())
print(f"\nRows with at least one missing predictor: {df.isna().any(axis=1).sum()}")
print("\nCategories after cleaning:")
for col in ["Gender", "Education", "Department"]:
    print(f"  {col}: {df[col].value_counts(dropna=False).to_dict()}")
