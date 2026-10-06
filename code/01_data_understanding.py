"""Data understanding and quality assessment (read-only profile of the original data)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "employee_income_original.csv")

print("=== SHAPE ===")
print(f"{df.shape[0]} observations, {df.shape[1]} variables\n")

print("=== DATA TYPES AS LOADED ===")
print(df.dtypes.to_string(), "\n")

print("=== MISSING VALUES ===")
miss = df.isna().sum()
print(pd.DataFrame({"missing": miss, "percent": (miss / len(df) * 100).round(1)}).to_string(), "\n")
print(f"Rows with at least one missing value: {df.isna().any(axis=1).sum()}\n")

print("=== DUPLICATES ===")
print(f"Fully identical rows (extra copies): {df.duplicated().sum()}")
print(f"Repeated Employee_ID values (extra copies): {df['Employee_ID'].duplicated().sum()}")
dup_ids = df.loc[df["Employee_ID"].duplicated(keep=False), "Employee_ID"].unique()
partial = [i for i in dup_ids if df[df["Employee_ID"] == i].drop_duplicates().shape[0] > 1]
print(f"IDs repeated with differing details: {len(partial)} {partial[:10]}\n")

print("=== CATEGORICAL VALUES (exact spellings) ===")
for col in ["Gender", "Education", "Department"]:
    print(f"-- {col}")
    print(df[col].value_counts(dropna=False).to_string(), "\n")
print("-- Job_Level")
print(df["Job_Level"].value_counts(dropna=False).sort_index().to_string(), "\n")

print("=== NUMERIC SUMMARY ===")
num = ["Age", "Years_Experience", "Job_Level", "Hours_Per_Week", "Performance_Score", "Training_Hours", "Monthly_Income"]
print(df[num].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.round(2).to_string(), "\n")

print("=== VALUES WORTH A CLOSER LOOK ===")
for col in num:
    s = df[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    print(f"{col:18s} min={s.min():>10} max={s.max():>10}  negatives={(s < 0).sum():>3}  zeros={(s == 0).sum():>3}  "
          f"IQR-outliers low={(s < lo).sum():>3} high={(s > hi).sum():>3}  "
          f"5 lowest={sorted(s)[:5]}  5 highest={sorted(s)[-5:]}")
print()
print("Non-whole values in Age / Years_Experience / Job_Level / Hours_Per_Week / Training_Hours:")
for col in ["Age", "Years_Experience", "Job_Level", "Hours_Per_Week", "Training_Hours"]:
    s = df[col].dropna()
    print(f"  {col}: {(s != s.round()).sum()}")
print()
age_exp = df.dropna(subset=["Age", "Years_Experience"])
print(f"Experience implies starting work before age 16: {(age_exp['Age'] - age_exp['Years_Experience'] < 16).sum()} rows")
print(f"Experience greater than age: {(age_exp['Years_Experience'] > age_exp['Age']).sum()} rows")
print(f"Performance score outside 1-5: {((df['Performance_Score'] < 1) | (df['Performance_Score'] > 5)).sum()} rows")
print(f"Hours per week above 80 or below 10: {((df['Hours_Per_Week'] > 80) | (df['Hours_Per_Week'] < 10)).sum()} rows")
print(f"Monthly income <= 0: {(df['Monthly_Income'] <= 0).sum()} rows")
print(f"Employee_ID format not EMP + 4 digits: {(~df['Employee_ID'].astype(str).str.fullmatch(r'EMP\d{4}')).sum()} rows")
