"""Checks on the selected model (Linear Regression).

- Paired comparison with Gradient Boosting and Ridge on the same 15 cross-validation folds (one-standard-error rule).
- Bootstrap 95% confidence intervals for the test-set MAE, RMSE and R2.
- Coefficients converted to GHS per natural unit.
- Error check by gender (gender is not a predictor; this confirms the model is not less accurate for either group).
- Final model refitted on all 685 employees for the example prediction.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m", ROOT / "code" / "04_models.py")
# Re-use the exact pipeline definitions without re-running 04_models.py's training section.
src = (ROOT / "code" / "04_models.py").read_text(encoding="utf-8").split("# ---------------- Split and models")[0]
ns = {"__file__": str(ROOT / "code" / "04_models.py")}
exec(compile(src, "04_models_definitions", "exec"), ns)
make_pipeline, RAW_PREDICTORS, NUMERIC, SEED, df = ns["make_pipeline"], ns["RAW_PREDICTORS"], ns["NUMERIC"], ns["SEED"], ns["df"]
from sklearn.model_selection import train_test_split

X = df[RAW_PREDICTORS].copy()
for c in ["Education", "Department"]:
    X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
y = df["Monthly_Income"].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=df["Job_Level"])

# 1. Paired fold-by-fold comparison (same 15 folds for every model).
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
models = {"Linear Regression": (LinearRegression(), True),
          "Ridge (alpha=10)": (Ridge(alpha=10), True),
          "Gradient Boosting (tuned)": (GradientBoostingRegressor(random_state=SEED, learning_rate=0.03, max_depth=2,
                                                                  n_estimators=400, subsample=0.8), False)}
fold_rmse = {n: -cross_val_score(make_pipeline(m, s), X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
             for n, (m, s) in models.items()}
print("=== Paired cross-validation comparison (15 folds, RMSE in GHS) ===")
base = fold_rmse["Linear Regression"]
se_best = base.std(ddof=1) / np.sqrt(len(base))
print(f"Linear Regression: mean {base.mean():.1f}; standard error of the mean {se_best:.1f}")
for n, v in fold_rmse.items():
    if n == "Linear Regression":
        continue
    d = v - base
    print(f"{n}: mean {v.mean():.1f}; difference vs Linear = {d.mean():+.1f} (SE {d.std(ddof=1)/np.sqrt(len(d)):.1f}); "
          f"Linear better in {(d > 0).sum()} of {len(d)} folds")

# 2. Bootstrap confidence intervals for the selected model on the test set.
lin = make_pipeline(LinearRegression(), True).fit(X_train, y_train)
pred = lin.predict(X_test)
rng = np.random.default_rng(SEED)
boot = []
yt, pt = y_test.to_numpy(), pred
for _ in range(2000):
    i = rng.integers(0, len(yt), len(yt))
    boot.append([mean_absolute_error(yt[i], pt[i]), np.sqrt(mean_squared_error(yt[i], pt[i])), r2_score(yt[i], pt[i])])
boot = np.array(boot)
print("\n=== Linear Regression test-set metrics with bootstrap 95% intervals (2,000 resamples) ===")
for k, (name, val) in enumerate([("MAE", mean_absolute_error(yt, pt)), ("RMSE", np.sqrt(mean_squared_error(yt, pt))), ("R2", r2_score(yt, pt))]):
    lo, hi = np.percentile(boot[:, k], [2.5, 97.5])
    print(f"  {name}: {val:,.3f}  (95% interval {lo:,.3f} to {hi:,.3f})")

# 3. Coefficients per natural unit.
names = lin.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lin.named_steps["model"].coef_, index=names)
sd = pd.Series(lin.named_steps["prep"].named_transformers_["num"].named_steps["scale"].scale_, index=NUMERIC)
per_unit = coef.copy()
for c in NUMERIC:
    per_unit[c] = coef[c] / sd[c]
units = {"Years_Experience": "per year of experience", "Job_Level": "per job level", "Hours_Per_Week": "per weekly hour",
         "Performance_Score": "per performance point", "Training_Hours": "per training hour",
         "Career_Start_Age": "per year older at career start", "Experience_per_Level": "per extra year of experience per level",
         "Education": "per education step (High School -> PhD)", "Department_HR": "HR vs Finance",
         "Department_IT": "IT vs Finance", "Department_Marketing": "Marketing vs Finance", "Department_Operations": "Operations vs Finance"}
print("\n=== Linear Regression coefficients in GHS per natural unit ===")
out = pd.DataFrame({"GHS": per_unit.round(1), "meaning": [units[c] for c in per_unit.index]}).sort_values("GHS", key=abs, ascending=False)
print(out.to_string())
out.to_csv(ROOT / "outputs" / "05_coefficients_per_unit.csv")

# 4. Error check by gender (gender is not used by the model).
g = df.loc[X_test.index, "Gender"]
err = pd.DataFrame({"gender": g, "abs_error": np.abs(yt - pt), "error": yt - pt})
print("\n=== Test-set errors by gender (gender is NOT a model input) ===")
print(err.groupby("gender").agg(n=("abs_error", "size"), MAE=("abs_error", "mean"), mean_error=("error", "mean")).round(0).to_string())

# 5. Final model on all 685 employees, used for the example prediction.
final = make_pipeline(LinearRegression(), True).fit(X, y)
new = pd.DataFrame([{"Age": 38, "Education": "Master's", "Years_Experience": 10, "Department": "IT", "Job_Level": 3,
                     "Hours_Per_Week": 42, "Performance_Score": 4.2, "Training_Hours": 35}])
p_train_only = lin.predict(new)[0]
p_final = final.predict(new)[0]
resid_sd = np.std(y - final.predict(X), ddof=1)
print("\n=== Example prediction ===")
print(f"Model fitted on training set only: GHS {p_train_only:,.2f}")
print(f"Final model refitted on all 685 employees: GHS {p_final:,.2f}")
q10, q90 = np.percentile(yt - pt, [10, 90])
print(f"Rough 80% range from test-set errors: GHS {p_final + q10:,.0f} to {p_final + q90:,.0f}")
contrib = pd.Series(final.named_steps["model"].coef_ * final.named_steps["prep"].transform(
    final.named_steps["features"].transform(final.named_steps["age_exp"].transform(new)))[0],
    index=final.named_steps["prep"].get_feature_names_out())
print(f"Intercept (average employee on scaled inputs): GHS {final.named_steps['model'].intercept_:,.0f}")
print("Contributions of each input relative to that average (GHS):")
print(contrib.round(0).sort_values(key=abs, ascending=False).to_string())
