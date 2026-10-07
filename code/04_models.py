"""Feature engineering, model development, evaluation and selection, and an example prediction.

All imputation, encoding and scaling happen inside scikit-learn pipelines fitted on the training set only.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
SEED = 42
df = pd.read_csv(ROOT / "data" / "employee_income_clean.csv")

EDU_ORDER = ["High School", "Diploma", "Bachelor's", "Master's", "PhD"]
TARGET = "Monthly_Income"
# Employee_ID is an identifier and is never a predictor.
# Gender is excluded: it shows no relationship with income (Kruskal-Wallis p = 0.75; adjusted p = 0.82) and using it
# to set pay predictions raises fairness concerns (checked against the data).
RAW_PREDICTORS = ["Age", "Education", "Years_Experience", "Department", "Job_Level",
                  "Hours_Per_Week", "Performance_Score", "Training_Hours"]


# ---------------- Feature engineering ----------------
class AgeExperienceImputer(BaseEstimator, TransformerMixin):
    """Fill a missing Age or Years_Experience from the other one, using the training median of
    (Age - Years_Experience), i.e. the typical age at which employees started work."""
    def fit(self, X, y=None):
        both = X[["Age", "Years_Experience"]].dropna()
        self.start_age_ = float((both["Age"] - both["Years_Experience"]).median())
        self.age_median_ = float(X["Age"].median())
        return self

    def transform(self, X):
        X = X.copy()
        X["Age"] = X["Age"].astype(float)
        X["Years_Experience"] = X["Years_Experience"].astype(float)
        a, e = X["Age"].isna(), X["Years_Experience"].isna()
        X.loc[a & ~e, "Age"] = X.loc[a & ~e, "Years_Experience"] + self.start_age_
        X.loc[e & ~a, "Years_Experience"] = (X.loc[e & ~a, "Age"] - self.start_age_).clip(lower=0)
        both = a & e
        X.loc[both, "Age"] = self.age_median_
        X.loc[both, "Years_Experience"] = max(self.age_median_ - self.start_age_, 0)
        return X


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Engineered features:
    - Career_Start_Age = Age - Years_Experience. Keeps the part of Age that experience does not already
      explain, so Age itself can be dropped (Age and experience correlate at 0.96).
    - Experience_per_Level = Years_Experience / Job_Level. Years of experience per grade: a high value
      means slower progression relative to experience.
    (A third candidate, Experience_x_Level, was tested and removed: it added no accuracy, its interaction
    term was not significant, and it pushed the variance inflation factor of Years_Experience to 46.)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Career_Start_Age"] = X["Age"] - X["Years_Experience"]
        X["Experience_per_Level"] = X["Years_Experience"] / X["Job_Level"].astype(float)
        return X.drop(columns=["Age"])


NUMERIC = ["Years_Experience", "Job_Level", "Hours_Per_Week", "Performance_Score", "Training_Hours",
           "Career_Start_Age", "Experience_per_Level"]


def make_pipeline(model, scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("edu", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ord", OrdinalEncoder(categories=[EDU_ORDER]))]), ["Education"]),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["Department"]),
    ], verbose_feature_names_out=False)
    return Pipeline([("age_exp", AgeExperienceImputer()), ("features", FeatureBuilder()), ("prep", pre), ("model", model)])


# ---------------- Split and models ----------------
X = df[RAW_PREDICTORS].copy()
for c in ["Education", "Department"]:
    X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
y = df[TARGET].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=df["Job_Level"])
print(f"Training set: {len(X_train)} employees; test set: {len(X_test)} employees (stratified by Job_Level, seed {SEED})")

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)            # for tuning settings
cv_eval = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)  # 15 fits per model, for comparing models
candidates = {
    "Baseline (training mean)": (DummyRegressor(strategy="mean"), False, None),
    "Linear Regression": (LinearRegression(), True, None),
    "Ridge Regression": (Ridge(), True, {"model__alpha": [0.1, 1, 3, 10, 30, 100]}),
    "Decision Tree": (DecisionTreeRegressor(random_state=SEED), False,
                      {"model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [5, 10, 20]}),
    "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1), False,
                      {"model__n_estimators": [300], "model__max_depth": [None, 8, 12],
                       "model__min_samples_leaf": [1, 3, 5], "model__max_features": [0.5, 1.0]}),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED), False,
                          {"model__n_estimators": [200, 400], "model__learning_rate": [0.03, 0.1],
                           "model__max_depth": [2, 3], "model__subsample": [0.8, 1.0]}),
}

def metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}

rows, fitted = [], {}
for name, (model, scale, grid) in candidates.items():
    pipe = make_pipeline(model, scale)
    if grid:
        search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train)
        pipe, best = search.best_estimator_, search.best_params_
    else:
        pipe.fit(X_train, y_train); best = {}
    # Repeated 5-fold cross-validation (3 repeats) on the training set with the chosen settings (stability check).
    cvres = cross_validate(make_pipeline(model, scale).set_params(**best), X_train, y_train, cv=cv_eval,
                           scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"})
    tr, te = metrics(y_train, pipe.predict(X_train)), metrics(y_test, pipe.predict(X_test))
    rows.append({"Model": name, "Settings": json.dumps({k.replace("model__", ""): v for k, v in best.items()}) if best else "-",
                 "CV_MAE": -cvres["test_MAE"].mean(), "CV_RMSE": -cvres["test_RMSE"].mean(), "CV_RMSE_sd": cvres["test_RMSE"].std(),
                 "CV_R2": cvres["test_R2"].mean(), "Train_RMSE": tr["RMSE"], "Train_R2": tr["R2"],
                 "Test_MAE": te["MAE"], "Test_RMSE": te["RMSE"], "Test_R2": te["R2"]})
    fitted[name] = pipe

res = pd.DataFrame(rows)
res.to_csv(OUT / "04_model_comparison.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== Model comparison (GHS; CV = repeated 5-fold cross-validation, 3 repeats, on the training set) ===")
print(res.round({"CV_MAE": 0, "CV_RMSE": 0, "CV_RMSE_sd": 0, "CV_R2": 3, "Train_RMSE": 0, "Train_R2": 3,
                 "Test_MAE": 0, "Test_RMSE": 0, "Test_R2": 3}).to_string(index=False))

# ---------------- Interpretable model: coefficients ----------------
lin = fitted["Linear Regression"]
names = lin.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lin.named_steps["model"].coef_, index=names).sort_values(key=abs, ascending=False)
print(f"\n=== Linear Regression coefficients (GHS change per 1 standard deviation for numeric features; "
      f"per category vs reference for one-hot; per education step for Education) ===\nIntercept {lin.named_steps['model'].intercept_:,.0f}")
print(coef.round(1).to_string())
coef.to_csv(OUT / "04_linear_coefficients.csv", header=["coefficient"])
num_sd = lin.named_steps["prep"].named_transformers_["num"].named_steps["scale"].scale_
print("\nStandard deviations used for scaling (training set):")
print(pd.Series(num_sd, index=NUMERIC).round(2).to_string())

# Variance inflation factors on the training design (numeric + education), to check multicollinearity.
from numpy.linalg import lstsq
Xt = pd.DataFrame(lin.named_steps["prep"].transform(lin.named_steps["features"].transform(lin.named_steps["age_exp"].transform(X_train))), columns=names)
vif = {}
for c in Xt.columns:
    others = Xt.drop(columns=c).assign(const=1.0)
    beta, *_ = lstsq(others.values, Xt[c].values, rcond=None)
    r2 = 1 - ((Xt[c] - others.values @ beta) ** 2).sum() / ((Xt[c] - Xt[c].mean()) ** 2).sum()
    vif[c] = 1 / (1 - r2) if r2 < 1 else np.inf
print("\n=== Variance inflation factors (linear model design) ===")
print(pd.Series(vif).round(2).sort_values(ascending=False).to_string())

# ---------------- Permutation importance on the test set ----------------
print("\n=== Permutation importance on the test set (increase in RMSE when a predictor is shuffled) ===")
imp_rows = {}
for name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
    pi = permutation_importance(fitted[name], X_test, y_test, n_repeats=30, random_state=SEED,
                                scoring="neg_root_mean_squared_error")
    imp_rows[name] = pd.Series(pi.importances_mean, index=X_test.columns)
imp = pd.DataFrame(imp_rows).sort_values("Gradient Boosting", ascending=False)
print(imp.round(0).to_string())
imp.to_csv(OUT / "04_permutation_importance.csv")

# ---------------- Residual check for the main candidates ----------------
print("\n=== Test-set residuals by job level (mean error, GHS; positive = model under-predicts) ===")
for name in ["Linear Regression", "Gradient Boosting", "Random Forest"]:
    r = (y_test - fitted[name].predict(X_test)).groupby(X_test["Job_Level"]).agg(["mean", "count"]).round(0)
    print(f"-- {name}\n{r.T.to_string()}")

# ---------------- Example prediction ----------------
new = pd.DataFrame([{"Age": 38, "Gender": "Female", "Education": "Master's", "Years_Experience": 10,
                     "Department": "IT", "Job_Level": 3, "Hours_Per_Week": 42, "Performance_Score": 4.2,
                     "Training_Hours": 35}])
print("\n=== Example prediction for one employee profile (GHS) ===")
preds = {name: float(p.predict(new)[0]) for name, p in fitted.items()}
for k, v in preds.items():
    print(f"  {k:26s} {v:>10,.2f}")
json.dump(preds, open(OUT / "04_example_predictions.json", "w"), indent=2)

# Similar employees in the data, for the reasonableness check.
sim = df[(df["Job_Level"] == 3) & df["Years_Experience"].between(8, 12)]
print(f"\nLevel 3, 8-12 years' experience: n={len(sim)}, median {sim[TARGET].median():,.0f}, "
      f"IQR {sim[TARGET].quantile(.25):,.0f}-{sim[TARGET].quantile(.75):,.0f}, range {sim[TARGET].min():,.0f}-{sim[TARGET].max():,.0f}")
sim2 = sim[sim["Education"] == "Master's"]
print(f"...and Master's: n={len(sim2)}, median {sim2[TARGET].median():,.0f}, range {sim2[TARGET].min():,.0f}-{sim2[TARGET].max():,.0f}")
sim3 = df[(df["Job_Level"] == 3) & (df["Department"] == "IT")]
print(f"Level 3 in IT: n={len(sim3)}, median {sim3[TARGET].median():,.0f}")
start = (df["Age"] - df["Years_Experience"]).dropna()
print(f"Career start age (Age - Experience) in the data: median {start.median():.0f}, range {start.min():.0f}-{start.max():.0f}; "
      f"this employee: {38 - 10}; share of employees with start age >= 28: {(start >= 28).mean():.1%}")
lvl3 = df[df["Job_Level"] == 3]["Years_Experience"].dropna()
print(f"Experience among level-3 employees: median {lvl3.median():.0f}, IQR {lvl3.quantile(.25):.0f}-{lvl3.quantile(.75):.0f}")

# Spread of plausible predictions: test-set errors of the selected model type near this income.
for name in ["Gradient Boosting", "Linear Regression"]:
    err = y_test - fitted[name].predict(X_test)
    print(f"{name}: test errors - 80% of employees within +/- {np.percentile(np.abs(err), 80):,.0f} GHS")

# ---------------- Figures ----------------
SURFACE, INK, INK2, MUTED, GRID, AXIS, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
ghs = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")

# Figure 7: model comparison (test RMSE with CV RMSE +/- sd).
m = res[res["Model"] != "Baseline (training mean)"].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.5, 4.3))
yy = np.arange(len(m))
ax.barh(yy - 0.18, m["CV_RMSE"], height=0.34, color="#86b6ef", label="Cross-validation RMSE (training set, mean ± SD of 15 folds)",
        xerr=m["CV_RMSE_sd"], error_kw=dict(ecolor=INK2, lw=1, capsize=3))
ax.barh(yy + 0.18, m["Test_RMSE"], height=0.34, color=BLUE, label="Test-set RMSE")
for i, r in m.iterrows():
    ax.text(r["Test_RMSE"] + 15, i + 0.18, f"{r['Test_RMSE']:,.0f}  (R² {r['Test_R2']:.3f})", va="center", fontsize=8, color=INK2)
ax.set_yticks(yy, m["Model"]); ax.invert_yaxis()
ax.set_xlabel("RMSE (GHS) – lower is better"); ax.xaxis.set_major_formatter(ghs)
ax.set_title("Figure 7. Model comparison: prediction error", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); ax.grid(axis="y", visible=False)
ax.set_xlim(0, m[["CV_RMSE", "Test_RMSE"]].max().max() * 1.35)
fig.tight_layout(); fig.savefig(FIG / "fig7_model_comparison.png", dpi=200); plt.close(fig)

# Figure 8: predicted vs actual (test set) for the two leading models.
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), sharex=True, sharey=True)
for ax, name in zip(axes, ["Linear Regression", "Gradient Boosting"]):
    p = fitted[name].predict(X_test)
    ax.scatter(y_test, p, s=14, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    lims = [min(y_test.min(), p.min()) - 300, max(y_test.max(), p.max()) + 300]
    ax.plot(lims, lims, color=INK, lw=1.2, ls="--")
    ax.set_title(name, loc="left", fontsize=11)
    ax.set_xlabel("Actual monthly income (GHS)"); ax.xaxis.set_major_formatter(ghs); ax.yaxis.set_major_formatter(ghs)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4000))
axes[0].set_ylabel("Predicted monthly income (GHS)")
fig.suptitle("Figure 8. Predicted vs actual income on the test set (n=137); dashed line = perfect prediction",
             x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "fig8_predicted_vs_actual.png", dpi=200); plt.close(fig)

# Figure 9: permutation importance (Gradient Boosting and Linear Regression).
top = imp[["Linear Regression", "Gradient Boosting"]].sort_values("Gradient Boosting")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
yy = np.arange(len(top))
ax.barh(yy - 0.18, top["Linear Regression"], height=0.34, color=BLUE, label="Linear Regression")
ax.barh(yy + 0.18, top["Gradient Boosting"], height=0.34, color=ORANGE, label="Gradient Boosting")
ax.set_yticks(yy, [c.replace("_", " ") for c in top.index])
ax.set_xlabel("Increase in test RMSE when the predictor is shuffled (GHS)")
ax.set_title("Figure 9. Predictor importance (permutation, test set)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "fig9_permutation_importance.png", dpi=200); plt.close(fig)
print("\nsaved figures/fig7_model_comparison.png, fig8_predicted_vs_actual.png, fig9_permutation_importance.png")
