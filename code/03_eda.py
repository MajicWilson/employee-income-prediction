"""Exploratory data analysis of Monthly_Income (cleaned data, 685 employees)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
df = pd.read_csv(ROOT / "data" / "employee_income_clean.csv")

# Palette (reference instance, light surface): one series = blue; ordered job levels = one-hue sequential ramp.
BLUE, SURFACE = "#2a78d6", "#fcfcfb"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LEVEL_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # Job level 1 -> 5
EDU_ORDER = ["High School", "Diploma", "Bachelor's", "Master's", "PhD"]
DEPT_ORDER = df.groupby("Department")["Monthly_Income"].median().sort_values().index.tolist()

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.labelsize": 10, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})
ghs = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    print(f"saved figures/{name}")

def boxplot_by(col, order, title, fname, xlabel):
    groups = [df.loc[df[col] == g, "Monthly_Income"].values for g in order]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bp = ax.boxplot(groups, patch_artist=True, widths=0.55, showfliers=True,
                    medianprops=dict(color=INK, linewidth=2),
                    boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE, linewidth=1.2),
                    whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor=BLUE, markeredgecolor=SURFACE, alpha=0.8))
    ax.set_xticks(range(1, len(order) + 1), [f"{g}\n(n={len(x)})" for g, x in zip(order, groups)])
    for i, x in enumerate(groups, start=1):
        ax.text(i, np.median(x), f" {np.median(x):,.0f}", va="bottom", ha="left", fontsize=8, color=INK2)
    ax.set_title(title, loc="left"); ax.set_xlabel(xlabel); ax.set_ylabel("Monthly income (GHS)")
    ax.yaxis.set_major_formatter(ghs); ax.grid(axis="x", visible=False)
    save(fig, fname)

# Figure 1: distribution of monthly income.
inc = df["Monthly_Income"]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(inc, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1)
ax.axvline(inc.median(), color=INK, linewidth=1.5, linestyle="--")
ax.axvline(inc.mean(), color=INK2, linewidth=1.5, linestyle=":")
ymax = ax.get_ylim()[1]
ax.text(inc.median(), ymax * 0.95, f"  median {inc.median():,.0f}", color=INK, fontsize=9, va="top")
ax.text(inc.mean(), ymax * 0.86, f"  mean {inc.mean():,.0f}", color=INK2, fontsize=9, va="top")
ax.set_title("Figure 1. Distribution of monthly income (n=685)", loc="left")
ax.set_xlabel("Monthly income (GHS)"); ax.set_ylabel("Number of employees")
ax.xaxis.set_major_formatter(ghs); ax.grid(axis="x", visible=False)
save(fig, "fig1_income_distribution.png")

# Figures 2-3: income by education and by department.
boxplot_by("Education", EDU_ORDER, "Figure 2. Monthly income by education level", "fig2_income_by_education.png",
           "Education (12 employees with missing education not shown)")
boxplot_by("Department", DEPT_ORDER, "Figure 3. Monthly income by department", "fig3_income_by_department.png",
           "Department (ordered by median income)")

# Figure 4: experience vs income, coloured by job level (sequential ramp), with overall trend line.
d4 = df.dropna(subset=["Years_Experience"])
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for lvl, color in zip(range(1, 6), LEVEL_RAMP):
    s = d4[d4["Job_Level"] == lvl]
    ax.scatter(s["Years_Experience"], s["Monthly_Income"], s=18, color=color, edgecolor=SURFACE,
               linewidth=0.6, label=f"Level {lvl} (n={len(s)})")
slope, intercept, r, p, _ = stats.linregress(d4["Years_Experience"], d4["Monthly_Income"])
xs = np.array([0, d4["Years_Experience"].max()])
ax.plot(xs, intercept + slope * xs, color=INK, linewidth=2)
ax.text(xs[1], intercept + slope * xs[1], f" r = {r:.2f}\n +{slope:,.0f} GHS/yr", color=INK, fontsize=9, va="center")
ax.set_title("Figure 4. Years of experience vs monthly income, by job level", loc="left")
ax.set_xlabel("Years of experience (16 missing not shown)"); ax.set_ylabel("Monthly income (GHS)")
ax.yaxis.set_major_formatter(ghs)
ax.legend(title="Job level", frameon=False, fontsize=8, title_fontsize=9, loc="upper left")
ax.set_xlim(-1, xs[1] + 7)
save(fig, "fig4_experience_vs_income.png")

# Figure 5: income by job level.
boxplot_by("Job_Level", [1, 2, 3, 4, 5], "Figure 5. Monthly income by job level", "fig5_income_by_job_level.png", "Job level")

# Figure 6: performance score vs income.
d6 = df.dropna(subset=["Performance_Score"])
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.scatter(d6["Performance_Score"], d6["Monthly_Income"], s=16, color=BLUE, edgecolor=SURFACE, linewidth=0.6, alpha=0.85)
slope6, int6, r6, p6, _ = stats.linregress(d6["Performance_Score"], d6["Monthly_Income"])
xs6 = np.array([d6["Performance_Score"].min(), d6["Performance_Score"].max()])
ax.plot(xs6, int6 + slope6 * xs6, color=INK, linewidth=2)
ax.text(xs6[1], int6 + slope6 * xs6[1], f" r = {r6:.2f}", color=INK, fontsize=9, va="bottom")
ax.set_title("Figure 6. Performance score vs monthly income", loc="left")
ax.set_xlabel("Performance score (14 missing not shown)"); ax.set_ylabel("Monthly income (GHS)")
ax.yaxis.set_major_formatter(ghs)
save(fig, "fig6_performance_vs_income.png")

# ---- Numbers behind the charts ----
print("\n=== Monthly income summary ===")
print(inc.describe().round(2).to_string())
print(f"skewness {inc.skew():.2f}; Shapiro-Wilk p = {stats.shapiro(inc).pvalue:.4f}")

def group_table(col, order):
    t = df.groupby(col)["Monthly_Income"].agg(["count", "mean", "median", "std", "min", "max"]).reindex(order).round(0)
    groups = [df.loc[df[col] == g, "Monthly_Income"] for g in order]
    kw = stats.kruskal(*groups); an = stats.f_oneway(*groups)
    print(t.to_string())
    print(f"Kruskal-Wallis H = {kw.statistic:.2f}, p = {kw.pvalue:.3g};  one-way ANOVA F = {an.statistic:.2f}, p = {an.pvalue:.3g}")

for col, order in [("Education", EDU_ORDER), ("Department", DEPT_ORDER), ("Job_Level", [1, 2, 3, 4, 5]), ("Gender", ["Female", "Male"])]:
    print(f"\n=== Income by {col} ===")
    group_table(col, order)

print("\n=== Experience and performance vs income ===")
print(f"Experience: Pearson r = {r:.3f} (p = {p:.3g}); slope = {slope:,.1f} GHS per year; n = {len(d4)}")
print(f"Performance: Pearson r = {r6:.3f} (p = {p6:.3g}); slope = {slope6:,.1f} GHS per point; n = {len(d6)}")
rs = stats.spearmanr(d6["Performance_Score"], d6["Monthly_Income"])
print(f"Performance: Spearman rho = {rs.statistic:.3f} (p = {rs.pvalue:.3g})")

print("\n=== Education within job level: does education add anything once job level is known? (median income) ===")
print(df.pivot_table(index="Job_Level", columns="Education", values="Monthly_Income", aggfunc="median").reindex(columns=EDU_ORDER).round(0).to_string())
print("\n=== Education mix by job level (counts) ===")
print(pd.crosstab(df["Job_Level"], df["Education"]).reindex(columns=EDU_ORDER).to_string())

print("\n=== Correlation matrix (pairwise complete) ===")
num = ["Age", "Years_Experience", "Job_Level", "Hours_Per_Week", "Performance_Score", "Training_Hours", "Monthly_Income"]
corr = df[num].corr().round(2)
print(corr.to_string())
corr.to_csv(ROOT / "outputs" / "03_correlations.csv")
