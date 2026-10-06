# GenAI Interaction 3 – Model selection, data splitting and choosing the best model

**Tool:** Claude (Anthropic), Claude Code, model Claude Opus 5.5 – a fresh instance with no memory of my own analysis.
**Date:** 5 October 2026
**Data given:** a copy of my cleaned file (employee_income_clean.csv) in a separate folder.

## Prompt (exact words)

> Data file: [local folder]\bdat618_genai_test\employee_income_clean.csv (use only files in that folder).
>
> Act as a senior data scientist. I need to predict Monthly_Income (GHS) for employees from this cleaned HR dataset (685 rows; predictors: Age, Gender, Education, Years_Experience, Department, Job_Level, Hours_Per_Week, Performance_Score, Training_Hours). I must develop and compare at least three regression models, at least one of which is reasonably interpretable, and evaluate them with MAE, RMSE and R².
>
> Recommend which three or four algorithms I should use and why, how I should split the data into training and test sets, and how I should choose the best model without simply picking the one with the highest R². Base your advice on the characteristics of this dataset (look at it) rather than general rules, and show any numbers you rely on. Do not modify the file and do not build the final models for me.

## AI response (verbatim)

I looked at `employee_income_clean.csv` (685 rows) using Anaconda Python. The file was only read, never changed. The diagnostic scripts are in my scratchpad, not in your project folder.

The main finding is that **income here is almost completely linear and additive**. That should drive which models you pick, how you split, and how you choose the winner.

### 1. What the data shows

- **The "clean" file still has missing values.** Age is missing in 18 rows, Years_Experience 16, Performance_Score 14, Education 12, Training_Hours 12 and Hours_Per_Week 3. That is 74 rows (10.8%) with at least one gap. Do not drop them. Impute inside the model pipeline, fitted on the training data only.
- **The target behaves well.** Monthly_Income runs from 4,937 to 20,613 GHS, with mean 12,379 and SD 3,108. Skew is 0.10 and there are no IQR outliers. Taking the log makes skew worse (-0.52) and lowers adjusted R² of a linear fit from 0.922 to 0.900. So model income in GHS directly, with no transform.
- **Three predictors dominate.** Correlations with income:

  | Predictor | Correlation with income |
  |---|---|
  | Job_Level | 0.869 |
  | Years_Experience | 0.868 |
  | Age | 0.840 |
  | Performance_Score | 0.112 |
  | Training_Hours | 0.056 |
  | Hours_Per_Week | 0.036 |

  Average income by job level is 8,312 / 10,810 / 12,687 / 15,015 / 17,448 GHS for levels 1 to 5, with only about 1,500 GHS spread within each level.
- **Age and experience say almost the same thing.** Their correlation is 0.955, and Age minus Experience always falls between 20 and 28. In a linear model their variance inflation factors are 12.2 and 14.0.
- **No curves or interactions are worth adding.** These checks used a linear fit on all complete rows (611):
  - All terms together give R² = 0.924 (adjusted 0.922).
  - Squared terms for experience, hours and training add nothing (adjusted R² 0.9215 vs 0.9219).
  - Department × Job_Level interaction: p = 0.083.
  - Experience and Performance × Job_Level interactions: p = 0.45.
  - Treating Job_Level as a number instead of categories loses nothing (p = 0.69). Its steps are roughly equal at about 1,100–1,300 GHS each.
  - Residuals are close to normal: skew 0.07 and 3 points beyond |3|. The test for unequal error spread gives p = 0.08, so that is not a problem.
- **Some predictors carry no signal once others are controlled.** Gender (p = 0.82), Hours_Per_Week (p = 0.86) and Training_Hours (p = 0.65) contribute nothing. Training_Hours is also right-skewed (skew 1.13, max 100).
- **Group sizes are fine for a split.** Every Department, Education level and Job_Level has at least 63 rows. The smallest groups are Job_Level 5 (63) and PhD (74).

### 2. Recommended algorithms

1. **Multiple linear regression (OLS), the interpretable model.** The data is linear and additive, so this is probably close to the best you can do, not just a baseline. Each coefficient reads as GHS per unit. For example, about +446 GHS per performance point, +1,856 GHS for a PhD vs a Bachelor's, and +4,703 GHS for level 5 vs level 1. Because Age and Experience overlap so much, report Experience and either drop Age or say clearly that their separate coefficients are unstable.
2. **Ridge or Lasso regression (regularised linear).** This handles the Age/Experience overlap without you choosing by hand. Lasso is likely to shrink Gender, Hours and Training to about zero, which matches the findings above. Tune the penalty strength by cross-validation on the training set only, and standardise the numeric columns.
3. **Random Forest.** A non-linear benchmark that copes with mixed data types and needs little tuning. Its job is to test whether there is any hidden non-linearity. I expect it to score at or below OLS, because tree-based models fit smooth straight-line trends in steps, and with only about 550 training rows that costs accuracy. That would itself be useful evidence for the write-up.
4. **Gradient boosting (sklearn's GradientBoosting / HistGradientBoosting, or XGBoost), optional fourth.** This is the strongest general non-linear learner. Tune learning rate, depth and number of trees by cross-validation. If it only matches OLS, that confirms the linear structure.

A shallow decision tree (depth 3–4) is a weaker interpretable alternative, but on this data it will clearly underperform OLS.

### 3. How to split the data

- **Hold out 20% as a test set** (about 137 rows test, 548 train) with a fixed random seed. Stratify on Job_Level so all five levels, especially the 63 level-5 rows, appear in both sets in proportion.
- **Choose settings only within the training set**, using 5-fold cross-validation repeated 3–5 times. Put imputation (median for numeric, most-frequent for categorical), one-hot encoding and scaling inside the pipeline so nothing from the test rows leaks into training.
- **Use the test set once**, for final reporting.
- **One split is noisy.** I fitted a plain OLS pipeline on 200 different random 80/20 splits, for diagnosis only. The test-set scores varied this much:

  | Metric | Mean | SD | Range |
  |---|---|---|---|
  | MAE (GHS) | 702 | 43 | 584–817 |
  | RMSE (GHS) | 904 | 54 | 751–1,083 |
  | R² | 0.915 | 0.010 | 0.872–0.940 |

  Repeated 5×5 cross-validation gave R² 0.914 ± 0.013 and RMSE 901 ± 55 GHS. So a gap in test R² of about 0.01, or in RMSE of about 50 GHS, is within noise.

### 4. Choosing the best model without just taking the highest R²

1. **Rank on cross-validated RMSE and MAE in GHS, with their spread across folds.** Use the same folds for every model and compare per-fold differences. All models will probably sit near R² 0.91–0.92, so R² won't separate them, while errors in GHS are what HR can act on. A baseline MAE of about 700 GHS is about 5.7% of mean income.
2. **Apply a one-standard-error rule.** Treat any model whose cross-validated RMSE is within one SE of the best (roughly 55 / √25 ≈ 11 GHS for the fold mean, or use the paired fold differences) as tied. Among tied models, pick the simplest and most interpretable. Given the linearity evidence, that will very likely be OLS or Ridge/Lasso.
3. **Check for overfitting.** Compare training and CV error. Random Forest will show a much lower training error; a large gap is a mark against it.
4. **Inspect residuals on the test set.** Plot errors against predictions and look at MAE by Job_Level and Department. Check whether any model is systematically wrong for level 5 or level 1 staff; trees tend to pull extreme salaries toward the middle.
5. **Weigh interpretability, stability and fairness.**
   - Check that coefficients and feature importances stay consistent across folds.
   - Gender has no effect on income here (−17 GHS, p = 0.82). Consider leaving it out of the final model; if you keep it, show that it doesn't change predictions.
   - Report the chosen model's test MAE, RMSE and R² once, with a bootstrap confidence interval.

## My decision and justification

[I checked these recommendations against the data before accepting, modifying or rejecting them; the outcome is summarised in the README under "Use of AI".]
