# GenAI Interaction 2 – Feature engineering and preprocessing for machine learning

**Tool:** Claude (Anthropic), Claude Code, model Claude Opus 5.5 – a fresh instance with no memory of my own analysis.
**Date:** 5 October 2026
**Data given:** a copy of my cleaned file (employee_income_clean.csv) in a separate folder.

## Prompt (exact words)

> Data file: [local folder]\bdat618_genai_test\employee_income_clean.csv (use only files in that folder).
>
> Act as a machine learning engineer. This is my cleaned HR dataset (685 employees) for predicting Monthly_Income in GHS with regression models. Duplicates, inconsistent categories and impossible values have already been dealt with; a few predictor cells are still blank and will be imputed inside the modelling pipeline. Columns: Employee_ID, Age, Gender, Education, Years_Experience, Department, Job_Level, Hours_Per_Week, Performance_Score, Training_Hours, Monthly_Income.
>
> Tell me how to prepare it for machine learning: (1) how to encode each categorical variable and why; (2) which numerical variables to scale and for which kinds of model; (3) which variables should not be used as predictors and why; (4) at least two new features I could construct from the existing variables, with the reasoning for each. Check your suggestions against the data and show any numbers you rely on. Do not modify the file.

## AI response (verbatim)

I checked every suggestion against `employee_income_clean.csv` and did not modify the file; all analysis was read-only. The file has 685 rows, 11 columns, no duplicate IDs, and Monthly_Income is always present. 74 rows have at least one blank predictor: Age 18, Years_Experience 16, Performance_Score 14, Education 12, Training_Hours 12, Hours_Per_Week 3. No row is missing both Age and Years_Experience.

The model results below come from 5-fold cross-validation (shuffled, random_state 42) on a scikit-learn pipeline: median imputation, scaling, and the encodings in section 1. Linear regression reached R² = 0.913 (RMSE about GHS 908); a random forest reached R² = 0.892 (RMSE about GHS 1,009). Statsmodels results drop incomplete rows (about 611 rows used). I ran Python from `~/anaconda3/python.exe` because the default `py` has no pandas.

### 1. Encoding the categorical variables

**Education: ordinal encoding** in the order High School=0, Diploma=1, Bachelor's=2, Master's=3, PhD=4.
- It has a natural order, and mean income rises steadily: 11,077 / 11,383 / 12,064 / 13,165 / 14,214 (counts 72 / 92 / 265 / 170 / 74).
- Spearman correlation with income is 0.288. In a multiple regression each step adds about GHS 747 (p < 0.001).
- Pass the order explicitly to `OrdinalEncoder(categories=[...])`; the default alphabetical order would be wrong.
- Impute its 12 blanks with the most frequent value inside the pipeline. One-hot is a fallback if you don't want to assume equal steps.

**Department: one-hot encoding** with `drop='first'` for linear models and `handle_unknown='ignore'`.
- It has 5 balanced levels (HR 144, Finance 141, Operations 140, Marketing 131, IT 129) and no natural order, so ordinal codes would create a fake ranking.
- Holding other variables fixed, HR, Marketing and Operations earn about GHS 325–395 less than Finance (p ≤ 0.003). IT is not significantly different (+131, p = 0.25).
- Tree models can use the full one-hot set.

**Gender: one binary column** (one-hot with one level dropped).
- It is two levels with almost the same income: Female 12,435 vs Male 12,325, adjusted difference −16 (p = 0.82).
- It adds nothing to prediction, so I'd leave it out (see section 3).

**Job_Level: keep as the integer 1–5** even though it is really an ordered category.
- Adjusted income steps between levels are near-equal: +1,300, +1,107, +1,111, +1,205.
- Treating it as linear gives adjusted R² 0.9192 versus 0.9190 for one-hot, so the integer is the simpler and equally good choice.

### 2. Scaling numerical variables

**Which columns:** Age, Years_Experience, Hours_Per_Week, Performance_Score, Training_Hours, the Education and Job_Level codes, and any engineered features.
- Their ranges differ a lot: Performance_Score 1.3–5, Hours 25–61, Training 2–100.
- Fit `StandardScaler` inside the pipeline after imputation, so it only learns from the training folds.

**Models that need it:**
- Ridge, Lasso and ElasticNet: the penalty depends on each variable's scale.
- KNN and SVR: they use distances.
- Neural networks: they train poorly on unscaled inputs.

**Models that don't:** plain least-squares linear regression (predictions are unaffected, though scaling makes coefficients comparable), and decision trees, random forests and gradient boosting.

**Training_Hours:** it is the only clearly skewed column (skew 1.13; 11 values at 80 or more). A `log1p` transform brings skew to −0.48. Use `RobustScaler` or the log if you're worried about the long tail. In my tests the log changed cross-validated R² only from 0.9130 to 0.9131.

**Target:** income is close to symmetric (skew 0.10), so it doesn't need transforming.

### 3. Variables that should not be predictors

- **Employee_ID:** a unique label (EMP0001–EMP0685) with no meaning. Its correlation with income is −0.03; a model would only memorise it.
- **Monthly_Income:** the target. Also avoid any feature built from it, such as hourly pay (income divided by hours), or department or level mean income computed on the whole dataset. These leak the answer.
- **Gender:**
  - It has no predictive value (p = 0.82, raw gap GHS 110).
  - Using it in pay predictions raises fairness and anti-discrimination concerns.
  - Keep it only to audit the model's errors by gender afterwards.
- **Age and Years_Experience together:**
  - Their correlation is 0.955, with variance inflation factors of 11.6 and 13.5 (above the usual limit of 10).
  - With both in, Age adds nothing once experience is known (coefficient +25, p = 0.088), while Years_Experience is strong (+153 per year, p < 0.001).
  - For linear models, keep Years_Experience and drop Age (or use Ridge). Tree models tolerate both.
- **Weak predictors (keep, but don't expect much):**
  - Hours_Per_Week: correlation 0.036, p = 0.95.
  - Training_Hours: correlation 0.056, p = 0.79.
  - Keep them as candidates and let regularisation or feature importance decide. Note that Performance_Score is significant (+444 per point, p < 0.001) even though its raw correlation is only 0.11.

### 4. New features

1. **Career_Start_Age = Age − Years_Experience**
   - Values are tightly bunched: mean 23.8, SD 2.56, range 20–28, none below 18. This also confirms the two columns are consistent.
   - It removes the overlap between Age and experience and keeps a separate signal about late starters (correlation with income −0.146).
   - Its biggest use is imputation. Because Age ≈ Years_Experience + 23.8, use `IterativeImputer`, or "the other column ± 23.8", to fill the 18 blank Ages and 16 blank experience values. No row lacks both, so this is far better than the median.
   - Education doesn't explain it (correlation 0.025; group means 23.1–24.4).
2. **Experience_per_Level = Years_Experience / Job_Level**
   - It measures how quickly someone has been promoted: two people at level 3 with 5 versus 20 years' experience are in different positions within the level's pay band.
   - Correlation with income is 0.394. Cross-validated linear R² went from 0.9130 to 0.9131.
3. **Log_Training = log1p(Training_Hours)**: fixes the skew noted in section 2.
4. **Overtime flag = Hours_Per_Week > 40**: 58.9% of staff qualify, but mean income is 12,409 vs 12,331, so it is useful mainly for interpretation.
5. **Interactions:**
   - Years_Experience × Job_Level has correlation 0.877 on its own, but the interaction term is not significant (p = 0.58).
   - Education × Job_Level has p = 0.23. An experience-squared term has p = 0.86.

**These features barely change accuracy.** The relationships are already almost linear, and Job_Level, Years_Experience and Education carry most of the signal. Every feature above moved cross-validated R² by no more than ±0.002. Adding Career_Start_Age and Experience×Level to the random forest slightly lowered it (0.892 to 0.888). Their value is in better imputation, less multicollinearity and easier interpretation. Keep any of them only if they improve validation scores.

### Recommended pipeline
- **Numeric columns:** IterativeImputer (or median), then StandardScaler.
- **Education:** most-frequent imputer, then OrdinalEncoder with the order above.
- **Department:** OneHotEncoder.
- **Excluded:** Employee_ID and Gender.
- **Linear models:** use Years_Experience, or Career_Start_Age, instead of Age.
- **Splitting:** split into train and test before fitting anything.

## My decision and justification

[I checked these recommendations against the data before accepting, modifying or rejecting them; the outcome is summarised in the README under "Use of AI".]
