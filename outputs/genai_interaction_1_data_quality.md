# GenAI Interaction 1 – Data-quality problems and preprocessing

**Tool:** Claude (Anthropic), Claude Code, model Claude Opus 5.5 – a fresh instance with no memory of my own analysis.
**Date:** 5 October 2026
**Data given:** a copy of employee_income.csv in a separate folder (bdat618_genai_test), so it could not see my cleaning work.

## Prompt (exact words)

> Data file: [local folder]\bdat618_genai_test\employee_income.csv (use only files in that folder).
>
> Act as a senior data analyst. I have an HR dataset, employee_income.csv, that I will use to build regression models predicting Monthly_Income (in GHS). Investigate the file and identify all the data-quality problems in it: missing values, duplicates, inconsistent categories, impossible or unrealistic values, outliers, incorrect data types and redundant variables. For each problem, give the count and the specific rows or values involved, and recommend exactly how I should handle it before modelling. Do not modify the file.

## AI response (verbatim)

## Data-quality audit: employee_income.csv

I only read the file and did not change it. It has 712 data rows and 11 columns, and every row parses cleanly. Line numbers below are file lines, with the header as line 1. I used Python's standard library because pandas is not installed. The scratch scripts are in the session scratchpad, not in your data folder.

**Recommended cleaning order:** remove duplicates → standardise categories → turn impossible values into NaN → drop rows with no target → split into train and test → impute (fitted on train only) → encode → model.

### 1. Duplicates: 12 extra rows
- Lines 702–713 are exact copies of 12 earlier rows, including the Employee_ID: EMP0112, 0151, 0199, 0207, 0210, 0231, 0310, 0384, 0422, 0444, 0599 and 0603.
- The IDs run EMP0001 to EMP0700 with no gaps, so the true dataset is 700 employees.
- Two of the copies also carry missing values (EMP0231 has no income; EMP0422 has no Education or Years_Experience), so they inflate the missing counts.
- **Action:** drop the exact duplicates and keep the first occurrence (`drop_duplicates()`), leaving 700 rows. Then check that Employee_ID is unique. Do this before anything else, or duplicates can land in both train and test.

### 2. Missing values: 84 blank cells, 79 rows affected
No other placeholders such as "NA" or "?" were found. Counts before de-duplication, with the count after in brackets:

| Column | Missing | Employee IDs |
|---|---|---|
| Age | 14 (14) | EMP0013, 0058, 0147, 0216, 0221, 0226, 0259, 0442, 0443, 0490, 0506, 0590, 0608, 0688 |
| Education | 13 (12) | EMP0034, 0093, 0121, 0127, 0157, 0181, 0218, 0321, 0422 (×2), 0477, 0645, 0675 |
| Years_Experience | 19 (18) | EMP0094, 0098, 0122, 0164, 0227, 0274, 0302, 0350, 0422 (×2), 0438, 0489, 0534, 0555, 0614, 0649, 0665, 0673, 0690 |
| Performance_Score | 15 (15) | EMP0005, 0068, 0071, 0091, 0094, 0123, 0202, 0217, 0336, 0337, 0360, 0471, 0486, 0512, 0652 |
| Training_Hours | 12 (12) | EMP0002, 0026, 0176, 0190, 0214, 0253, 0299, 0412, 0513, 0604, 0676, 0691 |
| **Monthly_Income (target)** | 11 (10) | EMP0094, 0231 (×2), 0245, 0285, 0315, 0379, 0429, 0568, 0660, 0665 |

- EMP0094 is missing 3 fields (Years_Experience, Performance_Score, income) and also has 100 hours a week.
- EMP0665 and EMP0422 are each missing 2 fields.
- After de-duplication, 623 of the 700 rows are complete.

**Action:**
- **Target:** drop the 10 rows with no Monthly_Income. Never impute the target.
- **Predictors:** impute after the train/test split, fitted on the training set only:
  - Age: median.
  - Years_Experience: median within Job_Level, or regress it on Age. Every valid row has Age − Experience between 20 and 28, so Age predicts it well.
  - Performance_Score and Training_Hours: median.
  - Education: mode ("Bachelor's") or a separate "Unknown" level.
- Optionally add missing-indicator flags.

### 3. Inconsistent categories
- **Gender: 8 rows use non-standard labels.**
  - "M" (2): EMP0076, EMP0618.
  - "F" (3): EMP0161, EMP0329, EMP0421.
  - "female" (3): EMP0178, EMP0299, EMP0619.
  - **Action:** map them to "Male" and "Female", leaving 2 levels.
- **Department: 8 rows use variant names.**
  - "Human Resources" (3): EMP0091, EMP0228, EMP0546 → HR.
  - "I.T." (2): EMP0179, EMP0456 → IT.
  - "Mktg" (2): EMP0335, EMP0626 → Marketing.
  - "finance" (1): EMP0625 → Finance.
  - **Action:** map them so there are 5 levels. Also trim spaces and fix case in general.
- **Education** has 5 consistent levels. It is ordinal (High School < Diploma < Bachelor's < Master's < PhD), so encode it as ordinal or one-hot. The apostrophes in "Bachelor's" and "Master's" are worth renaming to keep code simple.

### 4. Impossible or unrealistic values
- **Age: 5 impossible values.**
  - EMP0348 and EMP0617 are aged 5 but have 18 and 22 years of experience and job levels 4 and 3.
  - EMP0212 and EMP0220 are aged 120.
  - EMP0236 is aged 150.
  - All valid ages are 21–64. **Action:** set these 5 to NaN and impute them. Age − Experience is consistently 20–28, so Experience + 24 is a good estimate.
- **Hours_Per_Week: 5 impossible values.**
  - −10 for EMP0182 and EMP0612.
  - 140 for EMP0046 and EMP0230 (140 of the 168 hours in a week).
  - 100 for EMP0094.
  - All other values are 25–61. **Action:** set them to NaN and impute the median (42). EMP0094 is dropped anyway because its income is missing.
- **Age − Experience below 18:** only the two age-5 rows above. Fixing Age resolves them.
- **Low-priority plausibility flag:** 4 PhD holders are aged 21–22 (EMP0211, 0260, 0330, 0502). This is unlikely but not impossible. Note it and leave the rows in.
- **No problems found in:**
  - Performance_Score (1.3–5.0, consistent with a 1–5 scale).
  - Training_Hours (2–100).
  - Job_Level (only values 1–5).
  - Years_Experience (0–42).

### 5. Outliers
- **Monthly_Income: 5 extreme values.** The interquartile-range (IQR) upper fence is about 21,163 GHS. Every other value is between 4,937 and 20,613.

  | Employee | Recorded income (GHS) | Job level | Experience |
  |---|---|---|---|
  | EMP0395 | 73,003.04 | 1 | 2 yrs |
  | EMP0501 | 107,285.84 | 3 | — |
  | EMP0482 | 110,365.36 | 4 | — |
  | EMP0046 | 136,628.00 | 5 | — (also 140 hours) |
  | EMP0608 | 154,744.40 | — | — (Age also missing) |

  - Each is roughly 10 times the median for its job level (Level 1 ≈ 8.2k up to Level 5 ≈ 17.2k), and dividing by 10 gives a plausible figure. They look like data-entry errors with the decimal point shifted.
  - On these columns' own correlations, these rows and the impossible Age and Hours values (5 each) drag Age vs income from about 0.84 down to 0.29, and Job_Level vs income from 0.87 down to 0.35.
  - **Action:** confirm with HR if you can. Otherwise exclude these 5 rows from training, since the target is wrong and should not be imputed. Dividing by 10 is a fallback only if you document it as an assumption. If you keep all rows for any reason, model log(income) or use robust regression.
- **Statistically flagged but genuine (keep them):**
  - Age 64: EMP0070, 0081, 0270.
  - Experience 39–42: EMP0239, 0351, 0465, 0081, 0070, 0270.
  - Training_Hours 74–100: 21 rows, which are part of a right-skewed tail. Consider a log or square-root transform.
  - Hours of 25–27 and 56–61: 17 rows.
  - Performance 1.3 and 1.5: EMP0332, EMP0699.
- After cleaning, income is only mildly right-skewed, so check the model residuals and log the target if needed.

### 6. Incorrect data types
- **Stored as decimals but should be whole numbers:** Age, Years_Experience and Training_Hours are written as "49.0". Convert them to integers (nullable Int64) after imputation.
- **Job_Level** (1–5) is ordinal. Treat it as an ordered numeric variable, or one-hot encode it if the step from level to level isn't constant. Don't leave it ambiguous.
- **Gender, Education and Department** load as plain text. Convert them to category types and encode them: one-hot for Gender and Department, ordinal or one-hot for Education.
- **Employee_ID** is a text identifier. Use it only as an index.
- Performance_Score and Monthly_Income are correctly stored as decimals. No numeric column contains text.

### 7. Redundant variables
- **Employee_ID** is a unique identifier and tells the model nothing. Drop it as a predictor once duplicates are removed.
- **Age and Years_Experience** have a correlation of r = 0.95 on clean rows, and every valid row has Age − Experience between 20 and 28. Experience is essentially Age minus a start age, which causes severe multicollinearity.
  - Keep one of them. Years_Experience is preferred because its correlation with income is 0.87 against Age's 0.84.
  - Alternatively, keep both only with regularisation (Ridge or Lasso) or with a variance inflation factor (VIF) check.
- **Job_Level** also correlates with Experience (r = 0.81) and Age (r = 0.77). Check VIF before deciding whether to keep it alongside Experience.
- **Weak predictors (keep, but expect little):** Hours_Per_Week, Performance_Score and Training_Hours each correlate 0.04–0.11 with income on clean rows. In the clean-row medians, Gender and Department show almost no pay difference (about 12.0k–12.6k).

### Expected modelling set
Starting from 712 rows, removing 12 duplicates, 10 missing-target rows and 5 rows with corrupt income leaves about **685 rows**, with 10 impossible Age and Hours values imputed.

## My decision and justification

[I checked these recommendations against the data before accepting, modifying or rejecting them; the outcome is summarised in the README under "Use of AI".]
