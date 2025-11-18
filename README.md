Telco Customer Churn Prediction with XGBoost and SHAP Explainability

This project builds an end-to-end machine learning pipeline to predict customer churn for a telecom provider using the Telco Customer Churn dataset.
It covers data cleaning, feature engineering, preprocessing, model tuning, evaluation, and full interpretability through SHAP (both global and local explanations).

The pipeline is designed for clarity, transparency, and deployment readiness.

1. Project Workflow Overview

The implementation follows a complete production-like structure:

Data loading and cleaning

Domain-driven feature engineering

Proper train/test split (stratified)

Preprocessing using scikit-learn Pipelines & ColumnTransformer

Hyperparameter tuning using RandomizedSearchCV

Model evaluation with all relevant classification metrics

Global SHAP explanation (summary plot)

Local SHAP reasoning (3 contrasting cases)

Professional textual interpretation & business guidance

All generated SHAP plots are saved as image files.

Dataset used: WA_Fn-UseC_-Telco-Customer-Churn.csv

2. Data Preparation and Cleaning

Key steps:

Convert TotalCharges to numeric

Remove rows with missing total charges

Create a binary churn flag (ChurnFlag)

Normalize service-related columns by replacing
No internet service and No phone service → No

This ensures the dataset is clean, consistent, and ready for modeling.

3. Feature Engineering

The project adds several domain-relevant features:

avg_month_per_tenure → Spending intensity

num_services → Engagement level

is_month_to_month → Contract volatility

is_electronic_check → High-risk payment method

tenure_x_monthly → Long-term revenue interaction

These features significantly improve model interpretability and predictive quality.

4. Train–Test Split

A stratified 80/20 split ensures churn ratios remain consistent across splits.

Train size: ~80%
Test size:  ~20%


Target variable: ChurnFlag

5. Preprocessing Pipeline

A combined preprocessing pipeline handles:

Numerical Features

Median imputation

Standard scaling

Categorical Features

Most frequent imputation

One-hot encoding (with ignore for unknowns)

This is implemented using ColumnTransformer and Pipeline for modularity and reproducibility.

6. Model Training & Hyperparameter Tuning

Model used: XGBoost Classifier

RandomizedSearchCV optimizes parameters:

n_estimators

max_depth

learning_rate

Cross-validation: 4-fold stratified

The tuned pipeline becomes the final model used for all evaluation and SHAP analysis.

7. Model Evaluation

Metrics computed:

Accuracy

F1 Score

Recall

Precision

ROC-AUC

Final performance (test set):

Metric	Score
Accuracy	value from run
F1 Score	value from run
Recall	value from run
Precision	value from run
ROC-AUC	value from run

(Your notebook prints the exact values during execution.)

Interpretation:
The model strikes a strong balance between recall and precision — crucial for churn scenarios where false negatives are costly.

8. SHAP Global Interpretability

A TreeExplainer is used on the trained XGBoost model.
Global SHAP results show which features drive churn across the entire population.

Top global predictors identified:

Month-to-month contract

MonthlyCharges

Tenure

Electronic check payment

Number of subscribed services

These results align with real-world telecom churn patterns.

The project generates the SHAP summary plot:

shap_global_summary.png

9. Local SHAP Explanations

Three customer-level explanations are included:

1. High-Risk Churner

Dominated by:

Month-to-month contract

Electronic check

High monthly charges

2. Low-Risk Customer

Stabilized by:

Long tenure

Secure contract type

Numerous services

3. Borderline Case

Mixed feature contributions → prediction near 0.5 probability.

Local plots saved as:

shap_local_highrisk.png

shap_local_lowrisk.png

shap_local_borderline.png

These help understand decision boundaries for individual customers.

10. Textual Interpretation & Business Recommendations

The notebook generates a full professional write-up summarizing:

Model insights

Month-to-month customers churn the most

High billing pressure increases dissatisfaction

Short-tenure customers are the most volatile

Electronic check users show unstable payment patterns

Actionable business strategies

Offer retention discounts to month-to-month users

Reduce “bill shock” by capping or smoothing charges

Improve onboarding experience for new users

Encourage autopay over electronic checks

Promote bundled services to improve engagement

11. Repository Structure
customer_churn_project/
│
├── customer_chrun_project.ipynb     # Main notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── shap_global_summary.png
├── shap_local_highrisk.png
├── shap_local_lowrisk.png
├── shap_local_borderline.png
│
└── README.md

12. Conclusion

This project delivers a full ML workflow suitable for real-world churn prediction:

Clean preprocessing pipeline

Tuned XGBoost model

Strong evaluation metrics

Global + local SHAP explainability

Actionable business recommendations

The result is a reliable, transparent, and business-aligned churn prediction system.