# Credit-Default-Payments

This project focuses on predicting the likelihood of credit default payments using a supervised machine learning approach. The dataset used is the UCI Credit Card dataset, which includes demographic information, credit history, payment behavior, and billing data for credit card clients.

# Key Features of the Code:
# Exploratory Data Analysis (EDA):

Boxplots and correlation heatmaps are generated to visualize the relationships between features such as credit limit, payment amounts, and demographic factors.
Identified trends and outliers in the data using visualizations.
# Data Preprocessing:

Missing values are analyzed and handled.
StandardScaler is used to standardize numerical features for consistency.
Principal Component Analysis (PCA) is applied to reduce dimensionality, retaining 95% of the variance.
Variance Inflation Factor (VIF):

VIF is calculated to detect multicollinearity and optimize the feature set.
Model Development and Evaluation:

# Various machine learning models are implemented, including:
Random Forest Classifier
AdaBoost Classifier
XGBoost
LightGBM
The models are evaluated on the test set for accuracy and performance, with LightGBM yielding the best results.
Prediction Pipeline:

User inputs are standardized and transformed using PCA before being passed to the trained LightGBM model.
Outputs whether the user is eligible or not for credit based on the prediction.
# User Interaction:
The code includes a user-friendly input mechanism to allow dynamic testing. Users can input demographic and financial details to receive real-time predictions.

# Tools and Libraries:
Python Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, lightgbm, xgboost
Metrics: Accuracy and ROC-AUC for evaluating model performance.
This project demonstrates a robust machine learning pipeline, combining statistical analysis, dimensionality reduction, and advanced classification algorithms to predict credit default payments effectively. It showcases the integration of domain knowledge with machine learning techniques for impactful real-world applications.


# Predictive Modeling for Credit Default Payments
This project focuses on creating a predictive model to assess the likelihood of credit default payments based on user financial and demographic data. The dataset used is sourced from the UCI Machine Learning Repository, specifically the "Default of Credit Card Clients Dataset".

# Objective
The objective of this project is to leverage machine learning models to predict whether a client will default on their next month's credit card payment based on various financial and behavioral attributes. This is achieved through exploratory data analysis, feature engineering, dimensionality reduction, and classification models.

Features
# Dataset Overview:
The dataset consists of the following key columns:

LIMIT_BAL: Amount of given credit (in NT dollars).
SEX: Gender (1 = male, 2 = female).
EDUCATION: Education level (1 = graduate school, 2 = university, 3 = high school, 4 = others).
MARRIAGE: Marital status.
AGE: Age of the client.
PAY_0 - PAY_6: Past payment records for the last six months.
BILL_AMT1 - BILL_AMT6: Amounts of bill statements for the last six months.
PAY_AMT1 - PAY_AMT6: Amounts of previous payments for the last six months.
default.payment.next.month: Target variable indicating whether the client defaults (1 = default, 0 = no default).
# Project Workflow :
# 1. Data Preprocessing
Handling Missing Values: Checked for and displayed missing values (none found in this dataset).
Standardization: Applied StandardScaler to normalize the feature values for better model performance.
Feature Selection: Key features were selected based on correlation and domain relevance.
# 2. Exploratory Data Analysis (EDA)
Boxplots: Explored distributions of features such as LIMIT_BAL, segmented by demographic attributes like SEX, MARRIAGE, and EDUCATION.
Correlation Analysis: Visualized correlations for key numerical variables such as bill amounts and payment amounts.
Custom Visualizations: Designed custom boxplots to analyze relationships between features.
# 3. Dimensionality Reduction
Principal Component Analysis (PCA):
Reduced the dataset dimensions while retaining 95% of the variance.
Determined optimal components to remove multicollinearity.
# 4. Variance Inflation Factor (VIF)
Calculated VIF to ensure multicollinearity among features is minimized.
# 5. Machine Learning Models
Four models were trained and evaluated:

Random Forest Classifier: Ensemble learning method for classification.
AdaBoost Classifier: Boosting model to improve weak learners.
XGBoost: Optimized gradient boosting algorithm for high performance.
LightGBM: Gradient boosting framework designed for fast computation and accuracy.
# 6. User Prediction Module
Allows users to input their financial and demographic data.
The input data is standardized, transformed using PCA, and passed through the LightGBM model for prediction.
Outputs whether the user is "Eligible" or "Not Eligible" for default prediction.
Results
Model Evaluation
Random Forest Score: (0.8104)
AdaBoost Score: (0.8068)
XGBoost Score: (0.8063)
LightGBM Score: (0.8127)
Why LightGBM?
The LightGBM model achieved the best performance in terms of accuracy and computational efficiency, making it the chosen model for this project.
