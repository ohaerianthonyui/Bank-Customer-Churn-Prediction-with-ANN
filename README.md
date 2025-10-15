# Bank-Customer-Churn-Prediction-with-ANN

---

# Customer Churn Prediction Project

## Project Overview

Customer churn — when customers stop doing business with a company — is a critical metric for many industries. Predicting churn early enables companies to proactively engage at-risk customers, improve retention, and increase revenue.

This project uses machine learning techniques to build a predictive model that identifies customers likely to churn, based on demographic, transactional, and behavioral data.

---

## Dataset

The dataset contains information about customers’ profiles and banking behavior, including:

* **Geography:** Customer’s country (France, Spain, Germany)
* **Gender:** Male or Female
* **Credit Score**
* **Age**
* **Tenure:** Number of years the customer has stayed with the bank
* **Balance:** Account balance
* **Number of Products:** Number of bank products used
* **Estimated Salary**
* **Has Credit Card:** Binary indicator
* **Is Active Member:** Binary indicator
* **Exited:** Target variable (1 if the customer churned, 0 otherwise)

---

## Objective

* Build a classification model to predict customer churn (Exited = 1).
* Handle class imbalance using SMOTE to improve model sensitivity to churners.
* Evaluate model performance with accuracy, precision, recall, and F1-score.
* Provide actionable insights for customer retention strategies.

---

## Methodology

1. **Data Preprocessing**

   * Handling missing values (if any)
   * Encoding categorical variables (Geography, Gender) using one-hot and label encoding
   * Scaling numerical features (Tenure, CreditScore, Age, Balance, NumOfProducts, EstimatedSalary)
   * Splitting data into training and testing sets
2. **Addressing Class Imbalance**

   * Used SMOTE (Synthetic Minority Over-sampling Technique) on training data to generate synthetic churn instances
3. **Model Building**

   * Built a neural network classifier with TensorFlow/Keras
   * Trained the model on balanced data
4. **Model Evaluation**

   * Evaluated on original test data without resampling
   * Calculated accuracy, precision, recall, and F1-score

---

## Results

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 82.45% |
| Precision (Churn) | 56%    |
| Recall (Churn)    | 63%    |
| F1-score (Churn)  | 59%    |

The model shows promising performance, especially in detecting churners (higher recall), which is crucial for proactive retention campaigns.

---

## Future Work

* Experiment with other models such as XGBoost and Random Forest for potential improvements.
* Perform hyperparameter tuning and threshold optimization.
* Use feature importance methods (e.g., SHAP) for explainability.
* Deploy the model into a real-time scoring pipeline.
* Integrate customer feedback and behavioral signals for richer modeling.

---

## Requirements

* Python 3.x
* Libraries: pandas, numpy, scikit-learn, imblearn, tensorflow, matplotlib, seaborn

---

## How to Run

1. Clone this repository

   ```
   git clone https://github.com/ohaerianthonyui/Bank-Customer-Churn-Prediction-with-ANN.git
   ```
2. Install dependencies

   ```
   pip install -r requirements.txt
   ```
3. Run the notebook or Python scripts to reproduce the analysis and modeling.

---
