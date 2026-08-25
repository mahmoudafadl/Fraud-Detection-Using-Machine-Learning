# Fraud Detection Using Machine Learning

A Machine Learning project for detecting potentially fraudulent transactions using supervised classification techniques.

The project focuses on building a complete data science workflow, starting from data understanding and cleaning, through exploratory data analysis and feature engineering, to model training, evaluation, comparison, and final model selection.

---

## Project Overview

Fraud detection is a binary classification problem where the goal is to identify whether a financial transaction is legitimate or fraudulent.

In this project, three different Machine Learning classification models were developed and compared:

- Logistic Regression
- Decision Tree
- Random Forest

The models were evaluated using multiple classification metrics, with particular attention to the trade-off between detecting fraudulent transactions and avoiding false alarms.

---

## Dataset

The dataset contains transaction-level information used to predict the `risk_label` of each transaction.

### Target Variable

- `risk_label = 0` → Legitimate Transaction
- `risk_label = 1` → Fraudulent Transaction

The original dataset contains 5,300 records and 15 columns.

After removing duplicate records, the dataset contains 5,000 transactions.

The cleaned dataset contains:

- 4,500 legitimate transactions
- 500 fraudulent transactions
- 90% legitimate transactions
- 10% fraudulent transactions

This class imbalance was taken into consideration during model development.

---

## Features

The original dataset contains transaction, customer, merchant, device, and risk-related information.

### Numerical Features

- `transaction_hour`
- `account_age_days`
- `previous_chargebacks`
- `transaction_amount`
- `transaction_velocity_1h`
- `transaction_velocity_24h`
- `avg_transaction_amount_30d`

### Categorical Features

- `merchant_category`
- `transaction_country`
- `device_type`

### Binary Features

- `is_international`
- `is_high_risk_merchant`

### Removed Identifier Features

The following identifiers were removed before modeling:

- `transaction_id`
- `customer_id`

These fields were excluded because they are identifiers rather than meaningful predictive features.

---

## Project Workflow

The project follows a complete Machine Learning pipeline:

1. Dataset Loading
2. Dataset Understanding
3. Exploratory Data Analysis (EDA)
4. Target Distribution Analysis
5. Duplicate Removal
6. Invalid Value Handling
7. Feature Analysis
8. Feature Engineering
9. Train/Test Split
10. Missing Value Handling
11. Outlier Detection and Analysis
12. Feature Scaling
13. Categorical Encoding
14. Class Weight Calculation
15. Model Training
16. Model Evaluation
17. Model Comparison
18. Confusion Matrix Analysis
19. Final Model Selection
20. Feature Importance / Interpretation

---

## Data Cleaning

Several data quality issues were identified and handled during preprocessing.

### Duplicate Records

The original dataset contained 300 duplicate rows.

These duplicate records were removed, reducing the dataset from 5,300 to 5,000 records.

### Invalid Values

Invalid values were identified in:

- `transaction_amount`
- `avg_transaction_amount_30d`
- `account_age_days`
- `transaction_hour`

A total of 32 invalid values were detected and converted into missing values before being handled during the missing-value preprocessing stage.

### Missing Values

Missing values were handled according to the feature type:

- Numerical features
- Binary features
- Categorical features

After preprocessing, the dataset contained no missing values.

---

## Feature Selection

Feature selection was performed to identify relevant variables for the prediction task.

Correlation analysis showed that some of the strongest numerical relationships with the target were associated with:

- `transaction_velocity_24h`
- `transaction_velocity_1h`
- `previous_chargebacks`
- `transaction_amount`

Fraud rates were also analyzed across categorical and binary features to understand how different transaction characteristics were associated with fraudulent behavior.

---

## Feature Engineering

Two additional features were created:

### `amount_to_avg_ratio`

Represents the relationship between the current transaction amount and the customer's average transaction amount over the previous 30 days.

### `is_night_transaction`

A binary feature indicating whether the transaction occurred during the defined night-time period.

After feature engineering, the model dataset contained 14 predictive features.

---

## Train/Test Split

The cleaned dataset was divided using a stratified 80/20 train-test split.

### Training Set

- 4,000 samples
- 90% legitimate
- 10% fraud

### Testing Set

- 1,000 samples
- 90% legitimate
- 10% fraud

Stratification was used to preserve the class distribution in both datasets.

---

## Outlier Analysis

Outliers were detected using the Interquartile Range (IQR) method.

The IQR boundaries were calculated using the training data only.

Instead of automatically removing the detected outliers, their relationship with fraud rates was analyzed.

Several outlier groups showed substantially higher fraud rates than their corresponding non-outlier observations.

For this reason, the detected outliers were retained because extreme transaction behavior may represent meaningful fraud signals.

---

## Preprocessing

A preprocessing pipeline was created for the Machine Learning models.

### Numerical Features

Standard Scaling was applied.

### Binary Features

Binary features were kept as `0/1`.

### Categorical Features

One-Hot Encoding was applied.

After preprocessing, the training data contained 26 processed features.

---

## Handling Class Imbalance

The target variable is imbalanced, with fraudulent transactions representing approximately 10% of the cleaned dataset.

To address this imbalance, balanced class weights were applied during model training.

The calculated training class weights were:

- Legitimate: `0.5556`
- Fraud: `5.0000`

This gives greater importance to the minority fraud class during training.

---

## Machine Learning Models

Three classification algorithms were trained and evaluated.

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

Configuration:

- Class weights: Balanced
- Maximum iterations: 1000
- Random state: 42

### 2. Decision Tree

Decision Tree was used to provide an interpretable tree-based classification approach.

Configuration:

- Class weights: Balanced
- Random state: 42

The trained tree contained:

- 451 nodes
- 226 leaves
- Depth of 23

### 3. Random Forest

Random Forest was used as an ensemble learning approach consisting of multiple decision trees.

Configuration:

- 100 trees
- Class weights: Applied
- Random state: 42

The model used 26 processed features.

---

## Model Evaluation

The models were evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

For fraud detection, accuracy alone is not sufficient because the dataset is imbalanced.

Therefore, Precision, Recall, F1-Score, ROC-AUC, and the Confusion Matrix were considered when comparing the models.

---

## Model Comparison

The three models provide different strengths:

### Logistic Regression

Provides strong fraud recall, but can produce a relatively high number of false positives.

### Decision Tree

Provides an interpretable set of decision rules and can capture non-linear relationships in the data.

### Random Forest

Provides a stronger overall balance between detecting fraud and reducing false alarms.

---

## Final Model

### Random Forest

Random Forest was selected as the final model because it provided the strongest overall balance among the evaluated approaches.

The final selection considered the trade-off between:

- Detecting fraudulent transactions
- Reducing false positives
- Overall classification performance

---

## Key Insights

The analysis showed that transaction behavior can provide important signals for fraud detection.

Some of the strongest relationships with the fraud label were associated with transaction velocity and previous chargebacks.

The analysis also showed that certain extreme transaction behaviors were associated with significantly higher fraud rates.

For example, outlier transactions in:

- `transaction_velocity_24h`
- `transaction_velocity_1h`
- `transaction_amount`
- `amount_to_avg_ratio`

showed higher fraud rates compared with their non-outlier observations.

This supports the decision to retain statistically detected outliers rather than automatically removing them.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Project Structure

```text
Fraud-Detection-Using-Machine-Learning/
│
├── images/
│   ├── ...
│
├── data/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── src/
│   └── ...
│
├── README.md
└── requirements.txt
