# Loan Default Prediction

A machine learning project for predicting whether a borrower is likely to default on a loan using financial and demographic information. The project compares multiple classification algorithms, performs feature engineering and preprocessing, and uses cross-validation and hyperparameter tuning to select a robust final model.

## Problem Statement

Financial institutions need reliable methods for assessing the risk associated with loan applicants. This project aims to develop a machine learning model that predicts whether a loan will default based on borrower demographics, financial information, credit history, and loan characteristics.

The target variable is `Default`:

- `0` — Loan does not default
- `1` — Loan defaults

## Dataset

The dataset contains **255,347 rows and 18 columns**, consisting of financial, demographic, employment, and loan-related attributes.

### Key Features

| Feature | Description |
|---|---|
| `LoanID` | Unique identifier for each loan |
| `Age` | Borrower's age |
| `Income` | Annual income |
| `LoanAmount` | Amount borrowed |
| `CreditScore` | Borrower's creditworthiness score |
| `MonthsEmployed` | Number of months employed |
| `NumCreditLines` | Number of open credit lines |
| `InterestRate` | Loan interest rate |
| `LoanTerm` | Loan term in months |
| `DTIRatio` | Debt-to-income ratio |
| `Education` | Highest level of education |
| `EmploymentType` | Type of employment |
| `MaritalStatus` | Marital status |
| `HasMortgage` | Whether the borrower has a mortgage |
| `HasDependents` | Whether the borrower has dependents |
| `LoanPurpose` | Purpose of the loan |
| `HasCoSigner` | Whether the loan has a co-signer |
| `Default` | Target variable |

## Exploratory Data Analysis

The dataset contained no missing values, but several preprocessing challenges were identified:

- Significant outliers in `LoanAmount`, `Income`, and `InterestRate`
- Right-skewed numerical features such as `Income`, `LoanAmount`, and `DTIRatio`
- Imbalanced categorical variables
- Large differences in scale among numerical features
- Significant class imbalance in the target variable

Only **11.63% of loans were defaults**, making stratified evaluation important.

## Data Preprocessing

### Outlier Removal

Outliers were detected using the **Interquartile Range (IQR)** method:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values outside these bounds were treated as outliers and removed.

### Encoding and Scaling

- One-hot encoding was applied to categorical variables.
- `Education` was ordinally encoded.
- `HasDependents`, `HasCoSigner`, and `HasMortgage` were converted from `Yes/No` values to boolean features.
- Numerical features were standardized using `StandardScaler`.

## Feature Engineering

Several domain-driven features were created to capture relationships between borrower characteristics.

### Income-to-Loan Ratio

```text
IncomeToLoanAmount = Income / LoanAmount
```

Represents the borrower's income relative to the amount being borrowed.

### Credit Score-to-Age Ratio

```text
CreditScoreToAge = CreditScore / Age
```

Provides an indication of creditworthiness relative to the borrower's age.

### Loan Amount × Interest Rate

```text
LoanAmountInterestRate = LoanAmount × InterestRate
```

Captures the overall interest burden associated with the loan.

### Employment-to-Age Ratio

```text
MonthsEmployedToAge = MonthsEmployed / Age
```

Provides an indication of employment history and job stability relative to age.

### DTI × Loan Amount

```text
DTIRatioLoanAmount = DTIRatio × LoanAmount
```

Captures the relationship between existing debt burden and the requested loan amount.

## Models Evaluated

The following classification algorithms were evaluated:

- Decision Tree
- K-Nearest Neighbors
- Random Forest
- AdaBoost
- XGBoost

Models such as Multinomial Naive Bayes, Bernoulli Naive Bayes, Gaussian Naive Bayes, and Logistic Regression were initially excluded based on their assumptions and suitability for the dataset.

## Initial Model Comparison

Models were evaluated using **10-fold Stratified K-Fold Cross-Validation**.

| Model | Mean Accuracy |
|---|---:|
| Random Forest | **0.8853** |
| AdaBoost | **0.8852** |
| XGBoost | **0.8845** |
| KNN | 0.8735 |
| Decision Tree | 0.8018 |

Random Forest and AdaBoost achieved the highest initial accuracy, while Decision Tree performed substantially worse.

## Hyperparameter Tuning

The top-performing ensemble models were further tuned using **RandomizedSearchCV** and Stratified K-Fold Cross-Validation.

### AdaBoost

The learning rate and number of estimators were tuned.

Best observed accuracy:

```text
0.8861
```

### Random Forest

Parameters including `n_estimators` and `ccp_alpha` were explored.

Best observed accuracy:

```text
0.8854
```

### XGBoost

Two preprocessing approaches were investigated.

**Approach 1**
- Existing encoding strategy
- Randomized hyperparameter search

**Approach 2**
- One-hot encoding for all categorical features
- Standard scaling of numerical features
- Randomized hyperparameter search

XGBoost was ultimately selected because of its strong and consistent performance, scalability, and built-in regularization.

## Additional Models

Additional models were investigated during the second checkpoint.

### Multi-Layer Perceptron

Best configuration:

```text
Solver: Adam
Hidden Layers: (100, 50)
Activation: Logistic
Alpha: 0.001
Random State: 42
Learning Rate: Constant
```

Accuracy:

```text
0.88825
```

### Support Vector Classifier

Best configuration:

```text
C: 10
Kernel: RBF
Gamma: Scale
Random State: 42
```

Accuracy:

```text
0.88758
```

### Logistic Regression

Best configuration:

```text
C: 0.1
Penalty: L2
Solver: liblinear
Max Iterations: 500
Random State: 42
```

Accuracy:

```text
0.88586
```

## Final Model

**XGBoost** was selected as the final model for the project.

The selection was motivated by:

- Strong predictive performance
- Ability to model non-linear relationships
- Ability to capture complex feature interactions
- Efficient training and inference
- Built-in regularization
- Scalability to large datasets
- Robustness against overfitting

## Key Insights

### Credit Risk Indicators

Features such as:

- `CreditScore`
- `DTIRatio`
- `IncomeToLoanAmount`

are important indicators of default risk. Lower credit scores, higher debt burdens, and insufficient income relative to the requested loan amount are associated with increased default risk.

### Class Imbalance

Only **11.63%** of loans defaulted. Stratified K-Fold Cross-Validation was therefore used to preserve class proportions across validation folds.

### Feature Engineering

Domain-specific engineered features provided additional information about borrower repayment capacity and financial behavior.

### Ensemble Learning

Ensemble methods, particularly Random Forest, AdaBoost, and XGBoost, performed strongly because they can model non-linear relationships and interactions between features.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib / Seaborn
- Jupyter Notebook

## Machine Learning Pipeline

```text
Raw Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Data Cleaning
     │
     ├── Outlier Detection
     ├── Categorical Encoding
     └── Numerical Scaling
     │
     ▼
Feature Engineering
     │
     ├── Income / LoanAmount
     ├── CreditScore / Age
     ├── LoanAmount × InterestRate
     ├── MonthsEmployed / Age
     └── DTIRatio × LoanAmount
     │
     ▼
Stratified K-Fold Cross-Validation
     │
     ▼
Model Comparison
     │
     ├── Decision Tree
     ├── KNN
     ├── Random Forest
     ├── AdaBoost
     └── XGBoost
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Final XGBoost Model
     │
     ▼
Loan Default Prediction
```

## Future Improvements

Potential directions include:

- Incorporating behavioral and transactional features
- Using historical loan repayment information
- Exploring stacking and other advanced ensemble methods
- Investigating neural-network-based feature learning
- Further improving evaluation of minority-class performance

## Repository

The project repository is available at:

https://github.com/py-xis/AIM-511-Machine-Learning-Project
