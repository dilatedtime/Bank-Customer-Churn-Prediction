# Customer Churn Prediction

## Overview  
This repository implements a robust machine‑learning pipeline to predict customer churn for a retail banking dataset. It encompasses end‑to‑end data processing, class imbalance mitigation, feature engineering, model training with hyperparameter optimization, and comprehensive evaluation of three classification algorithms.

## Key Features  
- **Exploratory Data Analysis & Visualization**  
- **Class Imbalance Handling via SMOTE**  
- **Categorical Encoding & Outlier Removal**  
- **Feature Scaling (MinMaxScaler)**  
- **Model Training & Hyperparameter Tuning (GridSearchCV)**  
- **Evaluation Reports & Confusion Matrix Visualizations**

## Dataset  
- **Source File:** `churn.csv`  
- **Dimensions:** ~10,000 rows × 14 columns  
- **Target Variable:** `Exited`  
  - `0` = Retained  
  - `1` = Churned  
- **Predictor Variables:**  
  - **Demographics:** `Age`, `Gender`, `Geography`, `Tenure`  
  - **Account Metrics:** `CreditScore`, `Balance`, `NumOfProducts`, `EstimatedSalary`  
  - **Account Flags:** `HasCrCard`, `IsActiveMember`

## Data Preprocessing  
1. **Column Pruning**  
   - Remove non‑informative identifiers: `RowNumber`, `Surname`, `CustomerId`.  
2. **Missing‑Value Handling**  
   - Drop records containing null values.  
3. **Categorical Encoding**  
   - Label‑encode `Gender` (Male/Female → 1/0)  
   - Label‑encode `Geography` (France/Spain/Germany → 0/1/2)  
4. **Class Balancing**  
   - Apply SMOTE to oversample minority class (`Exited = 1`).  
5. **Outlier Filtering**  
   - Remove extreme values in `Age`, `NumOfProducts`, `CreditScore` using the IQR method.  
6. **Feature Scaling**  
   - Apply `MinMaxScaler` to `CreditScore`, `Balance`, and `EstimatedSalary`.

## Model Architectures & Hyperparameter Tuning  
All models are trained on an 80/20 train–test split with five‑fold cross‑validation for tuning.

| Model                     | Base Estimator                                               | Tuned Hyperparameters (GridSearchCV)                                                                                  |
|---------------------------|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Decision Tree**         | `DecisionTreeClassifier(random_state=42)`                    | `max_depth`: [None,10,20], `min_samples_split`: [2,5], `min_samples_leaf`: [1,2,4], `max_features`: ['auto','sqrt'], `criterion`: ['gini','entropy'] |
| **Random Forest**         | `RandomForestClassifier(random_state=42)`                    | `n_estimators`: [30,50,70], `max_depth`: [None,10,20], `min_samples_split`: [2,5], `min_samples_leaf`: [1,2,4], `max_features`: ['sqrt'], `criterion`: ['gini','entropy'] |
| **K‑Nearest Neighbors**   | `KNeighborsClassifier()`                                     | `n_neighbors`: [3,5,7,9], `weights`: ['uniform','distance'], `metric`: ['euclidean','manhattan']                        |

## Evaluation Metrics  
For each model—both pre‑ and post‑tuning—the following metrics are reported on the hold‑out test set:  
- **Accuracy**  
- **Precision, Recall & F1‑Score** (for the churn class)  
- **Confusion Matrix** (visualized via `ConfusionMatrixDisplay`)

| Model                   | Accuracy (Before Tuning) | Accuracy (After Tuning) | Precision (Churn) | Recall (Churn) | F1‑Score (Churn) |
|-------------------------|--------------------------|-------------------------|-------------------|----------------|------------------|
| Decision Tree           | 84.5%                    | 86.2%                   | 0.40              | 0.45           | 0.42             |
| Random Forest           | 87.1%                    | 89.0%                   | 0.48              | 0.52           | 0.50             |
| K‑Nearest Neighbors     | 80.3%                    | 83.5%                   | 0.35              | 0.38           | 0.36             |


## Execution  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/customer-churn-prediction.git
   cd Bank Customer-Churn-Prediction

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the analysis notebook**

   ```bash
   jupyter notebook BankCustomerChurn.ipynb
   ```
4. **Follow the notebook** to execute cells in order—starting with EDA, through preprocessing, model training, tuning, and evaluation.

## Results & Conclusion

* **Top Performer:** Random Forest (tuned) achieved **89.0% accuracy**, **0.48 precision**, **0.52 recall**, and **0.50 F1‑score** on the churn class.
* **Key Insights:**

  * Hyperparameter optimization consistently improved model performance by approximately 2–3%.
  * SMOTE oversampling significantly enhanced recall for the minority class.
  * Outlier removal and feature scaling contributed to model stability and convergence.
* **Conclusion:**
  The implemented pipeline demonstrates strong predictive capability for customer churn, leveraging ensemble methods and rigorous preprocessing to address data imbalance and feature variability.
