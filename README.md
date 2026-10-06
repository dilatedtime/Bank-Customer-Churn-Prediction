# Bank Customer Churn Prediction

This notebook studies which retail banking customers are likely to leave. It walks through data cleaning, exploratory charts, class balancing with SMOTE, feature scaling, and three classification models: decision tree, random forest, and K-nearest neighbors.

The repository includes the dataset used by the notebook, so you can run the analysis without downloading a separate file.

## What is in the project

- `BankCustomerChurn.ipynb`: the complete analysis and saved model output
- `churn.csv`: 10,002 customer records with account, demographic, and churn fields

The target column is `Exited`, where `1` means the customer left and `0` means the customer stayed. The notebook removes identifiers, drops incomplete rows, encodes `Gender` and `Geography`, balances the classes, filters selected outliers, and scales the main numeric features.

## Models and saved results

The notebook uses an 80/20 train-test split and compares each base model with a grid-searched version.

| Model | Base accuracy | Tuned accuracy |
| --- | ---: | ---: |
| Decision tree | 82.51% | 82.51% |
| Random forest | 88.29% | 88.00% |
| K-nearest neighbors | 78.36% | 81.69% |

These figures come from the output currently saved in the notebook. The tuned random forest did not beat its untuned run, while tuning helped KNN. One decision-tree grid option, `max_features='auto'`, is invalid in newer scikit-learn releases, so those grid combinations are skipped with a warning.

## Run the notebook

```bash
git clone https://github.com/dilatedtime/Bank-Customer-Churn-Prediction.git
cd Bank-Customer-Churn-Prediction
python -m venv .venv
```

Activate the environment, then install the notebook dependencies:

```bash
python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook BankCustomerChurn.ipynb
```

Run the cells in order. The notebook applies SMOTE before the train-test split, so treat the saved scores as exploratory rather than as a production estimate. For a stricter evaluation, split first and apply SMOTE only to the training data.
