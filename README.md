# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Upload and load the Salary.csv dataset and handle missing values.

2.Encode the Position column and select Position and Level as features and Salary as the target.

3.Split the dataset into 80% training and 20% testing data.

4.Train a Decision Tree Regressor and predict salaries for the test data.

5.Evaluate using the R² score and visualize the decision tree and feature importance.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Kailaash V
Register Number: 212224060112
# ============================================
# Salary Prediction using Decision Tree Regressor
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------
# 1) Upload Salary.csv file
# --------------------------------------------

from google.colab import files

uploaded = files.upload()

# Automatically get uploaded file name
csv_path = list(uploaded.keys())[0]

# Load dataset
data = pd.read_csv(csv_path)

print("Dataset Loaded Successfully!")
print("Uploaded File:", csv_path)


# --------------------------------------------
# 2) Data Exploration
# --------------------------------------------

print("\nDataset Shape:", data.shape)

print("\nFirst 5 Rows:")
display(data.head())

print("\nDataset Information:")
data.info()

print("\nMissing Values:")
print(data.isnull().sum())


# --------------------------------------------
# 3) Handle Missing Values
# --------------------------------------------

if data.isnull().any().any():

    print("\nMissing values found.")
    print("Dropping rows with missing values...")

    data = data.dropna()

    print("New Dataset Shape:", data.shape)


# --------------------------------------------
# 4) Encode categorical column 'Position'
# --------------------------------------------

if "Position" in data.columns:

    le = LabelEncoder()

    data["Position"] = le.fit_transform(
        data["Position"].astype(str)
    )

    print("\nLabel Encoding Mapping (Position):")

    mapping = dict(
        zip(
            le.classes_,
            le.transform(le.classes_)
        )
    )

    print(mapping)


# --------------------------------------------
# 5) Select Features and Target
# --------------------------------------------

X = data[[
    "Position",
    "Level"
]]

y = data["Salary"]


print("\nFeature Sample:")
display(X.head())

print("\nTarget Sample:")
display(y.head())


# --------------------------------------------
# 6) Train-Test Split
# --------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)


# --------------------------------------------
# 7) Create Decision Tree Regressor
# --------------------------------------------

dt = DecisionTreeRegressor(
    random_state=10
)

# Train the model
dt.fit(X_train, Y_train)

print("\nModel Training Completed!")


# --------------------------------------------
# 8) Prediction
# --------------------------------------------

y_pred = dt.predict(X_test)

print("\nPredicted Salaries:")
print(y_pred)


# --------------------------------------------
# 9) Evaluate Model
# --------------------------------------------

r2 = metrics.r2_score(
    Y_test,
    y_pred
)

print("\nR² Score:", round(r2, 4))


# --------------------------------------------
# 10) Compare Actual vs Predicted Salary
# --------------------------------------------

comparison = pd.DataFrame({
    "Actual Salary": Y_test.values,
    "Predicted Salary": y_pred
})

print("\nActual vs Predicted Salary:")
display(comparison)


# --------------------------------------------
# 11) Visualize Decision Tree
# --------------------------------------------

plt.figure(figsize=(16, 10))

plot_tree(
    dt,
    feature_names=[
        "Position",
        "Level"
    ],
    filled=True,
    rounded=True
)

plt.title(
    "Decision Tree Regressor for Salary Prediction"
)

plt.show()


# --------------------------------------------
# 12) Feature Importance
# --------------------------------------------

importances = pd.Series(
    dt.feature_importances_,
    index=[
        "Position",
        "Level"
    ]
).sort_values(ascending=False)

print("\nFeature Importances:")

display(importances) 
*/
```

## Output:
<img width="1068" height="829" alt="image" src="https://github.com/user-attachments/assets/124be67f-b08f-49a6-b1a3-6796f76cbb83" />

```
Label Encoding Mapping (Position):
{'Business Analyst': np.int64(0), 'C-level': np.int64(1), 'CEO': np.int64(2), 'Country Manager': np.int64(3), 'Junior Consultant': np.int64(4), 'Manager': np.int64(5), 'Partner': np.int64(6), 'Region Manager': np.int64(7), 'Senior Consultant': np.int64(8), 'Senior Partner': np.int64(9)}
```

<img width="880" height="715" alt="image" src="https://github.com/user-attachments/assets/2527041c-eebf-40b4-84f8-6d5cde0d4775" />

<img width="874" height="264" alt="image" src="https://github.com/user-attachments/assets/463405d4-62d6-4ede-b845-1a8fc15bb2c1" />

<img width="1260" height="812" alt="image" src="https://github.com/user-attachments/assets/6a36b660-281f-48c6-adf7-55f15652df9c" />

<img width="1008" height="211" alt="image" src="https://github.com/user-attachments/assets/5f9032d8-a52a-4018-9769-f9ba8120e6f5" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
