# EX - 02 Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Step 1: Import Libraries

2. Step 2: Create Dataset (Hours studied vs Marks scored)

3. Step 3: Split into Features and Target

4. Step 4: Train-test split

5. Step 5: Train Linear Regression Model

6. Step 6: Predictions

7. Step 7: Model Evaluation

8. Step 8: Visualization

9. Step 9: Predict Marks for custom input
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S LOGESHWAR
RegisterNumber: 25014493 
*/

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1) Load dataset
df = pd.read_csv('exp_2_dataset_student_scores.csv')   # CSV should have two columns, e.g. "Hours","Scores"
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")



# 2) Prepare input (X) and output (Y)
# Assume CSV columns: Hours (feature) and Scores (target)
X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)
Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)
print("X (features):", X.flatten())
print("Y (targets):", Y)

```

## Output:
<img width="187" height="341" alt="image" src="https://github.com/user-attachments/assets/8c99c35f-77a8-4bbf-8e6e-ec8fe442adb0" />

<img width="847" height="92" alt="image" src="https://github.com/user-attachments/assets/81b57a4f-967b-4b4d-903a-74ae517e8fb7" />

<img width="722" height="82" alt="image" src="https://github.com/user-attachments/assets/a7cd8895-8e42-4d15-a123-e9b9c5623532" />

<img width="780" height="502" alt="image" src="https://github.com/user-attachments/assets/8d84adcd-1586-43e4-aa31-0f5ced780aea" />

<img width="683" height="487" alt="image" src="https://github.com/user-attachments/assets/00cd6f80-aa8e-4c11-adb3-ac473eab6be9" />

<img width="1245" height="814" alt="image" src="https://github.com/user-attachments/assets/698b0a8b-722a-4096-9d80-38027a934c86" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
