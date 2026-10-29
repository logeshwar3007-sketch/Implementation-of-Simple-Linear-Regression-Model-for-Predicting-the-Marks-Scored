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

# 3) Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))

# 4) Create and train the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)   # fit on training data

# 5) Predict on the test set
Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)

# 6) Plot training results
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

# 7) Plot testing results (use X_test sorted for a nicer line)
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

# 8) Evaluation metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# 9) Example: predict for new students
new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
```

## Output:
<img width="187" height="341" alt="image" src="https://github.com/user-attachments/assets/8c99c35f-77a8-4bbf-8e6e-ec8fe442adb0" />

<img width="847" height="92" alt="image" src="https://github.com/user-attachments/assets/81b57a4f-967b-4b4d-903a-74ae517e8fb7" />

<img width="722" height="82" alt="image" src="https://github.com/user-attachments/assets/a7cd8895-8e42-4d15-a123-e9b9c5623532" />

<img width="780" height="502" alt="image" src="https://github.com/user-attachments/assets/8d84adcd-1586-43e4-aa31-0f5ced780aea" />

<img width="683" height="487" alt="image" src="https://github.com/user-attachments/assets/00cd6f80-aa8e-4c11-adb3-ac473eab6be9" />

<img width="522" height="95" alt="image" src="https://github.com/user-attachments/assets/46e29bc5-cd01-4b57-b79f-e11986c541ca" />

<img width="522" height="48" alt="image" src="https://github.com/user-attachments/assets/6d9fbcbf-f2b9-4de0-bfef-d69c779ce6d2" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
