import pandas as pd
import numpy as np
from RidgeModel import Ridge_Regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Init Ridge Regression classes 
ridge = Ridge_Regression(param_weight=0.1) # Own solution
sklearn_ridge = Ridge() # Sklearn solution


dataset = load_diabetes()

X = dataset.data
y = dataset.target

# Train/test split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train own model
ridge.fit(X_train, y_train)
y_own_pred, weights = ridge.predict(X_test)

# Train sklearb model
sklearn_ridge.fit(X_train, y_train)
y_pred_sk = sklearn_ridge.predict(X_test)


print(f"Weights for columns using my solution {weights}")
print("RMSE for my class: ", mean_squared_error(y_test, y_own_pred) ** 0.5)

print(f"Weights for columns using sklearb solution {sklearn_ridge.coef_}")
print("RMSE for sklearn class: ", mean_squared_error(y_test, y_pred_sk) ** 0.5)