import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes # Toy dataset
from sklearn.metrics import mean_squared_error # Evaluation metric
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Comparison with sklearn model
from LinRegModel import Linear_Regression # my class



dataset = load_diabetes()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    
# Instance of my class
class_reg = Linear_Regression()
class_reg.fit(X_train, y_train)

y_pred = class_reg.predict(X_test)

print("RMSE for my class: ", mean_squared_error(y_test, y_pred) ** 0.5)



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred_test = lin_reg.predict(X_test)

print("RMSE for sklearn class: ", mean_squared_error(y_test, y_pred_test) ** 0.5)