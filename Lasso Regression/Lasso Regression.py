import pandas as pd
import numpy as np
from LassoRegModel import LassoRegression
from sklearn.linear_model import Lasso

np.random.seed(42)

X = np.random.randn(100, 100)
y = np.random.randn(100)

lasso_reg = LassoRegression()

lasso_reg.fit(X, y)

y_pred, weights, intercept = lasso_reg.predict(X)

print(f"Number of nullified features {len(weights[weights == 0])}")

