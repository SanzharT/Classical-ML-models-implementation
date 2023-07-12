# Import libraries
import pandas as pd
import numpy as np


class Ridge_Regression():

    def __init__(self, n_iters = 1000, eta=0.01, param_weight=0.01, base_points=True, tolerance=1e-3):

        self.n_iters = n_iters
        self.eta = eta
        self.base_points = base_points
        self.tolerance = tolerance
        self.param_weight = param_weight

        self.intercept = 0
        self.slope = 1

        self.prev_intercept = 0
        self.prev_slope = 1

        self.curr_error = None
        self.y_pred = None
        self.grad_intercept = None
        self.grad_slope = None
        self.norms_of_slopes = None
        self.n_features = None

    def fit(self, X, y):

        self.n_features = X.shape[-1]

        # Set n-length array for slope and intercept
        if self.base_points:

            self.slope = np.ones(self.n_features)
            self.prev_slope = np.ones(self.n_features)

            self.intercept = np.array([1])
            self.prev_intercept = np.array([1])
        else:
            # If base points set to False, then pick randomly starting weights
            self.slope = np.random.randn(self.n_features) * 10
            self.prev_slope = np.random.randn(self.n_features) * 10

            self.intercept = np.random.randn(1) * 5
            self.prev_intercept = np.random.randn(1) * 5

        for i in range(self.n_iters):
                
            # Set prev weight vals to current vals
            self.prev_intercept = self.intercept.copy()
            self.prev_slope = self.slope.copy()

            self.y_pred = np.dot(X, self.slope) + self.intercept

            # Compute curr diff between y and y_pred
            self.curr_error = (y - self.y_pred) ** 2

            self.norms_of_slopes = np.sqrt(np.sum(self.slope))

            self.grad_intercept = np.mean((y - self.y_pred) * 2) 
            self.grad_slope = np.dot((y - self.y_pred) * 2, X) + (2 * self.param_weight * self.norms_of_slopes)
            #print(f"Gradient slope at {i}th iteration: {self.grad_slope}")

            self.intercept = self.intercept + (self.grad_intercept * self.eta)
            self.slope = self.slope + np.dot(self.grad_slope, self.eta)


            # Early stop activation
            if np.linalg.norm(self.slope - self.prev_slope) < self.tolerance and np.abs(self.intercept - self.prev_intercept) < self.tolerance:
                break


    def predict(self, X):

        pred = np.dot(X, self.slope) + self.intercept

        return (pred, self.slope)