import pandas as pd
import numpy as np

class Linear_Regression():
    
    def __init__(self, n_iters = 1000, early_stopping=True, regularization=None, cost_error='MSE', learning_rate=0.01):
        
        self.n_iters = n_iters
        self.early_stopping = early_stopping
        self.regularization = regularization
        self.cost_error = cost_error
        self.intercept = 0
        self.slope = 1
        self.learning_rate = learning_rate
        self.grad_b = 0
        self.grad_m = 0
        
        self.best_intercept = 0
        self.best_slope = 1
        self.best_error = 10000
        
        self.y_pred = None
        self.feature_nums = 0
        self.error = 100
        
    def fit(self, X, y):
        
        if len(X.shape) == 1:
            self.feature_nums = 1
        else:
            self.feature_nums = X.shape[1]
        
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
        
        if isinstance(y, pd.DataFrame):
            y = np.array(y)
            
        # Create slope and intercept arrays
        self.slope = np.array([self.slope] * self.feature_nums)
        self.intercept = np.array([self.intercept])
        
         
        for i in range(self.n_iters):
            
            # Prediction in current step
            self.y_pred = np.dot(X, self.slope) + self.intercept
            
            # Calculate error in current step
            if self.cost_error == 'MSE':
                self.error = np.mean((y - self.y_pred)**2)
            elif self.cost_error == 'MAE':
                self.error = np.mean(abs(y - self.y_pred))
            
            # Early stopping
            if ((((self.best_error / self.error) - 1) < -0.05) or ((1 - (self.best_error / self.error)) > 0.01)) and (i / self.n_iters >= 0.4):
                
                print("Early stopping activated...")
                break
            
            
            if self.best_error > self.error:
                self.best_error = self.error
                self.best_slope = self.slope
                self.best_intercept = self.intercept
                
            
            if self.cost_error == 'MAE':
                
                # Calculate intercept
                self.grad_b = np.mean((y - self.y_pred) / abs(y - self.y_pred))
                
                # Calculate slope
                self.grad_m = np.dot(((y - self.y_pred) / abs(y - self.y_pred)), X)
            
            elif self.cost_error == 'MSE':
                
                self.grad_m = np.dot((y - self.y_pred * 2), X)
                
                self.grad_b = np.mean((y - self.y_pred) * 2)
                
            # Calculate updated values of intercept and slope
            self.intercept = self.intercept + (self.grad_b * self.learning_rate)
            self.slope = self.slope + np.dot(self.grad_m, self.learning_rate)
            
            
    
    def predict(self, X):
        
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
        
        return np.dot(X, self.best_slope) + self.best_intercept