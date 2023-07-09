import pandas as pd
import numpy as np

class LassoRegression():
    
    def __init__(self, max_iter = 1000, alpha = 1.0, tolerance=1e-4):
        
        ''' Init parameters that will be used in this class '''
        self.max_iter = max_iter
        self.alpha = alpha
        self.tolerance = tolerance
        
        # Initialize parameters of linear regression
        self.intercept = 0
        self.slope = None
        
        # Initialize other params
        self.n_features = None
        self.n_samples = None
        
        # Init params to monitor change in intercept and slope
        self.prev_slope = None
        self.prev_intercept = None
        
        # Init var for y predictions
        self.y_pred = None
        self.curr_residuals = None
        
    def _soft_thresholding(self, x, alpha):
        '''
        Function for penalizing large weights, that will update them closer to zero
        '''
        if x > alpha:
            return x - alpha
        elif x < -alpha:
            return x + alpha
        else:
            return 0
    
    def fit(self, X, y):
        
        # Set feature num and sample num to our current train set
        self.n_features, self.n_samples = X.shape
        
        # Set slope shape to our current train set
        self.slope = np.zeros(self.n_features)
        
        # Start looping iters
        for iters in range(self.max_iter):
            
            # Update previous weights to current before changing them 
            self.prev_slope = self.slope
            self.prev_intercept = self.intercept
            
            # Lasso implemented with Coordinate descent updates weights column (feature) by column
            for curr_col in range(self.n_features):
                
                # Prediction with current weights
                self.y_pred = np.dot(X, self.slope) + self.intercept
                
                # Update weights for current feature
                self.curr_residuals = y - self.y_pred + self.slope[curr_col] * X[:, curr_col]
                
                # Update current slope
                self.slope[curr_col] = self._soft_thresholding(np.dot(X[:, curr_col], self.curr_residuals) / self.n_samples, self.alpha)
                
                # Update current intercept
                self.intercept = np.mean(y - np.dot(X, self.slope))
                
                if np.linalg.norm(self.slope - self.prev_slope) < self.tolerance and np.abs(self.intercept - self.prev_intercept) < self.tolerance:
                    break
        
        
    
    def predict(self, X):
        return (np.dot(X, self.slope) + self.intercept, self.slope, self.intercept)