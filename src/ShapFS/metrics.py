import numpy as np
import scipy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

class Metrics():
    def __init__(self, y, yhat, yrange=None):
        if yrange==None:
            self.yrange = max(y.flatten()) - min(y.flatten())
        else:
            self.yrange=yrange
        self.y = y
        self.yhat = yhat
    
    def MAE(self):
        """
        Mean Absolute Error 
        """
        return np.mean(np.abs(self.y - self.yhat))

    def RMSE(self):
        """
        Root Mean Squared Error
        """
        return np.sqrt(MSE(self.y, self.yhat))

    def RMSEP(self): 
        """
        Root mean squared error in percentage of range of target variable
        """
        rmsep = Metrics.RMSE(self) * 100 / self.yrange
        return rmsep

    def RPD(self):
        rmse = Metrics.RMSE(self) # Calculate RMSEP (Root Mean Square Error of Prediction)
        std_dev_observed = np.std(self.y, ddof=1)  # ddof=1 for sample standard deviation
        return std_dev_observed / rmse

    def R2(self):
        return r2_score(self.y, self.yhat)

    def reg_intercept(self):
        reg_= scipy.stats.linregress(np.array([i[0] for i in self.y]), np.array([i[0] for i in self.yhat]))
        return reg_.intercept

    def reg_slope(self):
        reg_= scipy.stats.linregress(np.array([i[0] for i in self.y]), np.array([i[0] for i in self.yhat]))
        return reg_.slope

    def x_plain(self): # x line for drowing linear regression line
        max_y = max(self.y) if isinstance(self.y, list) else np.max(self.y)
        # If max_y is still an array, extract the scalar value using .item()
        if isinstance(max_y, np.ndarray):
            max_y = max_y.item()

        return np.array([i for i in range(int(max_y + max_y * 0.03))])

    def y_reg(self):
        x_plain = Metrics.x_plain(self)
        reg_= scipy.stats.linregress(np.array([i[0] for i in self.y]), np.array([i[0] for i in self.yhat]))
        return np.array([reg_.intercept + reg_.slope * i for i in x_plain])
    

    
    

if __name__ == "__main__":
    O = Metrics(y=[0, 1, 2], yhat=[0.1, 1.1, 2])
    