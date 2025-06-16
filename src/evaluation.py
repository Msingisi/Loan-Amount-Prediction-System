import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the scores for the model

        Args:
            y_true: True labels
            y_pred: Predicted labels
        return:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred:np.ndarray):
         try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
         except Exception as e:
             logging.error("Error in Calculating MSE: {}".format(e))


class R2(Evaluation):
    """
    Evaluation Strategy that uses R2 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred:np.ndarray):
         try:
            logging.info("Calculating MSE")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 score: {}".format(r2))
            return r2
         except Exception as e:
             logging.error("Error in calcualting R2 Score: {}".format(e))
             raise e
         

class RMSE(Evaluation):
    """
    Evaluation Strategy that uses RMSE
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred:np.ndarray):
         try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE: {}".format(rmse))
            return rmse
         except Exception as e:
             logging.error("Error in calcualting RMSE: {}".format(e))
             raise e