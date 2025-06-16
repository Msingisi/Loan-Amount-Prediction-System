import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor


class Model(ABC):
    """"Abstract call for all models"""
    
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        pass

class RandomForestModel(Model):
    """
    Random Forest model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model

        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        try:
            reg = RandomForestRegressor(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Training completed")
            return reg
        except Exception as e:
            logging.error("Error in Training model: {}".format(e))
            raise e