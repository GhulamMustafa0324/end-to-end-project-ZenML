import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):
    """
    Abstract Class for all the models
    """

    @abstractmethod
    def train(self,X_train, y_train):
        """
        Trains the model
        
        Args:
            X_train: pd.DataFrame: Training data
            y_train: pd.Series: Training labels
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self,X_train, y_train,**kwargs):
        """
        Trains the Linear Regression Model
        
        Args:
            X_train: pd.DataFrame: Training data
            y_train: pd.Series: Training labels
        Returns:
            None
        """
        logging.info('Training Linear Regression Model...')
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info('Model trained successfully.')
            return reg
        except Exception as e:
            logging.error('Error in training model: {}'.format(e))
            raise e