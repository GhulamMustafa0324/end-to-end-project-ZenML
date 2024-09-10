import logging
from abc import ABC, abstractmethod
import numpy as np

from sklearn.metrics import mean_squared_error,r2_score

class Evaluation(ABC):
    """
    Abstract Class for all the models
    """

    @abstractmethod
    def calcuate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):    
        """
        calculate the scores of the model

        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Mean Squared Error
    """
    def calcuate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        calculate the scores of the model

        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        Returns:
            None
        """
        logging.info('Calculating Mean Squared Error...')
        try:
            mse = mean_squared_error(y_true, y_pred)
            logging.info('Mean Squared Error calculated successfully: {}.'.format(mse))
            return mse
        except Exception as e:
            logging.error('Error in calculating Mean Squared Error: {}'.format(e))
            raise e
        
class R2(Evaluation):
    """
    Evaluation strategy that used R2 Score
    """
    
    def calcuate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        calculate the scores of the model

        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        Returns:
            None
        """
        logging.info('Calculating R2 Score...')
        try:
            r2 = r2_score(y_true, y_pred)
            logging.info('R2 Score calculated successfully: {}.'.format(r2))
            return r2
        except Exception as e:
            logging.error('Error in calculating R2 Score: {}'.format(e))
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation strategy that used RMSE Score
    """
    
    def calcuate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        calculate the scores of the model

        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        Returns:
            None
        """
        logging.info('Calculating Root Mean Squared Error...')
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info('Root Mean Squared Error calculated successfully: {}.'.format(rmse))
            return rmse
        except Exception as e:
            logging.error('Error in calculating Root Mean Squared Error: {}'.format(e))
            raise e
        
