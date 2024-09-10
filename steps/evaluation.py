import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
from src.evaluation import MSE, R2, RMSE

@step
def evaluate_model(model: RegressorMixin, X_test:pd.DataFrame,y_test:pd.DataFrame) -> Tuple[
    Annotated[float, 'r2'],
    Annotated[float, 'rmse']
]:
    """
    Evaluates the model using the test data.
    
    Args:
        model: RegressorMixin: Trained model
        X_test: pd.DataFrame: Testing data
        y_test: pd.DataFrame: Testing labels
    
    """
    
    logging.info('Evaluating model...')
    try:
        predictions = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calcuate_scores(y_test, predictions)
        R2_class = R2()
        r2 = R2_class.calcuate_scores(y_test, predictions)
        RMSE_class = RMSE()
        rmse = RMSE_class.calcuate_scores(y_test, predictions)
        
        return r2,  rmse
    except Exception as e:
        logging.error('Error in evaluating model: {}'.format(e))
        raise e
        