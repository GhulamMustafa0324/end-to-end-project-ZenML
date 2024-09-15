import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Trains the model.

    Args:
        data: pd.DataFrame: Data to be trained.
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            mlflow.sklearn.autolog()
            trained_model = model.train(X_train, y_train)
            logging.info("Model trained successfully.")
            return trained_model
        else:
            raise ValueError("Model name not found.")
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e
