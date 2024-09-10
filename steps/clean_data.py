import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreprocessStrategy, DataDivideStrategy, DataCleaning
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    """
    Cleans the data and divides it into training and testing data.
    
    Args:
        data: pd.DataFrame: Data to be cleaned and divided.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training and testing data.
    """
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning and division complete.")
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e
        