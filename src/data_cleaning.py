import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True
            )
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True
            )
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True
            )
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(e)
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Data division strategy which divides the data into training and testing data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into training and testing data.
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
        
class DataCleaning:
    """
    class for cleaning the data which processes the data and divides it into training and testing data. 
    """
    
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self):
        """
        Processes the data and divides it into training and testing data.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("error in handling data: {}".format(e))
            raise e
        
# if __name__ == "__main__":
#     data = pd.read_csv("data/olist_order_items_dataset.csv")
#     data_cleaning = DataCleaning(data, DataPreprocessStrategy())
#     data_cleaned = data_cleaning.handle_data()
#     print(data_cleaned.head())
    
#     data_cleaning = DataCleaning(data_cleaned, DataDivideStrategy())
#     X_train, X_test, y_train, y_test = data_cleaning.handle_data()
#     print(X_train.head())
#     print(y_train.head())
#     print(X_test.head())
#     print(y_test.head())
        