import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f'Reading data from {self.data_path}')
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingest data from a CSV file.
    
    Args:
        data_path: Path to the CSV file.
    Returns:
        A pandas DataFrame containing the data.
    
    """
    try:
        return IngestData(data_path).get_data()
    except Exception as e:
        logging.error(f'Error reading data from {data_path}: {e}')
        raise e
    