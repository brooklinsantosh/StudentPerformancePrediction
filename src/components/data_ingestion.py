# Importing required libraries
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#Importing from own modules
from src.exception import CustomException
from src.logger import logging
import config

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',config.TRAIN_DATA_FILE_NAME)
    test_data_path: str = os.path.join('artifacts',config.TEST_DATA_FILE_NAME)
    raw_data_path: str = os.path.join('artifacts',config.RAW_DATA_FILE_NAME)

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> None:
        logging.info("Entered the data ingestion method or component.")
        try:
            df = pd.read_csv(config.DATA_PATH)
            logging.info("Read the data as dataframe.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split intiated.")
            train_set, test_set = train_test_split(
                df,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed.")

        except Exception as e:
            raise CustomException(e,sys)
