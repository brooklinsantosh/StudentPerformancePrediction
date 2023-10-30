import sys
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException


class TrainPipeline:
    def __init__(self) -> None:
        self.data_ingestion_obj = DataIngestion()
        self.data_transformation_obj = DataTransformation()
        self.model_trainer_obj = ModelTrainer() 
    def train(self)-> None:
        try:
            self.data_ingestion_obj.initiate_data_ingestion()
            dt_result = self.data_transformation_obj.initiate_data_transformation(
                DataIngestionConfig.train_data_path,
                DataIngestionConfig.test_data_path
            )
            self.model_trainer_obj.initiate_model_trainer(dt_result.train_arr, dt_result.test_arr)
        except Exception as e:
            raise CustomException(e,sys)





