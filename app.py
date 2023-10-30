from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig, DataTransformationResult
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


data_ingestion_obj = DataIngestion()
data_ingestion_obj.initiate_data_ingestion()

data_transformation_obj = DataTransformation()
dt_result = data_transformation_obj.initiate_data_transformation(
    DataIngestionConfig.train_data_path,
    DataIngestionConfig.test_data_path
)

model_trainer = ModelTrainer() 
model_trainer.initiate_model_trainer(dt_result.train_arr, dt_result.test_arr)


