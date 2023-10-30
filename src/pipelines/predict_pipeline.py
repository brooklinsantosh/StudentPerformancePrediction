import sys
import pandas as pd
from dataclasses import dataclass

import config
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

@dataclass    
class CustomData:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int
    
class PredictPipeline:
    def __init__(self, data: CustomData) -> None:
        self.data = data
    
    def get_data_as_dataframe(self):
        try:
            print(self.data.__dict__)
            return pd.DataFrame(self.data.__dict__, index=[0])
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict(self,features):
        model = load_object(config.MODEL_FILE_PATH)
        preprocessor = load_object(config.PREPROCESSOR_OBJ_PATH)
        try:
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)
        