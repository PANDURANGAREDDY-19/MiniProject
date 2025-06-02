import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import LoadObject

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            logging.info("Prediction Pipeline started")
            model = LoadObject(file_path = model_path)
            preprocessor = LoadObject(file_path=preprocessor_path)
            data_scale = preprocessor.transform(features)
            predict = model.predict(data_scale)
            logging.info("Values are predicted sucessfully")
            return predict
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
            self,
            Sex: str,
            Cholesterol:int,
            Age: int,
            ChestPainType: str,
            ST_Slope: str,
            Oldpeak: float,
            ExerciseAngina: str,
            RestingECG: str,
            FastingBS: int,
            RestingBP: int,
            MaxHR: int
    ):
        self.Sex = Sex
        self.Age = Age
        self.Oldpeak = Oldpeak
        self.ExerciseAngina = ExerciseAngina
        self.FastingBS = FastingBS
        self.RestingBP = RestingBP
        self.RestingECG = RestingECG
        self.MaxHR = MaxHR
        self.Cholesterol = Cholesterol
        self.ChestPainType = ChestPainType
        self.ST_Slope = ST_Slope
        
    def get_data_as_dataframe(self):
        try:
            data_input = {
                "Age":[self.Age],
                "Sex":[self.Sex],
                "Cholesterol":[self.Cholesterol],
                "RestingECG":[self.RestingECG],
                "RestingBP":[self.RestingBP],
                "FastingBS":[self.FastingBS],
                "ChestPainType":[self.ChestPainType],
                "Oldpeak":[self.Oldpeak],
                "ExerciseAngina":[self.ExerciseAngina],
                "MaxHR":[self.MaxHR],
                "ST_Slope":[self.ST_Slope]
            }
            return pd.DataFrame(data_input)
        except Exception as e:
            raise CustomException(e,sys)
        