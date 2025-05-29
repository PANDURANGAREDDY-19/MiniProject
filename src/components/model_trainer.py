import sys
import os

from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.ModelTrainerConfig = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Train and Test Data for input")
            X_train,Y_train,X_test,Y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest Classifier" : RandomForestClassifier(),
                "Logistic Regression" : LogisticRegression(),
                "Decision Tree Classifier" : DecisionTreeClassifier(),
                "AdaBoost Classifier" : AdaBoostClassifier(),
                "K-Neighbors Classifier" : KNeighborsClassifier(),
                "Gradient Boost Classifier" : GradientBoostingClassifier(),
                "XGB Classifier" : XGBClassifier(),
                "CatBoost Classifier" : CatBoostClassifier(
                    learning_rate=0.1,
                    iterations=100,
                    verbose=0
                ),
            }

            model_report : dict = evaluate_model(X_train = X_train,Y_train = Y_train,X_test = X_test,Y_test = Y_test,models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if(best_model_score < 0.7):
                raise CustomException("No Best model found")
            logging.info("Best Model found for dataset")

            save_object(file_path=self.ModelTrainerConfig.trained_model_file_path,obj=best_model)
            predicted = best_model.predict(X_test)

            f1_Score_value = f1_score(Y_test,predicted)
            return f1_Score_value

        except Exception as e:
            raise CustomException(e,sys)        
        