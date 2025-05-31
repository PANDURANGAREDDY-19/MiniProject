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
from sklearn.model_selection import GridSearchCV

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
            parameters = {
                "Random Forest Classifier": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10]
                    },
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "solver": ["liblinear", "lbfgs"]
                    },
                "Decision Tree Classifier": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10]
                    },
                "AdaBoost Classifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0]
                    },
                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                    },
                "Gradient Boost Classifier": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                    },
                "XGB Classifier": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7]
                    },
                "CatBoost Classifier": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "iterations": [50, 100, 200],
                    "depth": [4, 6, 8]
                    }
            }
            model_report : dict = evaluate_model(X_train = X_train,Y_train = Y_train,X_test = X_test,Y_test = Y_test,models=models,parameters=parameters)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if(best_model_score < 0.7):
                raise CustomException("No Best model found")
            logging.info("Best Model found for dataset")

            save_object(file_path=self.ModelTrainerConfig.trained_model_file_path,obj=best_model)
            if best_model_name in parameters:
                grid = GridSearchCV(
                estimator=models[best_model_name],
                param_grid=parameters[best_model_name],
                cv=5,
                scoring='f1'
                )
                grid.fit(X_train, Y_train)
                best_model = grid.best_estimator_
                print(f"Best parameters: {grid.best_params_}")
                predicted = best_model.predict(X_test)

                f1_Score_value = f1_score(Y_test,predicted)
            return f1_Score_value

        except Exception as e:
            raise CustomException(e,sys)        
        