import sys
import os
import dill

from sklearn.metrics import f1_score

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,Y_train,X_test,Y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,Y_train)
            y_test_pred = model.predict(X_test)

            test_model_score = f1_score(Y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            
        return report

    except Exception as e:
        raise CustomException(e, sys)

def LoadObject(file_path):
    try:
        with open(file_path,'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e,sys)