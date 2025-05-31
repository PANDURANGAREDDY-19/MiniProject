import sys
import os
import dill

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,Y_train,X_test,Y_test,models,parameters):
    try:
        report = {}

        for model_name, model in models.items():
            # Get parameters for this model
            param_grid = parameters.get(model_name, {})
        
            # If parameters exist, use GridSearchCV
            if param_grid:
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )
                grid.fit(X_train, Y_train)
                model = grid.best_estimator_
                print(f"Best parameters for {model_name}: {grid.best_params_}")
            else:
                # Just fit the model with default parameters
                model.fit(X_train, Y_train)
        
            # Make predictions and calculate f1 score
            y_pred = model.predict(X_test)
            f1 = f1_score(Y_test, y_pred)
            report[model_name] = f1
    
        return report

    except Exception as e:
        raise CustomException(e, sys)