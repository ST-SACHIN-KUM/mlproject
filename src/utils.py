import os
import sys
import dill     #libraries to create pickle file
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(models, X_train,y_train,X_test,y_test, params = None):
    
    result = {}
    fitted_models = {}
    for name, model in models.items():
        if params and name in params:
            random_search = RandomizedSearchCV(model, params[name], n_iter = 5, cv=3, scoring='r2', n_jobs=-1)
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train, y_train)
            
        y_pred = best_model.predict(X_test)
        score = r2_score(y_test, y_pred)
        result[name] = score
        fitted_models[name] = best_model
    return result,fitted_models


def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)