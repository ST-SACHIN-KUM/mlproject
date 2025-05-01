# Basic Import
import os
import sys
import pandas as pd
from dataclasses import dataclass
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import warnings

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Start splitting and model training")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Linear Regression" : LinearRegression(),
                "XGBoost Regression" : XGBRegressor()
            }
            params = {
                "Random Forest" : {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, None]},
                "XGBoost Regression": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 10]}
            }
            model_report, fitted_models = evaluate_models(models, X_train,y_train,X_test,y_test, params)
            best_model_name = max(model_report, key=model_report.get)
            for model_name, score in model_report.items():
                logging.info(f"Model: {model_name}, R² Score: {score:.4f}")
            
            best_model = fitted_models[best_model_name]
            logging.info(f"Best Model saved {best_model}")
            save_object("artifacts/best_model.pkl", best_model)
        except Exception as e:
            raise CustomException(e,sys)
