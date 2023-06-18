import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from catboost import CatBoostRegressor
from sklearn.ensemble import ( AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "XGB Regressor": {
                    'max_depth': [1,2,3,4,5,6,7,8,10,11,12], 
                    'n_estimators':[8,16,32,64,128,256],

                },

                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report : dict=evaluate_models (x_train = x_train, y_train = y_train, x_test = x_test,
                                                 y_test=y_test, models = models, param = params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.5:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_pred_train = best_model.predict(x_train)
            
            y_pred_test = best_model.predict (x_test)
            
            score_train = r2_score (y_train, y_pred_train)
            
            print ("R2 score train:", score_train)
            
            score_test = r2_score (y_test, y_pred_test)
            
            print ("R2 score test:", score_test)
            
            MAE_train = mean_absolute_error (y_train, y_pred_train)
            
            print ('Mean Absolute Error Train:', MAE_train)
            
            MAE_test = mean_absolute_error (y_test, y_pred_test)
            
            print ('Mean Absolute Error Test:', MAE_test)
            
            MSE_train = mean_squared_error (y_train, y_pred_train)
            
            print ('Mean Squared Error Train:', MSE_train)
            
            MSE_test = mean_squared_error (y_test, y_pred_test)
            
            print ('Mean Squared Error Test:', MSE_test)
            
            RMSE_train = np.sqrt (mean_squared_error (y_train, y_pred_train))
            
            print ('Root Mean Squared Error Train:', RMSE_train)
            
            RMSE_test = np.sqrt (mean_squared_error (y_test, y_pred_test))
            
            print ('Root Mean Squared Error Test:', RMSE_test)
            
            return score_train, score_test, MAE_train, MAE_test, MSE_train, MSE_test, RMSE_train, RMSE_test, best_model
            
        except Exception as e:
            raise CustomException(e,sys)