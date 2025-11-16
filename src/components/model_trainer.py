import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trainedModel_file_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Split Training Data and Test Data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Linear Regression" : LinearRegression(),
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGB Regressor" : XGBRegressor(),
                "Cat Boost Regressor" : CatBoostRegressor(verbose=False),
                "AdaBoost Regressor" : AdaBoostRegressor()
            }
            # -------------------------------
            # 1. Linear Regression
            # -------------------------------
            param_grid_lr = {
                "fit_intercept": [True, False]
            }

            # -------------------------------
            # 2. Lasso Regression
            # -------------------------------
            param_grid_lasso = {
                "alpha": [0.01, 0.1, 1],
                "max_iter": [2000]
            }

            # -------------------------------
            # 3. Ridge Regression
            # -------------------------------
            param_grid_ridge = {
                "alpha": [0.1, 1, 10]
            }

            # -------------------------------
            # 4. KNN Regressor
            # -------------------------------
            param_grid_knn = {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            }

            # -------------------------------
            # 5. Decision Tree Regressor
            # -------------------------------
            param_grid_dt = {
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }

            # -------------------------------
            # 6. Random Forest Regressor
            # -------------------------------
            param_grid_rf = {
                "n_estimators": [50, 100],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }

            # -------------------------------
            # 7. Gradient Boosting Regressor
            # -------------------------------
            param_grid_gb = {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 4]
            }

            # -------------------------------
            # 8. XGBoost Regressor
            # -------------------------------
            param_grid_xgb = {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }

            # -------------------------------
            # 9. CatBoost Regressor
            # -------------------------------
            param_grid_cat = {
                "iterations": [200, 400],
                "learning_rate": [0.03, 0.1],
                "depth": [4, 6]
            }

            # -------------------------------
            # 10. AdaBoost Regressor
            # -------------------------------
            param_grid_ada = {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1]
            }

            # =======================================================
            # ✔ FINAL Combined Dictionary (same format as you wanted)
            # =======================================================
            params = {
                "Linear Regression": param_grid_lr,
                "Lasso": param_grid_lasso,
                "Ridge": param_grid_ridge,
                "K-Neighbors Regressor": param_grid_knn,
                "Decision Tree Regressor": param_grid_dt,
                "Random Forest Regressor": param_grid_rf,
                "Gradient Boosting": param_grid_gb,
                "XGB Regressor": param_grid_xgb,
                "Cat Boost Regressor": param_grid_cat,
                "AdaBoost Regressor": param_grid_ada
            }


            model_report : dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and test dataset.")

            save_object(
                file_path=self.model_trainer_config.trainedModel_file_path,obj=best_model
            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test,predicted)
            return score,best_model_name
        
        except Exception as e:
            raise CustomException(e,sys)