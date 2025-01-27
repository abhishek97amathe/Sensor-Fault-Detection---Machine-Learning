import sys
from typing import List, Dict
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainer:
    models: Dict[str, object] = None

    def __init__(self):
        self.models = {
            "XGBClassifier": XGBClassifier(),
            "SVC": SVC(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier()
        }

    def evaluate_models(self, X, y, models: Dict[str, object]) -> Dict[str, float]:
        try:
            model_report = {}
            for model_name, model in models.items():
                model.fit(X, y)
                y_pred = model.predict(X)
                model_report[model_name] = accuracy_score(y, y_pred)
            return model_report
        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_name: str, best_model_object: object, X_train, y_train):
        try:
            # Implement fine-tuning logic here
            # For example, using GridSearchCV or other hyperparameter tuning methods
            return best_model_object
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_arr: List, test_arr: List) -> float:
        try:
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f"Extracting model config file path")
            model_report: dict = self.evaluate_models(X=x_train, y=y_train, models=self.models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = self.models[best_model_name]
            best_model = self.finetune_best_model(best_model_name=best_model_name, best_model_object=best_model, X_train=x_train, y_train=y_train)
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)
            print(f"best model name {best_model_name} and score: {best_model_score}")
            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")
            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)