import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging

import src.exception

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise ConnectionRefusedError(e,sys)
    

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import sys
from src.exception import CustomException

def evaluate_model(x_train, x_test, y_train, y_test, models, params):
    try:
        report = {}

        for name, model in models.items():
            model_params = params.get(name, {})  # Safely get params for this model

            if model_params:
                gs = GridSearchCV(model, model_params, cv=3, n_jobs=-1, scoring='r2')
                gs.fit(x_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(x_train, y_train)
                best_model = model
            logging.info(f"Best parameters for {name}: {gs.best_params_}")

            y_test_pred = best_model.predict(x_test)
            test_score = r2_score(y_test, y_test_pred)

            report[name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
