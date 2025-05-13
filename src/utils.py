import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import src.exception

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise ConnectionRefusedError(e,sys)
    

def evaluate_model(x_train,x_test,y_train,y_test,models):
    try:
        report = {}
        for name,model in models.items():
            #model = list(model.key())[i]
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            y_train_score = r2_score(y_train, y_train_pred)
            y_test_score = r2_score(y_test, y_test_pred)

            report[name] = y_test_score

        return report

    except Exception as e:
        raise ConnectionRefusedError(e,sys)
        