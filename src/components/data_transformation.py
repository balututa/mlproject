import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoding',OneHotEncoder())
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data,test_data):
        try:
            train_data = pd.read_csv(train_data)
            test_data = pd.read_csv(test_data)

            preprocess_obj = self.get_data_transformation_object()

            target_col = 'math_score'

            input_feature_train_data = train_data.drop(columns=[target_col],axis = 1)
            output_feature_train_data = train_data[target_col]

            input_feature_test_data = test_data.drop(columns=[target_col],axis = 1)
            output_feature_test_data = test_data[target_col]

            input_feature_train_arr = preprocess_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocess_obj.transform(input_feature_test_data)


            train_arr = np.c_[input_feature_train_arr, np.array(output_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array(output_feature_test_data)]


            logging.info('saved preprocessing obj')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocess_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e,sys)


