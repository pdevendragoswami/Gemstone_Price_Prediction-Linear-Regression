import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_obj
from src.components.data_ingestion import DataIngestion
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

@dataclass
class DataTransfomationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfomation_config = DataTransfomationConfig()

    def get_data_transformation_obj(self):
        try:
            #segregating numerical columns
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline creation is initiated')

            numerical_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,
                    color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_cols),
                ('categorical_pipeline',categorical_pipeline,categorical_cols)
            ])

            logging.info('Pipeline creation is done successfully')
            return preprocessor
                    
        except Exception as e:
            logging.info('There is some issue at get data transformation part')
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info('Data transformation is initiated')

            logging.info('Reading testing and training data')
            df_train = pd.read_csv(train_data_path)
            df_test = pd.read_csv(test_data_path)

            logging.info('Data reading completed successfully')

            target_feature = 'price'
            drop_features = ['id',target_feature]

            df_train_input_feature = df_train.drop(drop_features,axis=1)
            df_train_target_feature = df_train[target_feature]

            df_test_input_feature = df_test.drop(drop_features,axis=1)
            df_test_target_feature = df_test[target_feature]

            preprocessor_obj = self.get_data_transformation_obj()

            input_feature_train_arr = preprocessor_obj.fit_transform(df_train_input_feature)
            input_feature_test_arr = preprocessor_obj.transform(df_test_input_feature)

            train_arr = np.c_[input_feature_train_arr,df_train_target_feature]
            test_arr = np.c_[input_feature_test_arr,df_test_target_feature]

            logging.info('Data transformation is completed')

            save_obj(
                file_path = self.data_transfomation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj)

            logging.info('Pickle file saved.')

            return(
                train_arr,
                test_arr,
                self.data_transfomation_config.preprocessor_obj_file_path
            )





        except Exception as e :
            logging.info('There is some issue in initiate data transformation')
            raise CustomException(e,sys)

if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    train_data_path,test_data_path = data_ingestion_obj.initiate_data_ingestion()
    data_transfomation_obj = DataTransformation()
    data_transfomation_obj.initiate_data_transformation(train_data_path,test_data_path)


