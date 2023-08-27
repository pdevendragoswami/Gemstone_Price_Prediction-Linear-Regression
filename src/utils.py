import os
import sys 
from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
import numpy as np

def save_obj(file_path,obj):
    try:
        logging.info("Save Obj is initiated")
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
        
        logging.info('Obj is saved')


    except Exception as e:
        logging.info('There is some issue in save obj function')
        raise CustomException(e,sys)

def evaluation_metrics(models,X_train,X_test,y_train,y_test):
    try:
        logging.info('Model Evaluation is started')
        reports = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            logging.info('Model fitted')
            model.fit(X_train,y_train)
            
            y_pred = model.predict(X_test)
            logging.info('Model has predicted y')

            accuracy = r2_score(y_test,y_pred)
            logging.info('Model checked the accuracy')

            reports[list(models.keys())[i]] = accuracy 
        
        return reports
        
        
    except Exception as e:
        logging.info('There is some issue at evaluation metrics')
        raise CustomException(e,sys)