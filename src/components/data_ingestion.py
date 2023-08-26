import os
import sys
from src.exception import CustomException
from src.logger  import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split



@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts','raw.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Data ingestion is initiated')
            df = pd.read_csv(os.path.join('notebooks','data','data.csv'))
            #print(df.head())

            logging.info('Data read as dataframe')

            
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok = True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info('Data saved in file raw.csv')

            logging.info('Data splited into training and testing')

            training_data,testing_data = train_test_split(df,test_size=0.25,random_state=1)
                   
            training_data.to_csv(self.data_ingestion_config.train_data_path,index=False)
            logging.info('Training data is saved in train.csv')
            testing_data.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info('Testing data is saved in test.csv')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('There is some issue in data ingestion')
            raise CustomException(e, sys)




