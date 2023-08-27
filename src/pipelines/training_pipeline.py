import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    train_data_path,test_data_path = data_ingestion_obj.initiate_data_ingestion()
    data_transfomation_obj = DataTransformation()
    train_arr,test_arr,_ = data_transfomation_obj.initiate_data_transformation(train_data_path,test_data_path)
    model_training_obj = ModelTrainer()
    model_training_obj.initiate_model_training(train_arr, test_arr)


