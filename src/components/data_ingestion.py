import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component")
        
        try:
            '''
            A little bit of preprocessing was done on this dataset before starting the work here.
            There were a few categorical columns which had some categories that were occuring just once or
            twice. This was causing unequal number of categories/unknow categories in the train and test datasets
            after the prerocessor was applied on these datasets in the data transformation file.
            It was a really big problem. So, all the categories in all the categorical columns that were occuring
            less than 10 times were replaced with missing values. This was done to ensure data consistency/equal data 
            distribution in the train and test datasets after train test split. After this, it was found that there
            was constant column in the dataset i.e. Utiilities which was not providing any useful info so that column
            will also be dropped after reading this dataset here.
            
            Furthermore, it was found during data transformation that the majority of data points in the column
            'Pool Area' were 0 and with a random state of 3 in the train test split all the entries in the pool area
            column of the test data set were 0 which was again an issue (constant columns). But if i went along with
            this process and dropped it from the test dataset, that would have caused unequal no of columns in the train
            and test dataset. With a change in the random-state from 3 to 8 it was continuously found that there were
            1 or maximum of three constant columns in the x_test after applying the preprocessor on x_test_transf
            (i.e. preprocessor.transform(x_test_transf). However this issue was addressed when i reached the random state
            of 9. With this there was no constant column in x_train and x_test before and after applying the preprocessor. 
            '''

            df = pd.read_csv('notebook/ames_housing_for_model_deployment.csv')

            logging.info('Read the dataset as dataframe')

            df = df.drop (['Alley', 'Pool_QC', 'Fence', 'Misc_Feature', 'Order', 'PID', 'Utilities'], axis=1)

            logging.info("Six Columns Dropped Having More Than 90 percent missing values and a constant column dropped")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Raw Data file saved")
            
            train, test = train_test_split (df, test_size=0.2, random_state=9)
            
            logging.info("Train Test Split Completed")

            train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            logging.info("Train Set Saved")

            test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Test Set Saved")

            logging.info("Ingestion of the data has completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    # obj.initiate_data_ingestion()

    # these pieces of codes are added later, after data creating data transformation
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))