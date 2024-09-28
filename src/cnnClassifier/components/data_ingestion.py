import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config

    def download_data(self):
        try:
            download_URL = self.config.source_URL
            zipfile_path = self.config.local_data_file
            root_dir = self.config.root_dir

            os.makedirs(root_dir,exist_ok=True)
            logger.info(f"Downloading dataset from {download_URL} into {zipfile_path}")

            file_id = download_URL.split('/')[-2]
            prefix_url = 'https://drive.google.com/uc?export=download&id='
            gdown.download(prefix_url+file_id,zipfile_path)
            logger.info(f"Downloaded dataset from {download_URL} into {zipfile_path}")

        except Exception as e:
            raise e
        
    def extract_data(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)