from cnnClassifier.constant import *
from cnnClassifier.utils.common import read_yaml,create_directories
from cnnClassifier import logger
import gdown
import zipfile
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self)-> str:
        try:
            data_set_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            create_directories([self.config.root_dir])
            file_id = data_set_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir)
            logger.info(f"Downloa data from {data_set_url} into {zip_download_dir} completed")
        except Exception as e:
            raise e
    
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        create_directories([unzip_path])
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)