import logging
import shutil
import configparser
from pathlib import Path
from datetime import datetime
from utils.logger import Logger
from utils.converfunc import Convert2dict
from utils.reading_local_file import create_folder, read_json
from .set_config import TrainingState, TrainingControl

# config_path = 'C:\\Users\\cloudy822\\Desktop\\sentiment_analysis\\config.ini'
class ConfigParser:
    def __init__(self, config_path, modification = None, resume = None, run_id = None):
        
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        self.config.set('training_config', 'batch_size',\
            self.config.get('data_loader','batch_size'))
        self.config.set('training_config', 'num_labels',\
            self.config.get('model','num_labels'))
        
        saving_path = Path(self.config.get('basic_settings','saving_path'))
        project_name = self.config.get('basic_settings','project_name')
        save_pretained_name=self.config.get('log','save_pretained_name')

        mode = self.config.get('basic_settings','mode')
        self.model_config_path = self.config.get('model','pretrained_model_name_or_path') + 'config.json'
        self.model_path = self.config.get('model','pretrained_model_name_or_path') 

        if run_id is None :
            run_id = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

        self.saving_folder =  saving_path / project_name / run_id       
        create_folder(self.saving_folder)
        self.save_pretrained_path = self.saving_folder / save_pretained_name
        
        self.logger = Logger(mode, self.saving_folder, mode + '_' + run_id + '.log').get_logger(mode)
        
        shutil.copytree(
            self.model_path,
            self.save_pretrained_path,
            dirs_exist_ok=True
        )
        shutil.copy(
            config_path,
            self.saving_folder / config_path 
        )

        self.config.set('log', 'save_pretrained_path', str(self.save_pretrained_path))

    @property
    def dataloader_config(self):
        return Convert2dict(self.config, 'data_loader')

    @property
    def model_basic_config(self):
        return Convert2dict(self.config, 'model')
    
    @property
    def model_config(self):
        return read_json(self.model_config_path)

    @property
    def training_config(self):
        return Convert2dict(self.config, 'training_config')
    
    @property
    def training_state(self):
        return TrainingState()
    
    @property
    def training_control(self):
        return TrainingControl()
    
    @property
    def callback_config(self):
        return Convert2dict(self.config, 'callback')
    
    @property
    def log_config(self):
        return Convert2dict(self.config, 'log')



if __name__ == '__main__':
    import os 
    os.chdir('C:\\Users\\cloudy822\\Desktop\\sentiment_analysis\\')
    config = ConfigParser('config.ini')
    logging.info('##### Data Pre-Processing #####')
