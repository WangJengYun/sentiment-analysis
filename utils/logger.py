import logging
from pathlib import Path
from logging import config

class Logger(object):

    def __init__(self,mode, folder_name, logging_name):
        self.config_path = './utils/logging_config.conf'
        self.folder_name = folder_name
        self.logging_name = logging_name
    
    def _set_log_config(self):   
        if isinstance(self.folder_name, Path):
            logging.folder_name = str(self.folder_name)
        else:    
            logging.folder_name = self.folder_name     
        logging.logging_name = self.logging_name
        config.fileConfig(self.config_path,disable_existing_loggers=False)
    
    def get_logger(self,template_name='root'):
        
        self._set_log_config()        
        logging.getLogger(template_name)

        return 