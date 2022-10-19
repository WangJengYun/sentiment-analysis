import os 
os.chdir('C:\\Users\\cloudy822\\Desktop\\sentiment_analysis\\')
import logging
from configuration import ConfigParser
from data_loader import SentimentDataLoader
from model import BertForSequenceClassifier
from trainer import Trainer
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    config = ConfigParser('config.ini') 

    logger.info('##### loading dataset  #####')
    dataloader = SentimentDataLoader(**config.dataloader_config)

    logger.info('##### loading model    #####')
    model = BertForSequenceClassifier.from_pretrained(**config.model_basic_config)
    # model = BertForSequenceClassification.from_pretrained(**config.model_basic_config)
    logger.debug(model)

    logger.info('##### creating trainer #####')
    training_process = Trainer(
        config = config, 
        model = model, 
        train_dataloader = dataloader.train_dataloader, 
        valid_dataloader = dataloader.valid_dataloader
    )
    logger.info('##### model training   #####')
    training_process.train()