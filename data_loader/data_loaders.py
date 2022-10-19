import pandas as pd 
from .dataset import SentimentDataset
from .base_dataloader import BaseDataLoader

class SentimentDataLoader(BaseDataLoader):
    def __init__(
        self,
        file_path, 
        token_path, 
        maxlen,
        batch_size, 
        split_weight, 
        num_workers
    ):
    
        self.file_path = file_path
        self.token_path = token_path
        
        senti_table = pd.read_csv(file_path)
        senti_data  = (senti_table['text'].tolist(), senti_table['sentiment_label'].tolist())
        self.dataset = SentimentDataset(senti_data, token_path = token_path, maxlen = maxlen)
        super().__init__(self.dataset , batch_size, split_weight, num_workers,collate_fn = self.dataset.collate_fn)
