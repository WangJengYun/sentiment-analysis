import numpy as np 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    
    def __init__(self, 
                dataset, 
                batch_size, 
                split_weight, 
                num_workers, 
                collate_fn):

        self.n_samples = len(dataset)
        if isinstance(split_weight, list):
            assert  len(split_weight) == 3, 'Please check the length of split_weight'
            split_weight = np.array(split_weight)/sum(split_weight)
        else : 
            raise TypeError('Please check split_weight param, it must be list')
        
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'pin_memory' : True,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        
        self.train_sampler, self.valid_sampler, self.test_sampler = self._split_sampler(split_weight)
        
        super().__init__(**self.init_kwargs)

    def _split_sampler(self, split_weight):
        
        idx_full = np.arange(self.n_samples)
        
        np.random.seed(0)
        np.random.shuffle(idx_full)

        len_train = int(self.n_samples * split_weight[0])
        len_valid = int(self.n_samples * split_weight[1])
        len_test = int(self.n_samples * split_weight[2])
        len_split =  len_train + len_valid + len_test
        
        assert len_split <= self.n_samples        
        if len_split < self.n_samples:
            len_train += self.n_samples - len_split
            len_split = len_train + len_valid + len_test
        
        train_idx = idx_full[0:len_train]
        valid_idx = idx_full[len_train:(len_train+len_valid)]
        test_idx = idx_full[(len_train+len_valid):len_split]
        
        train_sampler = SubsetRandomSampler(train_idx) if len_train > 0 else None 
        valid_sampler = SubsetRandomSampler(valid_idx) if len_valid > 0 else None 
        test_sampler = SubsetRandomSampler(test_idx) if len_test > 0 else None 

        return  train_sampler, valid_sampler, test_sampler
    
    @property
    def train_dataloader(self):
        return DataLoader(sampler=self.train_sampler, **self.init_kwargs) \
                    if self.train_sampler is not None else None
    @property
    def valid_dataloader(self):
        return DataLoader(sampler=self.valid_sampler, **self.init_kwargs) \
                    if self.valid_sampler is not None else None
    @property
    def test_dataloader(self):
        return DataLoader(sampler=self.test_sampler, **self.init_kwargs) \
                    if self.test_sampler is not None else None

if __name__ == '__main__':
    import os 
    import pandas as pd 
    os.chdir('\\\\cloudsys/user_space/Cloud/Side_Project/sentiment_analysis')
    from data_loader.dataset import  SentimentDataset
    data = pd.read_csv('source/dataset/original_data/sentiment_data.csv')
    
    token_path = './source/pretrain-model/ckip-transformers/bert-base-chinese'
    senti_data  = (data['text'].tolist(), data['sentiment_label'].tolist())
    input_dataset = SentimentDataset(senti_data, token_path = token_path, device = 'cpu')
    #dataloader = DataLoader(input_dataset, batch_size=32, num_workers = 3, collate_fn=input_dataset.collate_fn)
    #AA = list(dataloader)
    BDL = BaseDataLoader(input_dataset,batch_size = 32, split_weight = [2,1,1], num_workers = 0, collate_fn = input_dataset.collate_fn)
    testBDL = BDL.test_dataloader()

    AA = list(testBDL)