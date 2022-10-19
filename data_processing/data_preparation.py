import jieba 
import swifter
import logging
import pandas as pd 
import numpy as np 
from opencc import OpenCC
from sklearn.model_selection import train_test_split

from utils.reading_local_file import write_json

cc = OpenCC('tw2s')

class Word_segmentation(object):
    def __init__(self,usedict = None, add_words =None):
        
        self.jieba = jieba
        
        if usedict  is not None:
            jieba.load_userdict(usedict) 

        if add_words is not None:
            for word in add_words:
                jieba.add_word(word, freq=None, tag=None)
    
    def cut(self,sentence):
        output = self.jieba.lcut(sentence,HMM = False)
        
        return output

# data,text_name,label_name, splie_ratio,is_convert_tw2s = data,'text','sentiment_label',[6,2,2],True
def generating_dataset(data,text_name,label_name,
                       splie_ratio,
                       is_convert_tw2s,
                       args):
    if is_convert_tw2s:
        data[text_name] = data[text_name].swifter.apply(lambda s : cc.convert(s))

    logging.info('===== word_segmentation =====')
    jieba_tokenizer = Word_segmentation(usedict=args['user_dict'])
    data['ws'] = data[text_name].swifter.apply(lambda s : jieba_tokenizer.cut(s))
    
    logging.info('===== splitting dataset =====')
    data = data.reset_index(drop=True)
    dataset_name = ['train','eval','test']
    splie_ratio = dict(zip(dataset_name,splie_ratio))
    
    total_sum = np.sum(list(splie_ratio.values()))
    train_idx,test_idx = train_test_split(data.index, 
                                          test_size = splie_ratio['test']/total_sum,
                                          shuffle=True,
                                          stratify=data[label_name])
    
    total_sum = splie_ratio['train'] + splie_ratio['eval']
    train_idx,eval_idx = train_test_split(train_idx, 
                                          test_size = splie_ratio['eval']/total_sum,
                                          shuffle=True,
                                          stratify=data[label_name][train_idx])
    logging.info('***** train = {} ; eval = {} ; test = {} *****'.format(len(train_idx),len(eval_idx),len(test_idx)))
    
    logging.info('===== saving split dataset =====')
    folder_path = args['dataset_folder']
    data.iloc[train_idx,:].to_csv(folder_path + '/train_dataset.csv',index = False)
    data.iloc[eval_idx,:].to_csv(folder_path + '/eval_dataset.csv',index = False)
    data.iloc[test_idx,:].to_csv(folder_path + '/test_dataset.csv',index = False)

    saving_json = {}
    saving_json['is_convert_tw2s'] = is_convert_tw2s
    saving_json['splie_ratio'] = splie_ratio
    write_json(saving_json,folder_path + '/data_info.json')


if __name__ == "__main__":
    pass