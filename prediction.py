import pandas as pd 
from tqdm import tqdm
from data_loader import SentimentDataset
from model import BertForSequenceClassifier

import torch
from torch.utils.data import DataLoader

device = 'cpu'
model_path = 'C:\\Users\\cloudy822\\Desktop\\sentiment_analysis\\TTT\\sentiment_bert_model'
token_path = 'C:\\Users\\cloudy822\\Desktop\\sentiment_analysis\\model\\pretrained_model\\ckip-bert-base-chinese'

model = BertForSequenceClassifier.from_pretrained(pretrained_model_name_or_path = model_path,num_labels = 2)

raw_data = pd.read_csv('source/dataset/original_data/sentiment_data.csv')[:10]

senti_data  = (raw_data['text'].tolist(), raw_data['sentiment_label'].tolist())
input_dataset = SentimentDataset(senti_data, token_path = token_path)
dataloader = DataLoader(input_dataset, batch_size=2, collate_fn=input_dataset.collate_fn)

total_logits = []
total_target = []
batchs_iterator = tqdm(dataloader, unit = 'batch', position = 0)  
for batch_idx, (_, data, target) in enumerate(dataloader):
    data = {k:v.to(device) for k,v in data.items()}
    target = target.to(device)
    with torch.no_grad():
        output = model(**data, return_dict = True)  
    total_logits.append(output.logits)
    total_target.append(target)

total_logits = torch.cat(total_logits, dim = 0)
pred_labels = torch.argmax(total_logits, dim=1).numpy()
true_labels = torch.cat(total_target).numpy()

raw_data['sentiment_pred_label'] = pred_labels
