import os 
import pandas as pd 
import swifter

from opencc import OpenCC
cc = OpenCC('s2tw')

# folder = negative_path
def reading_file_without_labeling(folder,label):
    files = os.listdir(folder)

    result = pd.DataFrame(columns = ['text','sentiment_label'])
    for file in files :
        file_path = os.path.join(folder,file)
        if file.endswith('xls'):
            data = pd.read_excel(file_path,header=None,names = ['text'])    
            # data['text'] =  data['text'].apply(lambda x : cc.convert(x))
            data['sentiment_label'] = label
            result = pd.concat([result,data],ignore_index = True)

    return result

def readling_file_with_labeling(folder):
    files = os.listdir(folder)
    
    result = pd.DataFrame(columns = ['text','sentiment_label'])
    for file in files:
        file_path = os.path.join(folder,file)
        if file.endswith('data'):
            data = pd.read_table(file_path,sep = '\t',header=None,names = ['text','sentiment_label'])
            result = pd.concat([result,data],ignore_index = True)
    return result

if __name__ == "__main__":
    import os 
    os.chdir('\\\\cloudsys/user_space/Cloud/Side_Project/sentiment_analysis')
    data_labeling_path = 'source/dataset/raw_data/sentiment/data_labeling'
    negative_path = 'source/dataset/raw_data/sentiment/negative'
    positive_path = 'source/dataset/raw_data/sentiment/positive'
    
    sentiment_table = readling_file_with_labeling(data_labeling_path)
    negative_table = reading_file_without_labeling(negative_path,0)
    positive_table = reading_file_without_labeling(positive_path,1)

    sentiment_table = pd.concat([sentiment_table,negative_table,positive_table],ignore_index = True)
    sentiment_table = sentiment_table.drop_duplicates(subset = ['text','sentiment_label']).reset_index(drop=True)

    sentiment_table = sentiment_table[sentiment_table['text'].notnull()]
    sentiment_table['text'] = sentiment_table['text'].astype(str)
    sentiment_table['text'] = sentiment_table['text'].apply(lambda x : cc.convert(x))

    sentiment_table.to_csv('source/dataset/original_data/sentiment_data.csv',index = False)