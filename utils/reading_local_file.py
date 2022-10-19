import os 
import json
import yaml

def read_yaml(file_path = './config/training_config.yaml'):
    with open(file_path) as f:
        return yaml.safe_load(f)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)          

def write_json(data,path):
    with open(path, 'w') as f:
        json.dump(data, f)

def read_json(path):
    with open(path) as f:
        return json.load(f)