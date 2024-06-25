import os
import pandas as pd
from datasets import load_dataset

def save_split_data(data_path, split, dataset):
    file_path = f'data/raw/{data_path}/test.csv'
    if os.path.exists(file_path):
        return
    
    data = [{'text': row['text'], 'label': row['gold']} for row in dataset]
    df = pd.DataFrame(data)
    
    df.to_csv(f'data/raw/{data_path}/{split}.csv', index=False)

def prepare_claim_analysis_data(data_path):
    for split in ['train', 'validation', 'test']:
        dataset = load_dataset("TheFinAI/cra-portoseguro", split=split, use_auth_token=True)
        save_split_data(data_path, split, dataset)
