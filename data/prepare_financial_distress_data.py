import os
import pandas as pd
import requests

def fetch_dataset(split, offset=0, length=100):
    url = f"https://datasets-server.huggingface.co/rows?dataset=ChanceFocus/cra-polish&config=default&split={split}&offset={offset}&length={length}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['rows']

def save_split_data(data_path, split):
    file_path = f'data/raw/{data_path}/test.csv'
    if os.path.exists(file_path):
        return
    
    all_rows = []
    offset = 0
    while True:
        rows = fetch_dataset(split, offset=offset, length=100)
        if not rows:
            break
        all_rows.extend(rows)
        offset += 100

    data = [{'text': row['row']['text'], 'label': row['row']['gold']} for row in all_rows]
    df = pd.DataFrame(data)
    df.to_csv(f'data/raw/{data_path}/{split}.csv', index=False)

def prepare_financial_distress_data(data_path):
    for split in ['train', 'validation', 'test']:
        save_split_data(data_path, split)
