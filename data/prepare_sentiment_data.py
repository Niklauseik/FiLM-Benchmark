import os
import pandas as pd
import requests

def fetch_dataset(split, offset, length=100):
    url = f"https://datasets-server.huggingface.co/rows?dataset=ChanceFocus%2Fflare-fiqasa&config=default&split={split}&offset={offset}&length={length}"
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return response.json()['rows']

def prepare_sentiment_data(data_path):
    splits = ['train', 'valid', 'test']
    for split in splits:
        file_path = f'data/raw/{data_path}/{split}.csv'
        if os.path.exists(file_path):
            continue
        
        all_rows = []
        offset = 0
        while True:
            rows = fetch_dataset(split, offset)
            if not rows:
                break
            all_rows.extend(rows)
            offset += 100
        
        # Convert gold to label
        for row in all_rows:
            row['row']['label'] = row['row']['gold']
            del row['row']['gold']
        
        # Assuming rows is a list of dictionaries
        df = pd.DataFrame([row['row'] for row in all_rows])
        df.to_csv(f'data/raw/{data_path}/{split}.csv', index=False)

