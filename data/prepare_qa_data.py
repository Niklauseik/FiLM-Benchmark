import os
import pandas as pd
import requests

def fetch_dataset(split, offset, length=100):
    url = f"https://datasets-server.huggingface.co/rows?dataset=ChanceFocus/flare-finqa&config=default&split={split}&offset={offset}&length={length}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['rows']

def prepare_qa_data(data_path):
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
        
        formatted_rows = []
        for row in all_rows:
            formatted_row = {
                'id': row['row']['id'],
                'query': row['row']['query'],  # Using 'query' instead of 'question'
                'context': row['row']['text'],
                'answer': row['row']['answer']
            }
            formatted_rows.append(formatted_row)
        
        df = pd.DataFrame(formatted_rows)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)

if __name__ == "__main__":
    prepare_qa_data('flare_finqa')
