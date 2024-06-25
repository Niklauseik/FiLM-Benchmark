import os
import pandas as pd
import requests

def fetch_dataset(split, offset, length=100):
    url = f"https://datasets-server.huggingface.co/rows?dataset=TheFinAI/flare-ner&config=default&split={split}&offset={offset}&length={length}"
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return response.json()['rows']

def prepare_ner_data(data_path):
    splits = ['train', 'test', 'valid']
    for split in splits:
        file_path = f'data/raw/{data_path}/{split}.csv'
        if os.path.exists(file_path):
            print(f"{split}.csv already exists, skipping download.")
            continue

        all_rows = []
        offset = 0
        while True:
            rows = fetch_dataset(split, offset)
            if not rows:
                break
            all_rows.extend(rows)
            offset += 100

        # Extract 'row' key from each entry
        data = [row['row'] for row in all_rows]

        # Create a DataFrame and save to CSV
        df = pd.DataFrame(data)
        os.makedirs(f'data/raw/{data_path}', exist_ok=True)
        df.to_csv(file_path, index=False)

