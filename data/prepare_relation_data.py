import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

def fetch_dataset(split, offset, length=100):
    url = f"https://datasets-server.huggingface.co/rows?dataset=TheFinAI/flare-finred&config=default&split={split}&offset={offset}&length={length}"
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return response.json()['rows']

def prepare_relation_data(data_path):
    split = 'test'
    raw_dir = f'data/raw/{data_path}'
    file_path = os.path.join(raw_dir, f'{split}.csv')
    if os.path.exists(file_path):
        print(f"{split}.csv already exists, skipping download.")
    else:
        all_rows = []
        offset = 0
        while True:
            rows = fetch_dataset(split, offset)
            if not rows:
                break
            all_rows.extend(rows)
            offset += 100

        # Assuming rows is a list of dictionaries
        df = pd.DataFrame([row['row'] for row in all_rows])

        # Split into train, validation, and test sets
        train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Save the splits
        train_df.to_csv(os.path.join(raw_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(raw_dir, 'valid.csv'), index=False)
        test_df.to_csv(os.path.join(raw_dir, 'test.csv'), index=False)
