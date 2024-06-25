import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

def fetch_dataset(split, offset=0, length=100):
    url = f"https://datasets-server.huggingface.co/rows?dataset=daishen/cra-ccfraud&config=default&split={split}&offset={offset}&length={length}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['rows']

def save_split_data(data_path, split, rows):
    data = [{'text': row['row']['text'], 'label': row['row']['gold']} for row in rows if row['row']['gold'] is not None]
    df = pd.DataFrame(data)
    df.to_csv(f'data/raw/{data_path}/{split}.csv', index=False)

def prepare_fraud_detection_data(data_path):
    file_path = f'data/raw/{data_path}/test.csv'
    if os.path.exists(file_path):
        return
    # Fetch all train data
    all_rows = []
    offset = 0
    while True:
        rows = fetch_dataset('train', offset=offset, length=100)
        if not rows:
            break
        all_rows.extend(rows)
        offset += 100

    # Split the data into train and test sets
    data = [{'text': row['row']['text'], 'label': row['row']['gold']} for row in all_rows if row['row']['gold'] is not None]
    df = pd.DataFrame(data)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Fetch all validation data
    all_val_rows = []
    offset = 0
    while True:
        rows = fetch_dataset('validation', offset=offset, length=100)
        if not rows:
            break
        all_val_rows.extend(rows)
        offset += 100

    val_data = [{'text': row['row']['text'], 'label': row['row']['gold']} for row in all_val_rows if row['row']['gold'] is not None]
    val_df = pd.DataFrame(val_data)

    # Save the splits
    train_df.to_csv(f'data/raw/{data_path}/train.csv', index=False)
    val_df.to_csv(f'data/raw/{data_path}/validation.csv', index=False)
    test_df.to_csv(f'data/raw/{data_path}/test.csv', index=False)

