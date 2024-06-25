import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

def fetch_dataset(offset, length=100):
    url = f"https://datasets-server.huggingface.co/rows?dataset=ChanceFocus/flare-finarg-ecc-arc&config=default&split=test&offset={offset}&length={length}"
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return response.json()['rows']

def prepare_relation_cls_data(data_path):
    file_path = f'data/raw/{data_path}/test.csv'
    if os.path.exists(file_path):
        return

    all_rows = []
    offset = 0
    while True:
        rows = fetch_dataset(offset)
        if not rows:
            break
        all_rows.extend(rows)
        offset += 100

    data = [{'text': row['row']['text'], 'gold': row['row']['gold']} for row in all_rows]
    df = pd.DataFrame(data)

    # Split the data into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    train_df.to_csv(f'data/raw/{data_path}/train.csv', index=False)
    valid_df.to_csv(f'data/raw/{data_path}/valid.csv', index=False)
    test_df.to_csv(f'data/raw/{data_path}/test.csv', index=False)
