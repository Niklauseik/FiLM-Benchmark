import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def prepare_numeric_labeling_data(data_path):
    if all(os.path.exists(f'data/raw/{data_path}/{split}.csv') for split in ['train', 'valid', 'test']):
        print("Data already exists. Skipping download.")
        return

    dataset = load_dataset("TheFinAI/flare-fnxl")

    df = pd.DataFrame(dataset['test'])
    df = df[['token', 'label']]  

    # Split the data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    # Save the split data to CSV files
    train_df.to_csv(f'data/raw/{data_path}/train.csv', index=False)
    valid_df.to_csv(f'data/raw/{data_path}/valid.csv', index=False)
    test_df.to_csv(f'data/raw/{data_path}/test.csv', index=False)