import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def save_preprocessed_data(encodings, labels, save_path):
    torch.save((encodings, labels), save_path)

def load_preprocessed_data(load_path):
    encodings, labels = torch.load(load_path)
    return encodings, labels

def save_model(model, tokenizer, model_dir):
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

def load_model(model_name, model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def get_or_preprocess_data(data_path, dataset_class, dataset_name):
    dataset_file_path = f"data/processed/{data_path}/{dataset_name}.pt"
    if os.path.exists(dataset_file_path):
        encodings, labels = load_preprocessed_data(dataset_file_path)
        dataset = dataset_class(encodings, labels)
        return 1, dataset
    else:
        return 0, None

def load_and_combine_datasets(task_name):
    train_data_path = f"data/raw/{task_name}/train.csv"
    valid_data_path = f"data/raw/{task_name}/valid.csv"
    test_data_path = f"data/raw/{task_name}/test.csv"

    train_data = pd.read_csv(train_data_path)
    valid_data = pd.read_csv(valid_data_path)
    test_data = pd.read_csv(test_data_path)

    combined_data = pd.concat([train_data, valid_data, test_data], ignore_index=True)
    return combined_data
