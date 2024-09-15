import os, sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import shutil

# Ensure the utils module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.file_utils import save_preprocessed_data, save_model, load_model, get_or_preprocess_data

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_data(file_path, tokenizer, max_length, save_path=None):
    df = pd.read_csv(file_path)
    encodings = tokenizer(list(df['text']), truncation=True, padding=True, max_length=max_length)
    labels = list(df['label'])
    if save_path:
        save_preprocessed_data(encodings, labels, save_path)
    dataset = SentimentDataset(encodings, labels)
    return dataset

def fine_tune_model(model_name, data_path, max_length=128, batch_size=32, epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    flag, train_dataset = get_or_preprocess_data(data_path, SentimentDataset, "train")
    if not flag:
        train_dataset = preprocess_data(f"data/raw/{data_path}/train.csv", tokenizer, max_length, f"data/processed/{data_path}/train.pt")
        
    flag, val_dataset = get_or_preprocess_data(data_path, SentimentDataset, "valid")
    if not flag:
        val_dataset = preprocess_data(f"data/raw/{data_path}/valid.csv", tokenizer, max_length, f"data/processed/{data_path}/valid.pt")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    output_dir = f'./results/{data_path}/{model_name}'

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
    save_model(model, tokenizer, f'./models/sentiment_analysis/{model_name}')
    
    # Cleanup checkpoints
    shutil.rmtree(output_dir)

    return model, tokenizer

def test_sentiment_analysis(model_name, data_path, max_length=128, batch_size=32):
    model_dir = f'./models/sentiment_analysis/{model_name}'
    if os.path.exists(model_dir):
        model, tokenizer = load_model(model_name, model_dir)
    else:
        model, tokenizer = fine_tune_model(model_name, data_path, max_length, batch_size)

    flag, test_dataset = get_or_preprocess_data(data_path, SentimentDataset, "test")
    if not flag:
        test_dataset = preprocess_data(f"data/raw/{data_path}/test.csv", tokenizer, max_length, f"data/processed/{data_path}/test.pt")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

