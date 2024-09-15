import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import shutil

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class HeadlineDataset(Dataset):
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
    encodings = tokenizer(list(df['query']), truncation=True, padding=True, max_length=max_length)
    labels = list(df['label'])
    dataset = HeadlineDataset(encodings, labels)
    if save_path:
        print(f"Saving preprocessed data to {save_path}")
        torch.save((encodings, labels), save_path)
    return dataset

def load_preprocessed_data(load_path):
    print(f"Loading preprocessed data from {load_path}")
    encodings, labels = torch.load(load_path)
    return HeadlineDataset(encodings, labels)

def fine_tune_model(model_name, data_path, max_length=128, batch_size=32, epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if preprocessed data exists
    train_data_path = f"data/processed/{data_path}/train.pt"
    val_data_path = f"data/processed/{data_path}/valid.pt"
    
    if os.path.exists(train_data_path) and os.path.exists(val_data_path):
        train_dataset = load_preprocessed_data(train_data_path)
        val_dataset = load_preprocessed_data(val_data_path)
    else:
        train_dataset = preprocess_data(f"data/raw/{data_path}/train.csv", tokenizer, max_length, train_data_path)
        val_dataset = preprocess_data(f"data/raw/{data_path}/valid.csv", tokenizer, max_length, val_data_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels as needed
    model.to(device)

    output_dir = f'./results/{data_path}/{model_name}'
    last_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
            last_checkpoint = os.path.join(output_dir, last_checkpoint)

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

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    model.save_pretrained(f'./models/{data_path}/{model_name}')
    tokenizer.save_pretrained(f'./models/{data_path}/{model_name}')
    
    # Cleanup checkpoints
    shutil.rmtree(output_dir)

    return model, tokenizer

def test_headline_classification(model_name, data_path, max_length=128, batch_size=32):
    model_dir = f'./models/{data_path}/{model_name}'
    if os.path.exists(model_dir):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        model, tokenizer = fine_tune_model(model_name, data_path, max_length, batch_size)

    test_data_path = f"data/processed/{data_path}/test.pt"
    if os.path.exists(test_data_path):
        test_dataset = load_preprocessed_data(test_data_path)
    else:
        test_dataset = preprocess_data(f"data/raw/{data_path}/test.csv", tokenizer, max_length, test_data_path)
    
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

