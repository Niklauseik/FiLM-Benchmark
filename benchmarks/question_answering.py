import os
import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score
import shutil

# Ensure the utils module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.file_utils import save_preprocessed_data, save_model, load_model, get_or_preprocess_data

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class QADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['start_positions'] = torch.tensor(self.labels['start_positions'][idx]).to(device)
        item['end_positions'] = torch.tensor(self.labels['end_positions'][idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels['start_positions'])

def preprocess_data(file_path, tokenizer, max_length, save_path=None):
    df = pd.read_csv(file_path)
    encodings = tokenizer(
        list(df['query']), 
        list(df['context']),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    labels = {'start_positions': [], 'end_positions': []}
    for i in range(len(df)):
        context = df.iloc[i]['context']
        answer = df.iloc[i]['answer']
        start_idx = context.find(answer)
        end_idx = start_idx + len(answer)
        
        labels['start_positions'].append(start_idx)
        labels['end_positions'].append(end_idx)
    
    if save_path:
        save_preprocessed_data(encodings, labels, save_path)
    
    dataset = QADataset(encodings, labels)
    return dataset

def fine_tune_model(model_name, data_path, max_length=128, batch_size=32, epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    flag, train_dataset = get_or_preprocess_data(data_path, QADataset, "train")
    if not flag:
        train_dataset = preprocess_data(f"data/raw/{data_path}/train.csv", tokenizer, max_length, f"data/processed/{data_path}/train.pt")
        
    flag, val_dataset = get_or_preprocess_data(data_path, QADataset, "valid")
    if not flag:
        val_dataset = preprocess_data(f"data/raw/{data_path}/valid.csv", tokenizer, max_length, f"data/processed/{data_path}/valid.pt")

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(device)

    output_dir = f'./results/{model_name}'

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
    
    save_model(model, tokenizer, f'./models/qa/{model_name}')
    
    # Cleanup checkpoints
    shutil.rmtree(output_dir)

    return model, tokenizer

def test_qa(model_name, data_path, max_length=128, batch_size=32):
    print(f"Using device: {device}")

    model_dir = f'./models/qa/{model_name}'
    if os.path.exists(model_dir):
        model, tokenizer = load_model(model_name, model_dir)
    else:
        model, tokenizer = fine_tune_model(model_name, data_path, max_length, batch_size)

    flag, test_dataset = get_or_preprocess_data(data_path, QADataset, "test")
    if not flag:
        test_dataset = preprocess_data(f"data/raw/{data_path}/test.csv", tokenizer, max_length, f"data/processed/{data_path}/test.pt")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            start_scores, end_scores = outputs.start_logits, outputs.end_logits
            for i in range(len(start_scores)):
                start_idx = torch.argmax(start_scores[i])
                end_idx = torch.argmax(end_scores[i])
                answer = tokenizer.decode(inputs['input_ids'][i][start_idx:end_idx+1])
                all_preds.append(answer)

    df = pd.read_csv(f"data/raw/{data_path}/test.csv")
    df['predicted_answers'] = all_preds
    df.to_csv(f"data/raw/{data_path}/predicted_test.csv", index=False)

    # Evaluate the predictions (assuming exact match evaluation)
    correct = 0
    for pred, true in zip(df['predicted_answers'], df['answer']):
        if pred.strip().lower() == true.strip().lower():
            correct += 1
    
    total = len(df)
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")
