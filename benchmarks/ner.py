import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import shutil
from torch.utils.data import DataLoader

# Ensure the utils module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ner import compute_metrics

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Define the mapping from labels to integers
label_mapping = {
    "O": 0,
    "B-ORG": 1,
    "I-ORG": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-PER": 5,
    "I-PER": 6,
}

def convert_labels_to_ids(labels, label_mapping):
    """Convert the labels to their integer representations."""
    return [[label_mapping[label] for label in label_sequence] for label_sequence in labels]

# Convert predictions and labels back to their string representations
def convert_ids_to_labels(ids, label_mapping):
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    return [inverse_label_mapping[id] for id in ids]

# Custom data collator to handle padding within batches
def custom_data_collator(features):
    batch = {k: [d[k] for d in features] for k in features[0]}
    max_length = max(len(x) for x in batch['input_ids'])
    
    def pad_sequence(seq, max_length, pad_value):
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        return seq + [pad_value] * (max_length - len(seq))
    
    for key in batch:
        if key == 'labels':
            batch[key] = [pad_sequence(x, max_length, -100) for x in batch[key]]
        else:
            batch[key] = [pad_sequence(x, max_length, 0) for x in batch[key]]
    
    batch = {k: torch.tensor(v) for k, v in batch.items()}
    return batch

def load_data(file_path):
    """Load data from CSV files."""
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].apply(eval).tolist()  # Assuming labels are stored as strings of lists in the CSV
    return texts, labels

def tokenize_and_align_labels(examples, tokenizer, max_length=128):
    """Tokenize the input text and align the labels manually."""
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for text, labels in zip(examples['text'], examples['labels']):
        words = text.split()
        token_ids = []
        label_ids = []
        for word, label in zip(words, labels):
            word_tokens = tokenizer.tokenize(word, add_special_tokens=False)
            word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)
            label_ids.extend([label] * len(word_token_ids))

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            label_ids = label_ids[:max_length]
        else:
            token_ids.extend([tokenizer.pad_token_id] * (max_length - len(token_ids)))
            label_ids.extend([-100] * (max_length - len(label_ids)))

        tokenized_inputs["input_ids"].append(token_ids)
        tokenized_inputs["attention_mask"].append([1] * len(token_ids))
        tokenized_inputs["labels"].append(label_ids)

    return tokenized_inputs

def preprocess_data(file_path, tokenizer, max_length, save_path=None):
    texts, labels = load_data(file_path)
    labels = convert_labels_to_ids(labels, label_mapping)
    encodings = tokenize_and_align_labels({'text': texts, 'labels': labels}, tokenizer, max_length)
    if save_path:
        torch.save((encodings, labels), save_path)
    dataset = Dataset.from_dict(encodings)
    return dataset

def load_preprocessed_data(load_path):
    encodings, labels = torch.load(load_path)
    dataset = Dataset.from_dict(encodings)
    return dataset

def fine_tune_model(model_name, data_path, max_length=128, batch_size=32, epochs=2):
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

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_mapping))
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
        learning_rate=2e-5,
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
        data_collator=custom_data_collator
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    model.save_pretrained(f'./models/ner/{model_name}')
    tokenizer.save_pretrained(f'./models/ner/{model_name}')
    
    # Cleanup checkpoints
    shutil.rmtree(output_dir)

    return model, tokenizer

def test_ner(model_name, data_path, max_length=128, batch_size=32):
    model_dir = f'./models/ner/{model_name}'
    if os.path.exists(model_dir):
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        model, tokenizer = fine_tune_model(model_name, data_path, max_length, batch_size)

    test_data_path = f"data/processed/{data_path}/test.pt"
    if os.path.exists(test_data_path):
        test_dataset = load_preprocessed_data(test_data_path)
    else:
        test_dataset = preprocess_data(f"data/raw/{data_path}/test.csv", tokenizer, max_length, test_data_path)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_data_collator)

    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=2)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    true_labels = [[label for label in labels if label != -100] for labels in all_labels]
    pred_labels = [[label for label in prediction if label != -100] for prediction in all_preds]

    true_labels_str = [convert_ids_to_labels(labels, label_mapping) for labels in true_labels]
    pred_labels_str = [convert_ids_to_labels(labels, label_mapping) for labels in pred_labels]

    metrics = compute_metrics(true_labels_str, pred_labels_str)
    
    print(f"Entity Precision: {metrics['precision']:.4f}")
    print(f"Entity Recall: {metrics['recall']:.4f}")
    print(f"Entity F1: {metrics['f1']:.4f}")
