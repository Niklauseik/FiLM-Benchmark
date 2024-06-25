import sys
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
import numpy as np
import pandas as pd

# Ensure the utils module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ner import compute_metrics

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

# Convert the labels to their integer representations
def convert_labels_to_ids(labels, label_mapping):
    return [[label_mapping[label] for label in label_sequence] for label_sequence in labels]

# Load data from CSV files
def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].apply(eval).tolist()  # Assuming labels are stored as strings of lists in the CSV
    return texts, labels

# Tokenize the input text and align the labels manually
def tokenize_and_align_labels(examples, tokenizer, max_length=128):
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

# Custom data collator to handle padding within batches
def custom_data_collator(features, tokenizer):
    max_length = max(len(feature["input_ids"]) for feature in features)
    batch = {k: [] for k in features[0].keys()}
    for feature in features:
        for k, v in feature.items():
            if k == "input_ids" or k == "attention_mask":
                batch[k].append(v + [tokenizer.pad_token_id] * (max_length - len(v)))
            elif k == "labels":
                batch[k].append(v + [-100] * (max_length - len(v)))
            else:
                batch[k].append(v)
    batch = {k: torch.tensor(v) for k, v in batch.items()}
    return batch

# Convert predictions and labels back to their string representations
def convert_ids_to_labels(ids, label_mapping):
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    return [inverse_label_mapping[id] for id in ids]

# Function to run the NER task
def run_ner(model_name):
    # Define your dataset using the real data
    train_texts, train_labels = load_data('data/raw/ner/train.csv')
    test_texts, test_labels = load_data('data/raw/ner/test.csv')
    valid_texts, valid_labels = load_data('data/raw/ner/valid.csv')

    # Convert the labels to their integer representations
    train_labels = convert_labels_to_ids(train_labels, label_mapping)
    test_labels = convert_labels_to_ids(test_labels, label_mapping)
    valid_labels = convert_labels_to_ids(valid_labels, label_mapping)

    # Create the Dataset objects
    train_data = {'text': train_texts, 'labels': train_labels}
    test_data = {'text': test_texts, 'labels': test_labels}
    valid_data = {'text': valid_texts, 'labels': valid_labels}

    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)
    valid_dataset = Dataset.from_dict(valid_data)
    datasets = DatasetDict({"train": train_dataset, "test": test_dataset, "valid": valid_dataset})

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_mapping))

    # Tokenize the datasets
    tokenized_datasets = datasets.map(lambda x: tokenize_and_align_labels(x, tokenizer, max_length=128), batched=True, remove_columns=["text", "labels"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=lambda features: custom_data_collator(features, tokenizer)
    )

    # Train the model
    trainer.train()

    # Get predictions
    outputs = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(outputs.predictions, axis=2)

    true_labels = [[label for label in labels if label != -100] for labels in outputs.label_ids]
    pred_labels = [[label for label in prediction if label != -100] for prediction in predictions]

    true_labels_str = [convert_ids_to_labels(labels, label_mapping) for labels in true_labels]
    pred_labels_str = [convert_ids_to_labels(labels, label_mapping) for labels in pred_labels]

    # Compute entity-level metrics
    metrics = compute_metrics(true_labels_str, pred_labels_str)

    print(f"Entity Precision: {metrics['precision']:.4f}")
    print(f"Entity Recall: {metrics['recall']:.4f}")
    print(f"Entity F1: {metrics['f1']:.4f}")

# Example usage:
run_ner("bert-base-cased")
