import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MultiLabelBinarizer

# Ensure the utils module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.numeric_utils import label_to_id, id_to_label

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class NumericLabelingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_data(file_path, tokenizer, max_length):
    df = pd.read_csv(file_path)
    texts = df['token'].apply(eval).tolist()  # Assuming tokens are stored as strings of lists in the CSV
    labels = df['label'].apply(eval).tolist()  # Assuming labels are stored as strings of lists in the CSV
    labels = [[label_to_id[label] for label in label_sequence] for label_sequence in labels]

    tokenized_inputs = tokenizer(
        texts,
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Ensure labels are the same length as the input_ids
    encoded_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        encoded_labels.append(label_ids)
    
    tokenized_inputs['labels'] = torch.tensor(encoded_labels)
    
    return NumericLabelingDataset(tokenized_inputs, encoded_labels)

# Custom data collator to handle padding within batches
def custom_data_collator(features):
    batch = {k: [d[k] for d in features] for k in features[0]}
    max_length = max(len(x) for x in batch['input_ids'])
    
    def pad_sequence(seq, max_length, pad_value):
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        return seq + [pad_value] * (max_length - len(seq))
    
    padded_batch = {}
    for key in batch:
        pad_value = -100 if key == 'labels' else 0
        padded_batch[key] = torch.tensor([pad_sequence(x, max_length, pad_value) for x in batch[key]], dtype=torch.long if key != 'labels' else torch.float)
    
    return padded_batch

def fine_tune_model(model_name, data_path, max_length=128, batch_size=32, epochs=10):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = preprocess_data(f"data/raw/{data_path}/train.csv", tokenizer, max_length)
    val_dataset = preprocess_data(f"data/raw/{data_path}/valid.csv", tokenizer, max_length)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_to_id))
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
        data_collator=custom_data_collator  # Use custom collator here
    )

    trainer.train()

    

    return model, tokenizer

def test_numeric_labeling(model_name, data_path, max_length=128, batch_size=32):
    model, tokenizer = fine_tune_model(model_name, data_path, max_length, batch_size)

    test_dataset = preprocess_data(f"data/raw/{data_path}/test.csv", tokenizer, max_length)
    
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

    true_labels_str = [[id_to_label[label] for label in labels] for labels in true_labels]
    pred_labels_str = [[id_to_label[label] for label in labels] for labels in pred_labels]

    mlb = MultiLabelBinarizer()
    true_labels_bin = mlb.fit_transform(true_labels_str)
    pred_labels_bin = mlb.transform(pred_labels_str)

    # Using precision_recall_fscore_support for aggregate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_bin, pred_labels_bin, average='macro', zero_division=0)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

# Running the test
if __name__ == "__main__":
    model_name = "distilbert-base-uncased"
    data_path = "numeric_labeling"
    test_numeric_labeling(model_name, data_path)
