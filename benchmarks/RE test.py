import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, pipeline
from transformers import DataCollatorWithPadding
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Relation to ID mapping
relation_to_id = {
    'legal_form': 0,
    'publisher': 1,
    'owner_of': 2,
    'employer': 3,
    'manufacturer': 4,
    'position_held': 5,
    'chairperson': 6,
    'industry': 7,
    'business_division': 8,
    'creator': 9,
    'original_broadcaster': 10,
    'chief_executive_officer': 11,
    'location_of_formation': 12,
    'operator': 13,
    'owned_by': 14,
    'founded_by': 15,
    'parent_organization': 16,
    'member_of': 17,
    'product_or_material_produced': 18,
    'brand': 19,
    'headquarters_location': 20,
    'director_/_manager': 21,
    'distribution_format': 22,
    'distributed_by': 23,
    'platform': 24,
    'currency': 25,
    'subsidiary': 26,
    'stock_exchange': 27,
    'developer': 28
}

# Load datasets
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

train_texts, train_labels = load_dataset("data/raw/relation_extraction/train.csv")
valid_texts, valid_labels = load_dataset("data/raw/relation_extraction/valid.csv")
test_texts, test_labels = load_dataset("data/raw/relation_extraction/test.csv")

# Split the labels into head, tail, and relation
def parse_labels(labels):
    all_parsed_labels = []
    for label in labels:
        label = label.strip("[]").replace("'", "").replace("\"", "").strip()
        triplets = label.split(", ")
        parsed_triplets = [tuple(triplet.split(" ; ")) for triplet in triplets]
        all_parsed_labels.append(parsed_triplets)
    return all_parsed_labels

train_parsed_labels = parse_labels(train_labels)
valid_parsed_labels = parse_labels(valid_labels)
test_parsed_labels = parse_labels(test_labels)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

# Tokenize the texts
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

train_tokenized_inputs = tokenize_texts(train_texts)
valid_tokenized_inputs = tokenize_texts(valid_texts)
test_tokenized_inputs = tokenize_texts(test_texts)

# Load pre-trained NER model and tokenizer
ner_model = AutoModelForTokenClassification.from_pretrained("distilbert-base-cased")
ner_pipeline = pipeline('ner', model=ner_model, tokenizer=tokenizer)

# Function to detect entities
def detect_entities(text):
    ner_results = ner_pipeline(text)
    entities = []
    for entity in ner_results:
        entities.append((entity['word'], (entity['start'], entity['end'])))
    return entities

# Create datasets for training, validation, and testing
def create_relation_dataset(texts, parsed_labels, tokenizer, relation_to_id):
    inputs = []
    labels = []
    for text, triplets in zip(texts, parsed_labels):
        entities = detect_entities(text)
        for head, tail, relation in triplets:
            head_pos = next((pos for ent, pos in entities if ent == head), None)
            tail_pos = next((pos for ent, pos in entities if ent == tail), None)
            if head_pos and tail_pos:
                head_start, head_end = head_pos
                tail_start, tail_end = tail_pos
                context = text[max(0, head_start-50):min(len(text), tail_end+50)]
                encoded = tokenizer(context, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                inputs.append(encoded)
                labels.append(relation_to_id[relation])
    return inputs, labels

train_inputs, train_labels = create_relation_dataset(train_texts, train_parsed_labels, tokenizer, relation_to_id)
valid_inputs, valid_labels = create_relation_dataset(valid_texts, valid_parsed_labels, tokenizer, relation_to_id)
test_inputs, test_labels = create_relation_dataset(test_texts, test_parsed_labels, tokenizer, relation_to_id)

class RelationExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].squeeze() for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = RelationExtractionDataset(train_inputs, train_labels)
valid_dataset = RelationExtractionDataset(valid_inputs, valid_labels)
test_dataset = RelationExtractionDataset(test_inputs, test_labels)

# Initialize the model for relation extraction
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-cased", num_labels=len(relation_to_id))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Metrics calculation
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model on validation set
eval_results = trainer.evaluate(eval_dataset=valid_dataset)
print("Validation results:", eval_results)

# Evaluate the model on test set
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test results:", test_results)
