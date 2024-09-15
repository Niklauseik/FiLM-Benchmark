import os
import pandas as pd
from transformers import AutoTokenizer

# Define the base path of your datasets
BASE_PATH = 'data/raw'

# Define the tasks and their respective directories
TASKS = [
    'sentiment_analysis',
    'causal_classification',
    'claim_analysis',
    'credit_scoring',
    'esg_classification',
    'financial_distress_identification',
    'flare_finqa',
    'fraud_detection',
    'headline_classification',
    'multiclass_classification',
    'ner',
    'numeric_labeling',
    'question_answering',
    'relation_classification',
    'relation_extraction',
    'stock_movement_prediction',
    'unit_classification'
]

# Define the possible splits
SPLITS = ['train', 'valid', 'validation', 'test']

# Initialize the tokenizer (you can choose any tokenizer)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
MAX_LENGTH = 512  # Set the maximum length for the tokenizer

# Function to count tokens in a text with truncation
def count_tokens(text):
    tokens = tokenizer.tokenize(text, max_length=MAX_LENGTH, truncation=True)
    return len(tokens)

# DataFrame to collect all token counts
all_token_counts = pd.DataFrame(columns=['task', 'input_tokens', 'output_tokens'])

# Possible column names for input and output
input_columns = ['query', 'text','token','question']
output_columns = ['answer', 'label','gold']

# Iterate through each task and each file
for task in TASKS:
    task_input_tokens = 0
    task_output_tokens = 0
    for split in SPLITS:
        file_path = os.path.join(BASE_PATH, task, f'{split}.csv')
        if os.path.exists(file_path):
            print(f'Processing {file_path}...')
            df = pd.read_csv(file_path)
            
            # Determine the actual column names
            input_col = next((col for col in input_columns if col in df.columns), None)
            output_col = next((col for col in output_columns if col in df.columns), None)
            
            if input_col is None or output_col is None:
                print(f"Skipping {file_path} due to missing required columns.")
                continue
            
            # Calculate tokens for input and output
            df['input_tokens'] = df[input_col].astype(str).apply(count_tokens)
            df['output_tokens'] = df[output_col].astype(str).apply(count_tokens)
            
            # Summarize the token counts for this task
            task_input_tokens += df['input_tokens'].sum()
            task_output_tokens += df['output_tokens'].sum()
        else:
            print(f'{file_path} does not exist. Skipping...')
    
    # Append the summary for the task to the main DataFrame
    task_summary = pd.DataFrame({
        'task': [task],
        'input_tokens': [task_input_tokens],
        'output_tokens': [task_output_tokens]
    })
    
    all_token_counts = pd.concat([all_token_counts, task_summary], ignore_index=True)

# Save the results to an Excel file
output_file = 'token_counts_summary.xlsx'
all_token_counts.to_excel(output_file, index=False)

print(f"Token counting complete. Results saved to {output_file}.")
