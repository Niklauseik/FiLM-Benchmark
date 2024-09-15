import pandas as pd
from datasets import Dataset
from transformers import pipeline

# Load your dataset
df = pd.read_csv('data/raw/sentiment_analysis/test.csv')  # Adjust the path to your dataset
dataset = Dataset.from_pandas(df)

# Load the models
llama_model = pipeline('sentiment-analysis', model='meta-llama/Llama-2-13b-chat-hf')
openai_model = pipeline('sentiment-analysis', model='openai-gpt')

# Function to get predictions
def get_predictions(model, dataset):
    predictions = []
    for example in dataset:
        pred = model(example['text'])
        predictions.append(pred[0]['label'])
    return predictions

# Get predictions
llama_predictions = get_predictions(llama_model, dataset)
openai_predictions = get_predictions(openai_model, dataset)

# Add predictions to the dataframe
df['llama_predictions'] = llama_predictions
df['openai_predictions'] = openai_predictions

# Evaluate performance (example for accuracy)
df['correct_llama'] = df['llama_predictions'] == df['answer']
df['correct_openai'] = df['openai_predictions'] == df['answer']

llama_accuracy = df['correct_llama'].mean()
openai_accuracy = df['correct_openai'].mean()

print(f"Llama Model Accuracy: {llama_accuracy}")
print(f"OpenAI Model Accuracy: {openai_accuracy}")

