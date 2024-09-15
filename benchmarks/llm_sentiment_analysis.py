import requests
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def query_huggingface(api_url, payload, api_token):
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

def get_predicted_label(result):
    print(f"Result: {result}")  # Debugging statement to inspect the result
    try:
        return max(result[0], key=lambda x: x['score'])['label']
    except (IndexError, KeyError) as e:
        print(f"Error processing result: {e}")
        return None

def llm_sentiment_analysis(model_url, combined_data, api_token):
    # Prepare lists to store results
    texts = combined_data['text'].tolist()
    true_labels = combined_data['answer'].tolist()  # Assuming 'answer' column contains the true labels
    predictions = []

    # Perform sentiment analysis
    for text in texts:
        result = query_huggingface(model_url, {"inputs": text}, api_token)
        predicted_label = get_predicted_label(result)
        if predicted_label:
            predictions.append(predicted_label)
        else:
            predictions.append("unknown")  # Handle the case where prediction fails

    # Map the model's output labels to your true labels
    label_mapping = {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive'
    }
    mapped_predictions = [label_mapping.get(pred, 'unknown') for pred in predictions]

    # Evaluate Predictions
    accuracy = accuracy_score(true_labels, mapped_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, mapped_predictions, average='macro')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
