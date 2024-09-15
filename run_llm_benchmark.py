import argparse
from tasks import TASKS
from utils.file_utils import load_and_combine_datasets

MODEL_URL_MAP = {
    'chatGPT': 'https://api-inference.huggingface.co/models/openai/chatgpt',
    'llama': 'https://api-inference.huggingface.co/models/huggyllama/llama-7b',
    # Add other model URLs here
}

API_TOKEN = ""  # Specify your Hugging Face API token here

def run_benchmark_task(task_name, test_func, model_url):
    # Load and combine datasets
    combined_data = load_and_combine_datasets(task_name)

    # Run the benchmark
    print(f"Running {task_name} benchmark...")
    test_func(model_url=model_url, combined_data=combined_data, api_token=API_TOKEN)
    print(f"Benchmark for {task_name} completed.")

def main(args):
    tasks = args.task.split(',')

    # Get the model URL from the map
    model_url = MODEL_URL_MAP.get(args.model_key)
    print(f"Model URL: {model_url}")
    if not model_url:
        raise ValueError(f"Model key {args.model_key} is not found in the URL map.")

    for task in tasks:
        task_name = task.strip()
        test_func = TASKS[task_name]
        run_benchmark_task(task_name, test_func, model_url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Comma-separated list of tasks to run benchmarks on: sentiment_analysis, headline_classification, ner, etc.")
    parser.add_argument("--model_key", required=True, type=str, help="Key for the model URL: chatGPT, llama, etc.")
    args = parser.parse_args()
    main(args)
