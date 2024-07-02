import argparse
import os
import asyncio
import multiprocessing as mp
from data import (
    prepare_sentiment_data, prepare_headline_data, prepare_ner_data, prepare_unit_data,
    prepare_relation_cls_data, prepare_multiclass_cls_data, prepare_esg_cls_data, prepare_causal_cls_data,
    prepare_stock_movement_data, prepare_credit_scoring_data, prepare_fraud_detection_data,
    prepare_financial_distress_data, prepare_claim_analysis_data, prepare_numeric_labeling_data, prepare_qa_data
)
from benchmarks import (
    test_sentiment_analysis, test_headline_classification, test_ner, test_unit_classification,
    test_relation_classification, test_multiclass_classification, test_esg_classification, test_causal_classification,
    test_stock_movement_prediction, test_credit_scoring, test_fraud_detection, test_financial_distress_identification,
    test_claim_analysis, test_numeric_labeling, test_qa
)
from utils.file_utils import ensure_directory_exists

TASKS = {
    'sentiment_analysis': (prepare_sentiment_data, test_sentiment_analysis),
    'headline_classification': (prepare_headline_data, test_headline_classification),
    'ner': (prepare_ner_data, test_ner),
    'unit_classification': (prepare_unit_data, test_unit_classification),
    'relation_classification': (prepare_relation_cls_data, test_relation_classification),
    'multiclass_classification': (prepare_multiclass_cls_data, test_multiclass_classification),
    'esg_classification': (prepare_esg_cls_data, test_esg_classification),
    'causal_classification': (prepare_causal_cls_data, test_causal_classification),
    'stock_movement_prediction': (prepare_stock_movement_data, test_stock_movement_prediction),
    'credit_scoring': (prepare_credit_scoring_data, test_credit_scoring),
    'fraud_detection': (prepare_fraud_detection_data, test_fraud_detection),
    'financial_distress_identification': (prepare_financial_distress_data, test_financial_distress_identification),
    'claim_analysis': (prepare_claim_analysis_data, test_claim_analysis),
    'numeric_labeling': (prepare_numeric_labeling_data, test_numeric_labeling),
    'question_answering': (prepare_qa_data, test_qa),
}

async def async_prepare_task(task_name, prepare_func):
    ensure_directory_exists(f"data/raw/{task_name}")
    ensure_directory_exists(f"data/processed/{task_name}")

    # Prepare the data
    print(f"Preparing {task_name} data...")
    await asyncio.to_thread(prepare_func, task_name)

def run_benchmark_task(task_name, test_func, model_name, max_length, batch_size):
    # Run the benchmark
    print(f"Running {task_name} benchmark...")
    test_func(model_name=model_name, data_path=task_name, max_length=max_length, batch_size=batch_size)
    print(f"Benchmark for {task_name} completed.")

async def main(args):
    tasks = args.task.split(',')
    
    # Run preparation tasks asynchronously
    prepare_tasks = [
        async_prepare_task(task.strip(), TASKS[task.strip()][0]) for task in tasks
    ]
    await asyncio.gather(*prepare_tasks)

    # Run benchmarking tasks in parallel using multiprocessing
    processes = []
    for task in tasks:
        task_name = task.strip()
        test_func = TASKS[task_name][1]
        p = mp.Process(target=run_benchmark_task, args=(task_name, test_func, args.model_name, args.max_length, args.batch_size))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Comma-separated list of tasks to run benchmarks on: sentiment_analysis, headline_classification, ner, etc.")
    parser.add_argument("--model_name", required=True, type=str, help="Name of the model")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    asyncio.run(main(args))
