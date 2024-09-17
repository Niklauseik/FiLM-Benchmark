# FiLM-Benchmark

FiLM-Benchmark is a comprehensive benchmarking pipeline designed to evaluate the performance of language models on financial data. It focuses on various financial tasks such as credit scoring, sentiment analysis, and more. The system is modular and easily extensible, making it straightforward to add new tasks and models. The goal of this project is to provide a standardized and automated benchmarking process that can serve both researchers and practitioners in the financial domain.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Preparing Data](#preparing-data)
  - [Running Benchmarks](#running-benchmarks)
  - [Evaluating Models](#evaluating-models)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Comprehensive Benchmarking**: Supports over 10 tasks including sentiment analysis, credit scoring, and unit classification.
- **Modular Design**: Easily add new tasks by creating specific data preparation and testing scripts.
- **Automated Process**: The pipeline prepares data, fine-tunes models, runs benchmarks, and evaluates performance automatically.
- **Metrics and Reporting**: Outputs performance metrics like F1 score, accuracy, and more.
- **Parallel Execution**: Supports parallel processing to speed up benchmarking.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Preparing Data

Data preparation involves fetching and preprocessing the raw data. The pipeline uses specific data preparation scripts for each task.

```bash
python run_benchmark.py --task sentiment_analysis --model_name distilbert-base-uncased
```

### Running Benchmarks

Run the benchmarking process for a specific task and model using the following command:

```bash
python run_benchmark.py --task sentiment_analysis --model_name distilbert-base-uncased
```

This will prepare the data, fine-tune the model, and run the benchmark.

### Evaluating Models

The evaluation results will be printed to the console and saved in the `results` directory. Metrics include accuracy, precision, recall, and F1 score.

## Configuration

The pipeline can be configured through command-line arguments:

- `--task`: Specifies the task to run (e.g., sentiment_analysis, credit_scoring).
- `--model_name`: Specifies the name of the model to use (e.g., distilbert-base-uncased).
- `--max_length`: Maximum sequence length for the tokenizer.
- `--batch_size`: Batch size for training and evaluation.

Example:

```bash
python run_benchmark.py --task sentiment_analysis --model_name distilbert-base-uncased --max_length 128 --batch_size 32 --epochs 3
```
