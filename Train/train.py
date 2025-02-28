import os
import sys
import numpy as np
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
from autotrain.params import TextClassificationParams
from autotrain.project import AutoTrainProject
import torch
import random

# Set the cache directory for Hugging Face and DeepSpeed Triton
os.environ['HF_HOME'] = '/scratch/jimhahn/transformers_cache'
os.environ['TRITON_CACHE_DIR'] = '/scratch/jimhahn/triton_cache'

# Verify numpy import path
print(f"Numpy is loaded from: {np.__file__}")

# Check current directory
print(f"Current working directory: {os.getcwd()}")

# Check sys.path for conflicts
print(f"sys.path: {sys.path}")

# Log in to Hugging Face
HF_USERNAME = "jimfhahn"
HF_TOKEN = "hf_token"

# Set the random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load and preprocess the dataset
data = load_dataset('jimfhahn/json', split='train')

def process_data(batch):
    batch['label'] = str(batch['label'])
    return batch

data = data.map(process_data)

# Check class distribution and filter out classes with fewer than 2 samples
label_counts = Counter(data['label'])
valid_labels = [label for label, count in label_counts.items() if count >= 2]

data = data.filter(lambda example: example['label'] in valid_labels)

# Convert to pandas DataFrame
data_df = data.to_pandas()

# Ensure all classes are represented in both training and validation sets
train_df, valid_df = train_test_split(data_df, test_size=0.2, stratify=data_df['label'], random_state=42)

# Ensure valid set has all classes
missing_classes = set(train_df['label'].unique()) - set(valid_df['label'].unique())
if missing_classes:
    for cls in missing_classes:
        sample = train_df[train_df['label'] == cls].sample(1)
        valid_df = pd.concat([valid_df, sample])
        train_df = train_df.drop(sample.index)

# Reset the index to avoid duplicate index columns
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

# Check for duplicate columns and remove them if necessary
train_df = train_df.loc[:, ~train_df.columns.duplicated()]
valid_df = valid_df.loc[:, ~valid_df.columns.duplicated()]

# Convert back to datasets.Dataset
train_data = Dataset.from_pandas(train_df)
valid_data = Dataset.from_pandas(valid_df)

# Ensure the jsonl directory exists
os.makedirs('./jsonl', exist_ok=True)

# Save to JSONL
train_data.to_json('./jsonl/train.jsonl', orient='records', lines=True)
valid_data.to_json('./jsonl/validation.jsonl', orient='records', lines=True)

# Define the parameters for the AutoTrain project
params = TextClassificationParams(
    model="google-bert/bert-base-multilingual-cased",
    data_path="./jsonl",
    text_column="text",
    train_split="train",
    valid_split="validation",
    target_column="label",
    epochs=20,
    batch_size=16,
    max_seq_length=512,
    lr=3e-5,
    optimizer="adamw_torch",
    scheduler="linear",
    gradient_accumulation=2,
    mixed_precision="fp16",
    project_name="base-multilingual-gnd-bert",
    log="tensorboard",
    push_to_hub=True,
    username="jimfhahn",
    token="hf_token",
)

# Create and train the project
project = AutoTrainProject(params=params, backend="local", process=True)
project.create()