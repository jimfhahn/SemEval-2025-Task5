import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import ast
from dotenv import load_dotenv
from joblib import Parallel, delayed
import numpy as np

# Disable parallelism in tokenizers to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process.log"),
        logging.StreamHandler()
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

def load_models():
    token = 'your_huggingface_token'
    if not token:
        raise ValueError("HuggingFace token not found. Please set it as an environment variable.")
    
    model_names = [
        "jimfhahn/bert-base-german-cased-gnd",
        "jimfhahn/base-multilingual-gnd-bert",
        "jimfhahn/base-multilingual-gnd-bert-uncased",
        "jimfhahn/ModernBERT-base-dnb"
    ]
    
    models = []
    for model_name in model_names:
        try:
            logging.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=token).to(device)
            models.append((model, tokenizer))
            logging.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
    
    return models

def limit_text(text, word_limit=32):
    if pd.isna(text):
        return ""
    return ' '.join(text.split()[:word_limit])

def classify_text_batch(texts, models, top_n=50):
    results = []
    for text in texts:
        limited_text = limit_text(text)
        if not limited_text.strip():
            results.append([])
            continue

        text_results = []
        for model, tokenizer in models:
            inputs = tokenizer(limited_text, return_tensors="pt", truncation=True, padding=True, max_length=32).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            top_probs, top_class_ids = torch.topk(probs, top_n, dim=-1)
            for i in range(top_n):
                label = model.config.id2label[top_class_ids[0, i].item()]
                confidence = top_probs[0, i].item()
                text_results.append({"predicted_label": label, "confidence": confidence})
        results.append(text_results)
    return results

def filter_and_aggregate(classifications, desired_label_count=50):
    aggregated = {}
    for c in classifications:
        lbl = c.get("predicted_label")
        confidence = c.get("confidence", 0.0)
        if lbl:
            aggregated[lbl] = aggregated.get(lbl, 0.0) + confidence
    
    sorted_aggregated = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    aggregated = dict(sorted_aggregated[:desired_label_count])
    return aggregated

def get_top_50_subjects(aggregated):
    sorted_items = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    top_labels = [label for label, _ in sorted_items[:50]]
    return top_labels

def validate_results(csv_path):
    df = pd.read_csv(csv_path)
    
    def parse_subjects(subjects):
        if pd.isna(subjects):
            return []
        if isinstance(subjects, list):
            return subjects
        try:
            return ast.literal_eval(subjects)
        except (ValueError, SyntaxError):
            logging.warning(f"Failed to parse subjects: {subjects}")
            return []
    
    df['top_50_subjects'] = df['top_50_subjects'].apply(parse_subjects)
    df['subject_count'] = df['top_50_subjects'].apply(len)
    invalid_counts = df[df['subject_count'] != 50]
    if not invalid_counts.empty:
        logging.error(f"{len(invalid_counts)} entries do not have exactly 50 subjects.")
    else:
        logging.info("All entries have exactly 50 subjects.")

def main():
    try:
        logging.info("Loading data...")
        df_original = pd.read_csv('ten_more/output.tsv', sep='\t')

        logging.info("Loading models...")
        models = load_models()

        logging.info("Classifying title_abstract from df_original...")
        df_original_unique = df_original.drop_duplicates(subset=['mmsid', 'title_abstract'])
        df_original_unique['classification_results'] = Parallel(n_jobs=4)(delayed(classify_text_batch)(batch, models) for batch in np.array_split(df_original_unique['title_abstract'].tolist(), 20))
        logging.info("Completed classification of title_abstract.")
        df_original_unique.to_csv('brief_records/df_original_unique_classified.csv', index=False)
        logging.info("Saved classified title_abstract results to disk.")

        logging.info("Aggregating title_abstract results...")
        aggregated_original = df_original_unique.groupby('mmsid')['classification_results'].apply(lambda x: filter_and_aggregate([item for sublist in x for item in sublist]))
        logging.info("Completed aggregation of title_abstract results.")
        aggregated_original.to_csv('brief_records/aggregated_original.csv')
        logging.info("Saved aggregated title_abstract results to disk.")

        logging.info("Determining the top 50 GND labels per MMSID based on scores...")
        top_50_per_mmsid = aggregated_original.apply(get_top_50_subjects).reset_index()
        top_50_per_mmsid.rename(columns={0: 'top_50_subjects'}, inplace=True)
        
        logging.info(f"Computed top 50 subjects for {len(top_50_per_mmsid)} MMSIDs.")
        
        # Save the results
        output_csv = 'brief_records/test-data-jan30-2025_top_50_subjects_per_mmsid_by_score.csv'
        top_50_per_mmsid.to_csv(output_csv, index=False)
        
        logging.info(f"Saved top 50 subjects per MMSID based on scores to {output_csv} successfully.")

        validate_results(output_csv)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()