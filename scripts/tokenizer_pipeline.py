# scripts/tokenizer_pipeline.py

import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
import os

# Step 1: Load CUAD Dataset 
cuad_path = "cuad_dataset.csv"  # Replace with actual path if needed
df = pd.read_csv(cuad_path)

# Step 2: Select relevant clause text
if 'text' not in df.columns or 'filename' not in df.columns:
    raise ValueError("Dataset must contain 'filename' and 'text' columns")

df = df[['filename', 'text']].dropna()
df['text'] = df['text'].astype(str)

# Step 3: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Step 4: Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

# Step 5: Convert to Hugging Face Dataset and tokenize
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 6: Save to disk for downstream model training
output_path = "tokenized_legal_training_data"
os.makedirs(output_path, exist_ok=True)
tokenized_dataset.save_to_disk(output_path)

print(f"Tokenization complete: {len(tokenized_dataset)} clauses saved to '{output_path}'")
