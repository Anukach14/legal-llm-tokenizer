#batch_processor.py

import os
import argparse
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def tokenize_sentences(text):
    sentences = text.split(". ")
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def process_file(file_path):
    text = extract_text_from_txt(file_path)
    sentences = tokenize_sentences(text)
    return [{"filename": os.path.basename(file_path), "text": s} for s in sentences]

def batch_process_folder(folder_path):
    all_data = []
    print(f"\n📂 Processing files in: {folder_path}")
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            examples = process_file(file_path)
            all_data.extend(examples)
    return Dataset.from_list(all_data)

def main(input_folder):
    dataset = batch_process_folder(input_folder)
    print(f"\n🧠 Tokenizing {len(dataset)} clauses...")
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, max_length=512),
        batched=True
    )
    os.makedirs("tokenized_data", exist_ok=True)
    tokenized_dataset.save_to_disk("tokenized_data")
    print(f"\n✅ Tokenization complete. Data saved to 'tokenized_data/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch tokenize raw legal documents")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to folder containing .txt files")
    args = parser.parse_args()
    main(args.input_folder)
