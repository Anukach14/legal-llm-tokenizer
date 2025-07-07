Legal Tokenizer & Dataset Builder for Corporate LLMs
This project demonstrates how to build a scalable pipeline that ingests legal case files such as business contracts, clauses, and company-to-company rulings, and converts them into tokenized datasets ready for training a company-owned Large Language Model (LLM).

🔍 Project Objective
To help train a proprietary legal LLM by:

Preprocessing and cleaning real-world corporate legal documents.

Splitting documents into semantically meaningful chunks.

Generating tokens using a HuggingFace-compatible tokenizer.

Saving the outputs in a format ready for pretraining or fine-tuning.

💼 Use Case
A law firm or AI company handling corporate legal data (e.g. contracts, NDAs, arbitration cases) wants to train an LLM that understands business legal language. This project provides the first critical step: high-quality, tokenized input data.

🔭 Future Scope
Add support for PDFs and scanned documents (OCR integration)

Integrate clause-type classification using NLP

Automate pretraining/fine-tuning jobs on cloud infrastructure

Add RAG (retrieval-augmented generation) support for legal assistant apps

📂 Project Structure
bash
Copy
Edit
legal-llm-tokenizer/
├── data/
│   └── sample_cases/               # Raw legal files (.txt, .csv)
├── notebooks/
│   └── token_stats.ipynb          # Token length and clause stats
├── scripts/
│   └── tokenizer_pipeline.py      # CUAD or CSV-based tokenization
│   └── batch_processor.py         # Folder-wise document processing
├── tokenized_data/                # Output directory
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies
└── .gitignore
🔧 How It Works
Raw Document Input: Legal contracts, clauses, or full cases in .txt or .csv format.

Text Preprocessing: Documents are split by sentence or clause.

Tokenization: Each chunk is tokenized using the GPT-2 tokenizer (or customizable).

Dataset Output: Hugging Face-style dataset with input_ids, attention_mask, and metadata.

🛠️ How to Run
Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
Run the Pipeline (CSV or CUAD data)
bash
Copy
Edit
python scripts/tokenizer_pipeline.py
Process a Folder of .txt Legal Files
bash
Copy
Edit
python scripts/batch_processor.py --input_folder data/sample_cases
📊 Output Format
Each example in the tokenized dataset includes:

json
Copy
Edit
{
  "filename": "contract_abc.txt",
  "text": "This agreement shall terminate if...",
  "input_ids": [464, 1532, 215, ...],
  "attention_mask": [1, 1, 1, ...]
}
🔁 Integration with LLM Training
This dataset can be directly used in:

Pretraining (next-token prediction with input_ids)

Fine-tuning (instruction/prompt-based models)

Evaluation of model performance in the legal domain

📈 Analysis Tools
Use the notebooks/token_stats.ipynb to:

Visualize average token length per clause

Plot document token distributions

Preview sample inputs and token sequences

🧠 Skills Demonstrated
Natural Language Processing (NLP)

Tokenization and sequence preparation for LLMs

Legal document preprocessing and clause extraction

Data pipeline design and automation

Dataset formatting for Hugging Face Trainer

Scalable processing of raw .txt legal case files

✅ Why This Project Matters
Shows real-world understanding of document-level tokenization

Demonstrates readiness to work with legal datasets

Scales to tens of thousands of case files

Aligns directly with tasks required by legal AI companies

👤 Author
Created by Anurag Kacherikar