# Legal Tokenizer & Dataset Builder for Corporate LLMs

This project demonstrates how to build a scalable pipeline that ingests legal case files—such as business contracts, clauses, and company-to-company rulings—and converts them into tokenized datasets ready for training a company-owned Large Language Model (LLM).

---

## 🔍 Project Objective

To help train a proprietary legal LLM by:
- Preprocessing and cleaning real-world corporate legal documents
- Splitting documents into semantically meaningful chunks
- Generating tokens using a Hugging Face-compatible tokenizer
- Saving the outputs in a format ready for pretraining or fine-tuning

---

## 💼 Use Case

A law firm or AI company handling corporate legal data (e.g. contracts, NDAs, arbitration cases) wants to train an LLM that understands business legal language. This project provides the first critical step: high-quality, tokenized input data.

---

## 🔭 Future Scope

- Add support for PDFs and scanned documents (OCR integration)
- Integrate clause-type classification using NLP
- Automate pretraining/fine-tuning jobs on cloud infrastructure
- Add RAG (retrieval-augmented generation) support for legal assistant apps

---

## 📚 Supported Dataset

This project supports the [CUAD dataset](https://github.com/TheAtticusProject/cuad), which contains clauses from real-world business contracts.
- Format: CSV with `filename` and `text` columns
- Domain: Business law, corporate agreements, NDAs, indemnity, etc.

You can also use any `.txt` legal documents placed in the `data/sample_cases/` folder.

---

## 📁 Project Structure

```
legal-llm-tokenizer/
├── data/
│   └── sample_cases/             # Raw legal files (.txt, .csv)
├── notebooks/
│   └── token_stats.ipynb         # Token length and clause stats
├── scripts/
│   ├── tokenizer_pipeline.py     # CUAD or CSV-based tokenization
│   └── batch_processor.py        # Folder-wise document processing
├── tokenized_data/               # Output directory (ignored by Git)
├── README.md                     # Project overview
├── requirements.txt              # Python dependencies
└── .gitignore                    # Ignore logs, outputs, and cache
```

---

## ⚙️ How It Works

**Text Preprocessing**: Documents are split by sentence or clause.  
**Tokenization**: Each chunk is tokenized using the GPT-2 tokenizer (or customizable).  
**Dataset Output**: Hugging Face-style dataset with `input_ids`, `attention_mask`, and metadata.

---

## 🛠️ How to Run

### Install Requirements
```bash
pip install -r requirements.txt
```

### Run the Pipeline (CSV or CUAD data)
```bash
python scripts/tokenizer_pipeline.py
```

### Process a Folder of .txt Legal Files
```bash
python scripts/batch_processor.py --input_folder data/sample_cases
```

---

## 📊 Output Format

Each example in the tokenized dataset includes:

```json
{
  "filename": "contract_abc.txt",
  "text": "This agreement shall terminate if...",
  "input_ids": [464, 1532, 215, ...],
  "attention_mask": [1, 1, 1, ...]
}
```

---

## 🔁 Integration with LLM Training

This dataset can be directly used in:
- Pretraining (next-token prediction with `input_ids`)
- Fine-tuning (instruction/prompt-based models)
- Evaluation of model performance in the legal domain

---

## 📈 Analysis Tools

Use the `notebooks/token_stats.ipynb` notebook to:
- Visualize average token length per clause
- Plot document token distributions
- Preview sample inputs and token sequences

---

## 🧠 Skills Demonstrated

- Natural Language Processing (NLP)
- Tokenization and sequence preparation for LLMs
- Legal document preprocessing and clause extraction
- Data pipeline design and automation
- Dataset formatting for Hugging Face Trainer
- Scalable processing of raw `.txt` legal case files

---

## ✅ Why This Project Matters

- Shows real-world understanding of document-level tokenization
- Demonstrates readiness to work with legal datasets
- Scales to tens of thousands of case files
- Aligns directly with tasks required by legal AI companies

---

## 👤 Author

*Generative AI Engineer Portfolio Project – Designed to show readiness for tokenization, dataset generation, and pretraining workflows in legal AI environments.*
