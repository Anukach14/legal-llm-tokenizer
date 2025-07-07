# train.py

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from datasets import load_from_disk
import os

# Load tokenized dataset from disk
dataset = load_from_disk("tokenized_data")

# Load tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set up training configuration
training_args = TrainingArguments(
    output_dir="legal-llm-finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,  # Set to False if not using GPU
    logging_dir="logs",
    save_total_limit=2,
    evaluation_strategy="no",  # You can set to "steps" or "epoch" if needed
    report_to="none"
)

# Initialize Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save final model
trainer.save_model("legal-llm-finetuned")
print("\n✅ Model training complete and saved to 'legal-llm-finetuned'")
