import re
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# =========================
# Config
# =========================
MODEL_NAME = "roberta-base"
TRAIN_FILE = "filepath"
VALID_FILE = "filepath"   # set to None if you do not have validation
OUTPUT_DIR = "filepath"
MAX_LENGTH = 256

LABELS = [
    "ANHEDONIA",
    "APPETITE_CHANGE",
    "COGNITIVE_ISSUES",
    "DEPRESSED_MOOD",
    "FATIGUE",
    "NO_SYMPTOM",
    "PSYCHOMOTOR",
    "SLEEP_ISSUES",
    "SPECIAL_CASE",
    "SUICIDAL_THOUGHTS",
    "WORTHLESSNESS",
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}


# =========================
# Load dataset
# =========================
data_files = {"train": TRAIN_FILE}
if VALID_FILE is not None:
    data_files["validation"] = VALID_FILE

dataset = load_dataset("json", data_files=data_files)

# Convert string label -> integer label id
def encode_labels(example):
    return {
        "sentence": example["sentence"],
        "label": label2id[example["label"]]
    }

dataset = dataset.map(encode_labels)

print(dataset)
print(dataset["train"][0])

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# Model
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
)

# =========================
# Training args
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch" if VALID_FILE is not None else "no",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=20,
    load_best_model_at_end=True if VALID_FILE is not None else False,
    metric_for_best_model="macro_f1" if VALID_FILE is not None else None,
    greater_is_better=True if VALID_FILE is not None else None,
    report_to="none",
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"] if VALID_FILE is not None else None,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if VALID_FILE is not None else None,
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Save
# =========================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved model to {OUTPUT_DIR}")