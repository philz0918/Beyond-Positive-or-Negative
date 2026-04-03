import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# =========================
# Config
# =========================
MODEL_DIR = "modelpath"
TEST_FILE = "filepath"
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

NO_SYMPTOM_ID = label2id["NO_SYMPTOM"]


# =========================
# Metrics helpers
# =========================
def compute_metrics_all(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def compute_metrics_excluding_no_symptom(y_true, y_pred, no_symptom_id):
    """
    Remove rows whose TRUE label is NO_SYMPTOM, then evaluate.
    This answers: how well does the model perform on symptom-containing cases?
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != no_symptom_id

    if mask.sum() == 0:
        return {
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "num_examples": 0,
        }

    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    acc = accuracy_score(y_true_filtered, y_pred_filtered)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_filtered,
        y_pred_filtered,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "num_examples": int(mask.sum()),
    }


# =========================
# Load saved model + tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# =========================
# Load test dataset
# Expected format:
# {"sentence":"...", "label":"FATIGUE"}
# =========================
test_dataset = load_dataset("json", data_files={"test": TEST_FILE})

def encode_labels(example):
    sentence = str(example["sentence"]).strip()
    label_name = str(example["label"]).strip()

    if label_name not in label2id:
        raise ValueError(f"Unknown label: {label_name}")

    return {
        "sentence": sentence,
        "label": label2id[label_name],
    }

test_dataset = test_dataset.map(encode_labels)


# =========================
# Tokenize
# =========================
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_test = test_dataset.map(tokenize_function, batched=True)


# =========================
# Build Trainer for prediction only
# =========================
test_args = TrainingArguments(
    output_dir="./tmp_test_output",
    per_device_eval_batch_size=16,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=test_args,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# =========================
# Predict
# =========================
pred_output = trainer.predict(tokenized_test["test"])

logits = pred_output.predictions
y_true = pred_output.label_ids
y_pred = np.argmax(logits, axis=-1)

# Convert to label strings
true_labels = [id2label[int(x)] for x in y_true]
pred_labels = [id2label[int(x)] for x in y_pred]

# =========================
# 1) Evaluation for ALL labels
# =========================
metrics_all = compute_metrics_all(y_true, y_pred)

print("\n=== Evaluation: ALL labels ===")
for k, v in metrics_all.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\nClassification report (ALL labels):")
print(classification_report(true_labels, pred_labels, digits=4, zero_division=0))


# =========================
# 2) Evaluation EXCEPT NO_SYMPTOM
# =========================
metrics_excl = compute_metrics_excluding_no_symptom(
    y_true, y_pred, NO_SYMPTOM_ID
)

print("\n=== Evaluation: EXCLUDING NO_SYMPTOM ===")
for k, v in metrics_excl.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

mask = np.array(y_true) != NO_SYMPTOM_ID
true_labels_excl = [id2label[int(x)] for x in np.array(y_true)[mask]]
pred_labels_excl = [id2label[int(x)] for x in np.array(y_pred)[mask]]

print("\nClassification report (EXCLUDING NO_SYMPTOM):")
print(classification_report(true_labels_excl, pred_labels_excl, digits=4, zero_division=0))


# =========================
# Save detailed prediction results
# =========================
results_df = pd.DataFrame({
    "sentence": test_dataset["test"]["sentence"],
    "true_label": true_labels,
    "pred_label": pred_labels,
    "correct": [t == p for t, p in zip(true_labels, pred_labels)],
})

results_df.to_csv("./model/test_predictions_detailed.csv", index=False)
print("\nSaved detailed predictions to test_predictions_detailed.csv")