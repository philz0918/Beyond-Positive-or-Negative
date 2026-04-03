import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =========================
# Config
# =========================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_DIR = "filepath"   # your saved adapter folder
TEST_FILE = "filepath"
OUTPUT_CSV = "filepath"

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

label_set = set(LABELS)
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}
NO_SYMPTOM_ID = label2id["NO_SYMPTOM"]

# =========================
# Quantization for inference
# =========================
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

# =========================
# Load tokenizer + base model + adapter
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# =========================
# Helpers
# =========================
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def split_prompt_and_gold(example):
    """
    Input format:
    {"messages":[
        {"role":"system","content":"..."},
        {"role":"user","content":"Sentence: ..."},
        {"role":"assistant","content":"FATIGUE"}
    ]}
    """
    prompt_messages = []
    gold_label = None

    for msg in example["messages"]:
        if msg["role"] == "assistant":
            gold_label = str(msg["content"]).strip()
        else:
            prompt_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    return prompt_messages, gold_label

def normalize_prediction(text):
    text = text.strip()

    if text in label_set:
        return text

    first_line = text.splitlines()[0].strip()
    if first_line in label_set:
        return first_line

    for label in LABELS:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label

    return "INVALID_PREDICTION"

@torch.no_grad()
def predict_one(prompt_messages, max_new_tokens=8):
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pred_label = normalize_prediction(generated_text)
    return pred_label, generated_text

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

def compute_metrics_excluding_no_symptom(y_true, y_pred):
    y_true = torch.tensor(y_true).cpu().numpy()
    y_pred = torch.tensor(y_pred).cpu().numpy()

    mask = y_true != NO_SYMPTOM_ID
    if mask.sum() == 0:
        return {
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "num_examples": 0,
        }

    yt = y_true[mask]
    yp = y_pred[mask]

    acc = accuracy_score(yt, yp)
    precision, recall, f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "num_examples": int(mask.sum()),
    }

# =========================
# Run prediction
# =========================
examples = load_jsonl(TEST_FILE)

records = []
true_ids = []
pred_ids = []

for ex in tqdm(examples):
    prompt_messages, gold_label = split_prompt_and_gold(ex)
    pred_label, raw_output = predict_one(prompt_messages)

    valid_gold = gold_label in label_set
    valid_pred = pred_label in label_set

    records.append({
        "gold_label": gold_label,
        "pred_label": pred_label,
        "raw_output": raw_output,
        "correct": (gold_label == pred_label) if (valid_gold and valid_pred) else False,
        "user_text": next((m["content"] for m in prompt_messages if m["role"] == "user"), "")
    })

    if valid_gold and valid_pred:
        true_ids.append(label2id[gold_label])
        pred_ids.append(label2id[pred_label])

results_df = pd.DataFrame(records)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}")

# =========================
# Evaluation
# =========================
if len(true_ids) > 0:
    metrics_all = compute_metrics_all(true_ids, pred_ids)
    print("\n=== ALL LABELS ===")
    for k, v in metrics_all.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    true_labels = [id2label[i] for i in true_ids]
    pred_labels = [id2label[i] for i in pred_ids]

    print("\nClassification report (ALL):")
    print(classification_report(true_labels, pred_labels, digits=4, zero_division=0))

    metrics_excl = compute_metrics_excluding_no_symptom(true_ids, pred_ids)
    print("\n=== EXCLUDING NO_SYMPTOM ===")
    for k, v in metrics_excl.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    true_arr = torch.tensor(true_ids).cpu().numpy()
    pred_arr = torch.tensor(pred_ids).cpu().numpy()
    mask = true_arr != NO_SYMPTOM_ID

    true_excl = [id2label[int(x)] for x in true_arr[mask]]
    pred_excl = [id2label[int(x)] for x in pred_arr[mask]]

    print("\nClassification report (EXCLUDING NO_SYMPTOM):")
    print(classification_report(true_excl, pred_excl, digits=4, zero_division=0))
else:
    print("No valid gold/pred pairs found.")