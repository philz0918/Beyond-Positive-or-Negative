import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import notebook_login

# =========================
# Configuration
# =========================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TRAIN_FILE = "filepath"
VALID_FILE = "filepath"
OUTPUT_DIR = "filepath"
MAX_SEQ_LENGTH = 256

# =========================
# Load dataset
# =========================
data_files = {"train": TRAIN_FILE}
if VALID_FILE is not None:
    data_files["validation"] = VALID_FILE

dataset = load_dataset("json", data_files=data_files)

print(dataset)
print(dataset["train"][0])

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# 4-bit config
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
# LoRA config
# =========================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# =========================
# SFT config
# =========================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,   # safer for Colab
    gradient_accumulation_steps=8,   # keep effective batch size
    learning_rate=2e-4,
    logging_steps=20,
    save_steps=200,
    eval_steps=200 if "validation" in dataset else None,
    eval_strategy="steps" if "validation" in dataset else "no",
    save_strategy="steps",
    bf16=use_bf16,
    fp16=not use_bf16,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    report_to="none",
    remove_unused_columns=False,
    max_length=MAX_SEQ_LENGTH,
    packing=False,
    assistant_only_loss=False,  # safer first run in Colab
    model_init_kwargs={
        "quantization_config": bnb_config,
        "device_map": "auto",
    },
)

# =========================
# Trainer
# =========================
trainer = SFTTrainer(
    model=MODEL_NAME,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"] if "validation" in dataset else None,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training complete. Model saved to: {OUTPUT_DIR}")