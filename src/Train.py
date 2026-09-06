
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

OUTPUT_DIR = "./outputs/qwen_response_05b"

MAX_LENGTH = 512

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16


# ============================================================
# Load Dataset
# ============================================================

dataset = load_dataset(DATASET_NAME)

dataset = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]


# ============================================================
# Load Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Tokenization
# ============================================================

def tokenize_function(example):
    prompt = f"""Customer: {example["instruction"]}
Response:"""

    response = example["response"]

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=True,
    )

    response_tokens = tokenizer(
        response,
        add_special_tokens=False,
    )

    input_ids = (
        prompt_tokens["input_ids"]
        + response_tokens["input_ids"]
    )

    attention_mask = [1] * len(input_ids)

    labels = (
        [-100] * len(prompt_tokens["input_ids"])
        + response_tokens["input_ids"]
    )

    # Keep sequences within the model input limit
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    remove_columns=eval_dataset.column_names,
)


# ============================================================
# Load Model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# Data Collator
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# ============================================================
# Training Arguments
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    fp16=False,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_strategy="steps",
    logging_steps=100,

    save_total_limit=2,

    report_to="none",
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    tokenizer=tokenizer,
    data_collator=data_collator,
)


# ============================================================
# Train
# ============================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Model saved to: {OUTPUT_DIR}")

