
import json
import os

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "./outputs/qwen_response_05b"
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

OUTPUT_DIR = "./outputs/evaluation"

MAX_NEW_TOKENS = 400
NUM_SAMPLES = 600
SEED = 42


# ============================================================
# Load Model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
).to("cuda")

model.eval()


# ============================================================
# Load Evaluation Data
# ============================================================

dataset = load_dataset(DATASET_NAME)["train"]

eval_dataset = dataset.train_test_split(
    test_size=0.1,
    seed=SEED,
)["test"]

eval_dataset = eval_dataset.select(
    range(min(NUM_SAMPLES, len(eval_dataset)))
)


# ============================================================
# Generate Responses
# ============================================================

predictions = []
references = []

for example in tqdm(eval_dataset):

    customer_message = example["instruction"]
    reference = example["response"]

    prompt = f"""Customer: {customer_message}
Response:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    prediction = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    predictions.append(prediction)
    references.append(reference)


# ============================================================
# ROUGE
# ============================================================

rouge = evaluate.load("rouge")

rouge_results = rouge.compute(
    predictions=predictions,
    references=references,
)


# ============================================================
# BERTScore
# ============================================================

bertscore = evaluate.load("bertscore")

bert_results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en",
)


bert_score_results = {
    "precision": sum(bert_results["precision"])
    / len(bert_results["precision"]),

    "recall": sum(bert_results["recall"])
    / len(bert_results["recall"]),

    "f1": sum(bert_results["f1"])
    / len(bert_results["f1"]),
}


# ============================================================
# Combine Results
# ============================================================

results = {
    "num_samples": len(predictions),

    "rouge": {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "rougeLsum": rouge_results["rougeLsum"],
    },

    "bertscore": bert_score_results,
}


# ============================================================
# Save Results
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

results_path = os.path.join(
    OUTPUT_DIR,
    "metrics.json",
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)


# ============================================================
# Print Results
# ============================================================

print("\nEvaluation Results")
print("=" * 50)

print(f"Samples: {results['num_samples']}")

print("\nROUGE:")
for metric, value in results["rouge"].items():
    print(f"{metric}: {value:.4f}")

print("\nBERTScore:")
for metric, value in results["bertscore"].items():
    print(f"{metric}: {value:.4f}")

print(f"\nResults saved to: {results_path}")

