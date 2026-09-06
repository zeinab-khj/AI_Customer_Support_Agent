
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "./outputs/qwen_response_05b"


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
# Generate Response
# ============================================================

def generate_response(customer_message: str) -> str:

    prompt = f"""Customer: {customer_message}
Response:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    customer_message = (
        "I need help cancelling my purchase {{Order Number}}."
    )

    response = generate_response(customer_message)

    print("Customer:")
    print(customer_message)

    print("\nAssistant:")
    print(response)

