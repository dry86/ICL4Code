import jsonlines
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_path, device):
    """Load the model and tokenizer from the specified path."""

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        pad_token_id=tokenizer.eos_token_id 
    ).to(device)
    return tokenizer, model

def generate_response(model, tokenizer, prompt, max_tokens=256):
    """Generate a response using AutoModelForCausalLM model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            # temperature=0,
            top_p=1.0,
            do_sample=False,
            pad_token_id = tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

def process_file(file_path, output_path, model, tokenizer):
    """Process the input JSONL file and generate outputs using the model."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    results = []
    fail = []

    with jsonlines.open(file_path) as reader:
        queries = [obj for obj in reader]

    for idx, query in enumerate(tqdm(queries, desc="Processing queries")):
        if idx > 100:
            break

        prompt = query.get("prompt", "")
        label = query.get("label", "")
        try:
            generated_text = generate_response(model, tokenizer, prompt)
            result = {
                "idx": idx,
                "label": label,
                "generated_text": generated_text
            }
            results.append(result)

            with jsonlines.open(output_path, mode="a") as writer:
                writer.write(result)

        except Exception as e:
            print(f"Error processing query {idx}: {e}")
            fail.append({"idx": idx, "error": str(e)})

    print(f"Processing completed. {len(fail)} queries failed.")

if __name__ == "__main__":
    # Directly specify inputs
    input_file = "/newdisk/public/wws/ICL4code/prompts/summary.jsonl"
    model_path = "/newdisk/public/wws/model_dir/Qwen2.5-Coder/Qwen2.5-Coder-7B"
    gpu_id = 1

    # Set device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Output file name
    output_file = os.path.splitext(input_file)[0] + "_results.jsonl"

    # Load model and tokenizer
    tokenizer, model = load_model(model_path, device)

    # Process input file
    print("Processing file...")
    process_file(input_file, output_file, model, tokenizer)

    print(f"Results saved to {output_file}")
