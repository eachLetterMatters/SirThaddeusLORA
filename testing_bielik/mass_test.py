import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

INPUT_JSONL = "../dataset_preparation/test_prompts.jsonl"
OUTPUT_JSON = "../dataset_preparation/testing_outputs_2.json"

NUM_SAMPLES_PER_PROMPT = 3
MAX_NEW_TOKENS = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

# Lista modeli do przetestowania
MODELS = [
    {
        "name": "bielik_sarmata_2",
        "base_model": "../models/bielik/",
        "adapter": "../models/bielik_sarmata_lora_2"
    },
    # {
    #     "name": "tinyllama_sarmata_2",
    #     "base_model": "../models/tinyllama/",
    #     "adapter": "../models/tinyllama_sarmata_lora_2"
    # }
]


def extract_assistant_answer(text: str) -> str:
    if "assistant" in text:
        return text.split("assistant")[-1].strip()
    return text.strip()


def load_model_and_tokenizer(base_model_path, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_assistant_answer(decoded)


# wczytanie promptów
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    prompts = [json.loads(line) for line in f]

# struktura wyników: na start puste results dla każdego pytania
final_results = [
    {
        "pytanie": item.get("input", ""),
        "results": []
    }
    for item in prompts
]

# główna pętla po modelach
for model_cfg in MODELS:
    model_name = model_cfg["name"]
    base_model = model_cfg["base_model"]
    adapter = model_cfg["adapter"]

    print(f"\n=== Ładowanie modelu: {model_name} ===")

    model, tokenizer = load_model_and_tokenizer(base_model, adapter)

    for idx, item in enumerate(tqdm(prompts, desc=f"Generowanie ({model_name})")):
        instruction = item.get("instruction", "")
        user_input = item.get("input", "")

        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": user_input})

        answers = {}
        for i in range(NUM_SAMPLES_PER_PROMPT):
            response = generate_response(model, tokenizer, messages)
            answers[f"odpowiedz_{i+1}"] = response

        final_results[idx]["results"].append({
            "model": model_name,
            "odpowiedzi": answers
        })

    # sprzątanie VRAM-u przed kolejnym modelem
    del model
    torch.cuda.empty_cache()

# zapis do JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

print("Zakończono generowanie. Wyniki zapisane do:", OUTPUT_JSON)
