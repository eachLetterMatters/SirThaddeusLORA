
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import pyttsx3
from pyttsx3 import speak

# --- TTS setup ---
engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

BASE_MODEL = "../models/bielik/"
ADAPTER_DIR = "../models/bielik_sarmata_lora_2"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Wczytanie adaptera LoRA
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()


def generate_response(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Wyciągnięcie samej odpowiedzi asystenta
    if "assistant" in decoded:
        decoded = decoded.split("assistant")[-1].strip()

    return decoded


print("Rozmowa rozpoczęta. Wpisz 'exit' aby zakończyć.\n")

while True:
    user_input = input("Ty: ").strip()
    if user_input.lower() in ["exit", "quit", "wyjdz", "koniec"]:
        print("Koniec rozmowy.")
        break

    messages = [{"role": "system", "content": "Odpowiadaj jako siedemnastowieczny szlachcic Rzeczypospolitej"},
                {"role": "user", "content": user_input}]

    print("Generuję odpowiedź...")
    response = generate_response(messages)
    print("\nSzlachcic:", response, "\n")
    # TTS
    if response.strip():
        speak(response)
