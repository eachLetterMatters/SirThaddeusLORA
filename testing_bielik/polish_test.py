import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ścieżka do folderu z lokalnymi plikami modelu
model_path = "../models/bielik/"  # np. "./Bielik-1.5B-v3.0"

# Wykrywanie urządzenia
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

# Wczytanie tokenizera
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Wczytanie modelu
if device == "cuda":
    # FP16 dla GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to(device)
else:
    # float32 dla CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32
    ).to(device)


# Funkcja do generowania odpowiedzi
def ask_bielik(prompt: str, max_new_tokens=128, temperature=0.7, top_p=0.9):
    # Tokenizacja
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # messages = [
    #     {"role": "system", "content": "Na każde pytanie odpowiadaj 'Nie wiem'"},
    #     {"role": "user", "content": "Kim jesteś?"},   # Nie wiem
    # ]

    # messages = [
    #     {"role": "system", "content": "Na każde pytanie odpowiadaj 'Nie wiem'"},
    #     {"role": "user", "content": "Kiedy wybuchła II wojna światowa?"},   # empty
    # ]

    # messages = [
    #     {"role": "system", "content": "Odpowiadaj jak siedemnastowieczny chłop"},
    #     {"role": "user", "content": "Jak się nazywasz?"}, # Jak się nazywam? Jestem parobkiem, nazywam się Jakub.
    # ]

    messages = [
        {"role": "system", "content": "Odpowiadaj jak siedemnastowieczny chłop"},
        {"role": "user", "content": "Jak działa internet?"}, # Jak się nazywam? Jestem parobkiem, nazywam się Jakub.
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # Generowanie
    outputs = model.generate(
        # **inputs,
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id
    )

    # Dekodowanie (pomijamy input)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply


# Przykładowy prompt
prompt = "Opowiedz po polsku o drugiej wojnie światowej"
reply = ask_bielik(prompt)
print("Prompt:", prompt)
print("Bielik odpowiada:", reply)
