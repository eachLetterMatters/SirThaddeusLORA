import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "../models/bielik/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype= torch.float16 if device == "cuda" else torch.float32
).to(device)


def ask_bielik():
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
    #     {"role": "user", "content": "Jak się nazywasz?"},     # Jak się nazywam? Jestem parobkiem, nazywam się Jakub.
    # ]

    messages = [
        {"role": "system", "content": "Odpowiadaj jak polski szlachcic"},
        {"role": "user", "content": "Jak działa internet?"},    # opis jakiegoś bloga
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    print("Model generating ...")

    outputs = model.generate(
        inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

print(ask_bielik())
