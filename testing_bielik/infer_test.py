from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "../models/bielik/"
ADAPTER_DIR = "../models/bielik_sarmata_lora"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używam urządzenia: {device}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype= torch.float16 if device == "cuda" else torch.float32
).to(device)

# Wczytanie adaptera LoRA
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

model.eval()

def generate_response():
    messages = [
        # {"role": "system", "content": "Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim."},
        # {"role": "user", "content": "Kiedy wybuchła druga wojna światowa?"},

        {"role": "system", "content": "Odpowiadaj jak polski szlachcic"},
        {"role": "user", "content": "Kiedy wybuchła druga wojna światowa?"}
        # Waszmość, druga wojna światowa wybuchła w 1939 roku, po agresji Niemiec na Polskę.
        # Miłościwy Panie, nie jestem w stanie przewidzieć przyszłości, lecz gdy wybuchła, miała miejsce w 1939 roku.
        # Miłościwy Panie, nie wiem, czy szlachcicowi to pytanie wypada zadawać. W mojej wiedzy dotyczy to jedynie czasów współczesnych.


        # {"role": "system", "content": "Odpowiadaj jak polski szlachcic"},
        # {"role": "user", "content": "Czy wiesz co to jest internet?"}
        # Mistrz, nie znam tego terminu. Internet to zjawisko komputerowe, o którym nie słyszałem.
        # Miłościwy Panie, internet to sieć połączeń między komputerami, umożliwiająca przesyłanie i odbieranie danych w sposób globalny.
        # Mistrz, o takim czymś słyszałem, ale nie używałem.
        # Miłościwy Panie, internet to sieć połączeń komputerowych, umożliwiających komunikację na odległość.
        # Miłościwy Panie, internet to sieć komputerowa umożliwiająca wymianę informacji na dużą odległość. Nie każdy o tym wie, a dla wielu to coś zupełnie nowego.
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,             # Different output each time
        temperature=0.7,            # 0 - deterministic, > 1 - creative
        top_p=0.9                   # avoids low-probability garbage tokens
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = generate_response()
print(response)
