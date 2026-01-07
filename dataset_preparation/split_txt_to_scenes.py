import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# code responsible for making the llm detect changing of a scene

# ==============================================================
#           currently model does not respond to prompt
# ==============================================================

print(torch.__version__)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_FILE = "pan-tadeusz.txt"
OUTPUT_FILE = "scenes.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

MAX_TOKENS = 1024
MAX_NEW_TOKENS = 90

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()

# prompt

SYSTEM_PROMPT = (
    "Jesteś narzędziem do analizy struktury narracyjnej. "
    "Nie streszczasz, nie interpretujesz i nie tworzysz treści. "
    "Odpowiadasz wyłącznie w JSON."
)

def build_user_prompt(prev, curr):
    return f"""
Czy pomiędzy Fragmentem A i Fragmentem B nastąpiła ISTOTNA ZMIANA narracyjna?

Odpowiedz TRUE, jeśli występuje JAKAKOLWIEK z poniższych:
- zmiana miejsca
- skok czasowy (świt, noc, nazajutrz, wspomnienie)
- zmiana głównej postaci lub punktu widzenia
- przejście opis → akcja lub dialog
- rozpoczęcie nowego zdarzenia

Fragment A:
\"\"\"
{prev}
\"\"\"

Fragment B:
\"\"\"
{curr}
\"\"\"

Odpowiedz WYŁĄCZNIE w JSON:
{{ "change_detected": true/false }}
"""

def clean_text(text):
    # normalizacja wielu nowych linii
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_into_fragments(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def is_meta_fragment(text):
    # nagłówki, streszczenia, tytuły
    return len(text.split()) < 20

def llm_detect_transition(prev, curr):
    # pomijamy metatekst
    if is_meta_fragment(prev) or is_meta_fragment(curr):
        return False

    user_prompt = build_user_prompt(prev, curr)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        add_generation_prompt=True
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    print("===== RAW LLM OUTPUT =====")
    print(decoded)
    print("==========================")

    # bierzemy OSTATNI poprawny JSON
    matches = re.findall(r"\{.*?\}", decoded, re.DOTALL)
    for m in reversed(matches):
        try:
            data = json.loads(m)
            return bool(data.get("change_detected", False))
        except json.JSONDecodeError:
            continue

    return False

# main part

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = clean_text(f.read())

fragments = split_into_fragments(text)
print("Liczba fragmentów:", len(fragments))

out = open(OUTPUT_FILE, "w", encoding="utf-8")
scene_id = 0

current_scene = fragments[0]

for i in range(1, len(fragments)):
    print(f"\n--- Fragment {i} ---")

    prev = fragments[i - 1]
    curr = fragments[i]

    start_new = llm_detect_transition(prev, curr)

    if start_new:
        out.write(json.dumps({
            "scene_id": scene_id,
            "text": current_scene.strip()
        }, ensure_ascii=False) + "\n")
        out.flush()

        scene_id += 1
        current_scene = curr
    else:
        current_scene += "\n\n" + curr

# ostatnia scena
if current_scene.strip():
    out.write(json.dumps({
        "scene_id": scene_id,
        "text": current_scene.strip()
    }, ensure_ascii=False) + "\n")
    out.flush()

out.close()

print(f"Zapisano {scene_id + 1} scen do {OUTPUT_FILE}")
