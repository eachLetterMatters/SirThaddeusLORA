import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# KONFIGURACJA
# =========================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_FILE = "pan-tadeusz.txt"
OUTPUT_FILE = "scenes.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

MAX_TOKENS = 1024
MAX_NEW_TOKENS = 80

# =========================
# MODEL
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()

# =========================
# PROMPT
# =========================

SYSTEM_PROMPT = """Jesteś narzędziem do analizy struktury narracyjnej.
Nie streszczasz, nie interpretujesz i nie tworzysz treści.
Twoim zadaniem jest wykrywanie granic scen.
"""

USER_PROMPT = """
Oceń przejście narracyjne pomiędzy dwoma fragmentami tekstu.

ZACZNIJ NOWĄ SCENĘ, jeśli występuje przynajmniej JEDNO z poniższych:
- wyraźna zmiana miejsca
- skok czasowy (świt, noc, nazajutrz, wspomnienie)
- zmiana głównej postaci lub fokalizacji
- przejście opis → akcja lub dialog
- nowe zdarzenie narracyjne

Fragment A:
\"\"\"
{prev}
\"\"\"

Fragment B:
\"\"\"
{curr}
\"\"\"

Odpowiedz WYŁĄCZNIE w JSON:

{{ "start_new_scene": true/false }}
"""

# =========================
# POMOCNICZE
# =========================

def clean_text(text):
    # clean places with many \n
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_into_fragments(text):
    # podział po akapitach (mechaniczny, deterministyczny)
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 0]

def llm_detect_transition(prev, curr):
    prompt = SYSTEM_PROMPT + USER_PROMPT.format(prev=prev, curr=curr)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    match = re.search(r"\{.*?\}", decoded, re.DOTALL)
    if not match:
        return False

    try:
        data = json.loads(match.group())
        return bool(data.get("start_new_scene", False))
    except json.JSONDecodeError:
        return False

# =========================
# PIPELINE
# =========================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = clean_text(f.read())

fragments = split_into_fragments(text)

print(len(fragments))

# output jsonl file with scenes
out = open(OUTPUT_FILE, "w", encoding="utf-8")
scene_id = 0

# scenes = []
current_scene = fragments[0]

for i in range(1, len(fragments)):
    print("Current fragment: " + str(i))
    prev = fragments[i - 1]
    curr = fragments[i]

    start_new = llm_detect_transition(prev, curr)

    if start_new:
        # save current scene into json file
        out.write(json.dumps({
            "scene_id": scene_id,
            "text": current_scene.strip()
        }, ensure_ascii=False) + "\n")
        out.flush()
        scene_id += 1
        current_scene = curr
    else:
        # append curr to current scene
        current_scene += "\n\n" + curr

# ostatnia scena
if current_scene.strip():
    out.write(json.dumps({
        "scene_id": scene_id,
        "text": current_scene.strip()
    }, ensure_ascii=False) + "\n")
    out.flush()

out.close()