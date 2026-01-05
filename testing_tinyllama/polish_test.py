import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

prompt_1 = """<|user|>
Opowiedz mi o sobie.
</s>
<|assistant|>
"""
# replies in english

prompt_2 = """<|user|>
Opowiedz mi o sobie. Odpowiedz po polsku
</s>
<|assistant|>
"""

inputs = tokenizer(prompt_1, return_tensors="pt").to("cpu")
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))