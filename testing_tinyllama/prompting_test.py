import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# ===============================================================
# testing System prompts - they seem not to work on TinyLLama
# ===============================================================

prompt_wrong_format = """
<s>[INST] <<SYS>>
You are assistant, who talks like Yoda
<</SYS>>
What is the weather today [/INST]</s>
"""

prompt_correct_format= """
<|system|>
You are assistant, who talks like Yoda.</s>
<|user|>
What is the weather today?</s>
<|assistant|>
"""

messages = [
    {"role": "system", "content": "You must answer with exactly: 12345. Don't say anything else."},
    {"role": "user", "content": "What's the weather today?"}
]

prompt_correct_format_2 = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ===============================================================
# testing prompts without System part
# ===============================================================

prompt_1 = """<|user|>
Reply ONLY in Spanish.
Use bullet points.
What is the weather today?
</s>
<|assistant|>
"""
# replied in Spanish, without bullet points

prompt_2 = """<|user|>
You must answer with exactly: 12345.
Do not say anything else.
What is the weather today?
</s>
<|assistant|>
"""
# answered about the weather

prompt_3 = """
<|user|>
Answer with exactly the word: BLUE
</s>
<|assistant|>
"""
# replied: "My favorite color is blue!"

prompt_4 = """<|user|>
Use bullet points in your reply. Do not say anything else.
What is the weather today?
</s>
<|assistant|>
"""
# replied literally with a letter to "promptness"

prompt_5 = """<|user|>
Please use bullet points if possible.
What is the weather today?
</s>
<|assistant|>
"""
# nice answer in bullet points

prompt_6 = """
What is the weather today?
The correct answer is: BLUE
Repeat the correct answer.
"""
# workaround not working at all, model speaks nonsense

inputs = tokenizer(prompt_6, return_tensors="pt").to("cpu")
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# =====================================================
#                     conclusions
# =====================================================

"""
TinyLLaMA Chat is not a rule-following model in the strict sense
- Trained mostly on instruction → natural language answer
- Strongly biased toward helpful conversational completion
- So when you say: “Answer with exactly the word BLUE” it does not interpret 
this as a constraint. It interprets it as topic steering (“the answer involves blue”).
- This is called instruction softening and is very common in small models.

- TinyLLaMA follows INTENT, not RULES.
"""
