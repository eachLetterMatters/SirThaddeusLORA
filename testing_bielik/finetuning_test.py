import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

MODEL_PATH = "../models/bielik/"
DATASET_PATH = "../dataset_preparation/sarmata_2.jsonl"
OUTPUT_DIR = "../models/bielik_sarmata_lora_2"
MAX_LENGTH = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Używam urządzenia:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype = torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=False
)

model.to(device)

lora_config = LoraConfig(
    r=8,                                        # lora matrixes size
    lora_alpha=16,                              # stronger influece if alpha > r
    lora_dropout=0.05,                          # prevent overfitting
    target_modules=["q_proj", "v_proj"],        # query and value projections
    bias="none",                                # keeps model`s biases the same
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files=DATASET_PATH)["train"]

def tokenize(example):
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# effective batch = per_device_train_batch_size * gradient_accumulation_steps * num_devices
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=2e-4,
    fp16= device == "cuda",
    bf16=False,         # FP16 = higher precision, smaller numeric range, BF16 = lower precision, larger numeric range
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    optim="adamw_torch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Zakończono trening i zapisano adapter LoRA")
