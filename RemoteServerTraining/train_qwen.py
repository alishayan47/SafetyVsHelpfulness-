"""
Qwen2.5-7B Safety Fine-tuning — Wolpertinger Server Edition
3x NVIDIA GeForce RTX 3090 (24576MiB each)

Run:
    python train_qwen.py                   # single-GPU (GPU 0)
    accelerate launch --multi_gpu --num_processes=3 train_qwen.py   # 3-GPU DDP
"""

import os
import json
import torch
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠  langdetect not installed — skipping language filter. Run: pip install langdetect")
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace login (set HF_TOKEN env var before running, or paste token below)
# ──────────────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # export HF_TOKEN=hf_...
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("✓ HuggingFace login successful")
else:
    print("⚠  HF_TOKEN not set — gated models (Llama, Gemma) will fail. Qwen is fine.")

# ──────────────────────────────────────────────────────────────────────────────
# Verify GPU
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nCUDA available: {torch.cuda.is_available()}")
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}  {props.total_memory / 1024**3:.1f} GB")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
OUTPUT_DIR = os.path.join(HOME, "qwen2.5-7b-safety-en")

MODEL_NAME = "Qwen/Qwen2.5-7B"
MAX_SEQ_LENGTH = 512

# LoRA
LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05

# Training — 3x RTX 3090, full bf16 LoRA, 3-GPU DDP
# Effective batch = BATCH_SIZE * GRAD_ACCUMULATION * num_gpus = 4 * 2 * 3 = 24
NUM_EPOCHS        = 6
BATCH_SIZE        = 4     # per-device; bf16 LoRA ~14GB base + activations, 24GB fits 4
GRAD_ACCUMULATION = 2     # effective batch = 4 * 2 * 3 GPUs = 24
LEARNING_RATE     = 8e-5

print(f"\n{'='*60}")
print(f"Model:            {MODEL_NAME}")
print(f"Output:           {OUTPUT_DIR}")
print(f"Epochs:           {NUM_EPOCHS}")
print(f"Batch/device:     {BATCH_SIZE}")
print(f"Grad accumulate:  {GRAD_ACCUMULATION}")
print(f"Effective batch:  {BATCH_SIZE} × {GRAD_ACCUMULATION} × 3 GPUs = {BATCH_SIZE * GRAD_ACCUMULATION * 3}")
print(f"LR:               {LEARNING_RATE}")
print(f"{'='*60}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset — load directly from HuggingFace Hub
# ──────────────────────────────────────────────────────────────────────────────
DATASET_HF = "gcuwajidali/safety-mix-3k-merged"
LOCAL_JSONL = os.path.join(HOME, "safety_mix_3k_merged.jsonl")

if os.path.exists(LOCAL_JSONL):
    print(f"Loading dataset from local file: {LOCAL_JSONL}")
    with open(LOCAL_JSONL, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f if line.strip()]
else:
    print(f"Loading dataset from HuggingFace: {DATASET_HF}")
    try:
        hf_dataset = load_dataset(DATASET_HF, split="train")
        raw_data = list(hf_dataset)
    except Exception as e:
        raise RuntimeError(
            f"Could not load dataset from HuggingFace ({e}).\n"
            f"Either upload {LOCAL_JSONL} to the server, or set HF_TOKEN."
        )

print(f"✓ Loaded {len(raw_data)} examples")

# Filter to English-only examples so the model learns to respond in English
if LANGDETECT_AVAILABLE:
    filtered = []
    for sample in raw_data:
        text = sample.get("output", "") or sample.get("response", "")
        try:
            if detect(text) == "en":
                filtered.append(sample)
        except LangDetectException:
            pass  # skip samples where language can't be detected
    print(f"✓ English filter: kept {len(filtered)}/{len(raw_data)} examples")
    raw_data = filtered
else:
    print("  Skipping language filter (langdetect not available)")

print(f"  First example: {json.dumps(raw_data[0], indent=2)[:200]}...\n")

# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
)
tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

# ──────────────────────────────────────────────────────────────────────────────
# Format dataset using model's native chat template
# ──────────────────────────────────────────────────────────────────────────────
def format_instruction(sample):
    instruction = sample.get("instruction", "")
    output      = sample.get("output", "")
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system",    "content": "You are a helpful, harmless, and honest AI assistant. Always respond in English only."},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": output},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        return (
            f"### System:\nYou are a helpful, harmless, and honest AI assistant. Always respond in English only.\n\n"
            f"### User:\n{instruction}\n\n"
            f"### Assistant:\n{output}"
        )

formatted_data = [format_instruction(s) for s in raw_data]
dataset = Dataset.from_dict({"text": formatted_data})
print(f"✓ Formatted {len(dataset)} examples")

# ──────────────────────────────────────────────────────────────────────────────
# Model — full bfloat16 LoRA (no quantization)
# RTX 3090 has 24GB; Qwen2.5-7B in bf16 = ~14GB — fits comfortably.
# Faster than QLoRA (no dequantization overhead) and cleaner 3-GPU DDP.
# For multi-GPU DDP via `accelerate launch`, device_map must NOT be "auto".
# ──────────────────────────────────────────────────────────────────────────────
USE_4BIT = False

model_kwargs = dict(torch_dtype=torch.bfloat16)

# device_map="auto" for single-GPU / pipeline parallel.
# For DDP (accelerate launch) use device_map=None (accelerate handles placement).
from accelerate import PartialState
using_ddp = PartialState().num_processes > 1
device_map = None if using_ddp else "auto"

print(f"\nLoading model {MODEL_NAME}  (4-bit={USE_4BIT}, DDP={using_ddp})...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device_map,
    trust_remote_code=True,
    **model_kwargs,
)
print("✓ Model loaded")

model.enable_input_require_grads()  # required for gradient checkpointing with LoRA
print("✓ Model prepared")

# ──────────────────────────────────────────────────────────────────────────────
# LoRA
# ──────────────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ──────────────────────────────────────────────────────────────────────────────
# Tokenize + split
# ──────────────────────────────────────────────────────────────────────────────
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
split = tokenized.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]
print(f"\nTrain: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ──────────────────────────────────────────────────────────────────────────────
# Training arguments
# ──────────────────────────────────────────────────────────────────────────────
NUM_GPUS        = PartialState().num_processes  # 3 for DDP, 1 for single-GPU
TOTAL_STEPS     = int((len(train_dataset) / (BATCH_SIZE * GRAD_ACCUMULATION * NUM_GPUS)) * NUM_EPOCHS)
WARMUP_STEPS    = int(TOTAL_STEPS * 0.08)
STEPS_PER_EPOCH = int(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUMULATION * NUM_GPUS))
EVAL_INTERVAL   = max(1, STEPS_PER_EPOCH // 2)

print(f"Total steps: {TOTAL_STEPS}  |  Warmup: {WARMUP_STEPS}  |  Eval every: {EVAL_INTERVAL}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=False,
    bf16=True,
    optim="adamw_torch_fused",
    max_grad_norm=0.3,
    weight_decay=0.001,
    eval_strategy="steps",
    eval_steps=EVAL_INTERVAL,
    save_strategy="steps",
    save_steps=EVAL_INTERVAL,
    save_total_limit=12,  # keep all checkpoints (6 epochs × 2 per epoch = 12)
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    logging_steps=10,
    report_to="none",
    seed=42,
)

# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

print("\nStarting training...")
print("=" * 60)
trainer.train()
print("=" * 60)
print("✓ Training complete!")

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nSaving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✓ Model saved!")

import subprocess
result = subprocess.run(["ls", "-lh", OUTPUT_DIR], capture_output=True, text=True)
print(result.stdout)
