"""
model_api.py
------------
Local inference backend — runs on Wolpertinger (port 8000).
Loads one model at a time into VRAM; swaps on demand.

Start:
    python3 model_api.py

Endpoints:
    GET  /status          — which model is loaded, is it switching
    POST /switch          — { "model": "qwen" | "llama" }
    POST /generate        — { "model": "qwen", "message": "...", "history": [...] }
"""

from flask import Flask, request, jsonify
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch
import gc
import re
import os

app = Flask(__name__)

# ── Model config ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.expanduser("~/models")

MODELS_CONFIG = {
    "qwen": {
        "local_path": os.path.join(BASE_DIR, "qwen2.5-7b-safetywolf"),
        "display":    "Qwen 2.5 7B Safety",
        "type":       "qwen",
    },
    "llama": {
        "local_path": os.path.join(BASE_DIR, "llama3.1-8b-safetywolf"),
        "display":    "Llama 3.1 8B Safety",
        "type":       "llama",
    },
}

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Always respond in English only."
)

# ── Stop patterns ─────────────────────────────────────────────────────────────
QWEN_STOP_PATTERN = re.compile(
    r'<\|im_end\|>|<\|im_start\|>'
    r'|\n\s*(?:user|assistant)\b'
    r'|[^\x00-\x7F]+(?:user|assistant)',
    re.IGNORECASE,
)

LLAMA_STOP_PATTERN = re.compile(
    r'<\|eot_id\|>|<\|start_header_id\|>'
    r'|\n\s*(?:user|assistant)\b'
    r'|[^\x00-\x7F]+(?:user|assistant)',
    re.IGNORECASE,
)


class StopOnPattern(StoppingCriteria):
    def __init__(self, pattern, tokenizer, prompt_len):
        self.pattern    = pattern
        self.tokenizer  = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        new_tokens = input_ids[0][self.prompt_len:]
        decoded    = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return bool(self.pattern.search(decoded))


# ── Global model state ────────────────────────────────────────────────────────
current_model_key = None
model             = None
tokenizer         = None
is_switching      = False


def unload_model():
    global model, tokenizer, current_model_key
    if model is not None:
        print(f"[→] Unloading {current_model_key}...")
        del model
        del tokenizer
        model             = None
        tokenizer         = None
        current_model_key = None
        torch.cuda.empty_cache()
        gc.collect()
        print("[✓] VRAM cleared")


def load_model(key):
    global model, tokenizer, current_model_key
    cfg = MODELS_CONFIG[key]
    print(f"[↓] Loading {cfg['display']} from {cfg['local_path']}...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["local_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["local_path"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    current_model_key = key
    print(f"[✓] {cfg['display']} loaded")


def ensure_model(key):
    """Load key if not already loaded."""
    global is_switching
    if current_model_key == key:
        return
    is_switching = True
    unload_model()
    load_model(key)
    is_switching = False


# ── Prompt builders ───────────────────────────────────────────────────────────
def build_qwen_prompt(history, user_message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_llama_prompt(history, user_message):
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
    )
    for msg in history:
        prompt += (
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
            f"{msg['content']}<|eot_id|>"
        )
    prompt += (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


# ── Core inference ────────────────────────────────────────────────────────────
def run_inference(key, history, user_message):
    cfg         = MODELS_CONFIG[key]
    stop_pat    = QWEN_STOP_PATTERN if cfg["type"] == "qwen" else LLAMA_STOP_PATTERN

    text        = (build_qwen_prompt if cfg["type"] == "qwen" else build_llama_prompt)(
                      history, user_message)
    inputs      = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len  = inputs["input_ids"].shape[1]

    stop_crit   = StoppingCriteriaList([
        StopOnPattern(stop_pat, tokenizer, prompt_len)
    ])

    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eot_id      = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_ids    = [tokenizer.eos_token_id]
    for tid in [im_end_id, eot_id]:
        if tid and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
            repetition_penalty=1.3,
            stopping_criteria=stop_crit,
        )

    new_tokens = outputs[0][prompt_len:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Post-process: strip fake follow-up turns
    response = re.split(stop_pat, response)[0]
    # Strip trailing non-ASCII bleed (Thai/Chinese/Turkish chars)
    response = re.sub(r'[^\x00-\x7F][\s\S]*$', '', response).strip()

    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "current_model": current_model_key,
        "is_switching":  is_switching,
        "loaded":        current_model_key is not None,
    })


@app.route("/switch", methods=["POST"])
def switch():
    data = request.get_json(silent=True)
    key  = data.get("model") if data else None

    if key not in MODELS_CONFIG:
        return jsonify({"error": f"Unknown model: {key}"}), 400

    if key == current_model_key:
        return jsonify({"status": "already_loaded", "model": key})

    ensure_model(key)
    return jsonify({"status": "ok", "model": key})


@app.route("/generate", methods=["POST"])
def generate():
    data    = request.get_json(silent=True)
    key     = data.get("model")         if data else None
    message = data.get("message", "")   if data else ""
    history = data.get("history", [])   if data else []

    if key not in MODELS_CONFIG:
        return jsonify({"error": f"Unknown model: {key}"}), 400
    if not message.strip():
        return jsonify({"error": "Empty message"}), 400

    # Switch model if needed
    if key != current_model_key:
        ensure_model(key)

    response = run_inference(key, history, message.strip())
    return jsonify({"response": response, "model": key})


# ── Boot: load default model (Qwen) on startup ────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Model API starting — loading default model (Qwen)...")
    print("=" * 60)
    load_model("qwen")
    app.run(host="0.0.0.0", port=8000, debug=False)
