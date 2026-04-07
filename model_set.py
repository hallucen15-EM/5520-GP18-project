import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig

model_name = "Qwen/Qwen3-8B"
current_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(current_dir, "qwen")
os.makedirs(cache_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    cache_dir=cache_dir
)

def generate_text(prompt, max_new_tokens=128, enable_thinking=True):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

print(generate_text("If jack have 3 apples, he then buys 2 more, now how many does he have", enable_thinking=False))