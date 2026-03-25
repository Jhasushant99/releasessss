from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

model_path = "models/qwen_32b"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def analyze_risk(json_data):
    prompt = f"""
You are a cybersecurity expert.

Analyze the following data and return FINAL JSON:

{json.dumps(json_data, indent=2)}

Output format:
{{
  "final_risk": "",
  "confidence": "",
  "action": "",
  "reason": ""
}}

Return ONLY JSON.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.0
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    return result
