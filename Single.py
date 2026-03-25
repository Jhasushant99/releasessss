from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoModelForCausalLM
)
from PIL import Image
import torch
import json
import re

# =========================
# PATHS (CHANGE THESE)
# =========================
VL_MODEL_PATH = "models/qwen_vl_7b"
LLM_32B_PATH = "models/qwen_32b"
IMAGE_PATH = "test.jpg"

# =========================
# LOAD MODELS
# =========================
print("🔄 Loading Vision Model...")
vl_processor = AutoProcessor.from_pretrained(
    VL_MODEL_PATH,
    trust_remote_code=True
)

vl_model = AutoModelForVision2Seq.from_pretrained(
    VL_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("🔄 Loading 32B Model...")
tokenizer = AutoTokenizer.from_pretrained(
    LLM_32B_PATH,
    trust_remote_code=True
)

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_32B_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# =========================
# HELPER: CLEAN JSON
# =========================
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"error": "Invalid JSON"}

# =========================
# STEP 1: IMAGE → JSON
# =========================
def analyze_image(image_path):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": """Return ONLY valid JSON:

{
  "type": "",
  "objects": [],
  "risk_level": "",
  "description": ""
}"""}
            ]
        }
    ]

    text = vl_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = vl_processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to("cuda")

    output = vl_model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.0
    )

    result = vl_processor.batch_decode(output, skip_special_tokens=True)[0]

    print("\n🖼️ Vision Raw Output:\n", result)

    return extract_json(result)

# =========================
# STEP 2: JSON → FINAL (32B)
# =========================
def analyze_risk(json_data):
    prompt = f"""
Analyze this security data and return FINAL JSON:

{json.dumps(json_data, indent=2)}

Output:
{{
  "final_risk": "",
  "confidence": "",
  "action": "",
  "reason": ""
}}

ONLY JSON.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = llm_model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.0
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n🧠 32B Raw Output:\n", result)

    return extract_json(result)

# =========================
# MAIN PIPELINE
# =========================
def run():
    print("🚀 Starting Analysis...\n")

    vision_json = analyze_image(IMAGE_PATH)
    print("\n✅ Vision JSON:\n", vision_json)

    final_result = analyze_risk(vision_json)
    print("\n🔥 FINAL RESULT:\n", final_result)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run()
