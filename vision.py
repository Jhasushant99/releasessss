from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import re, json

model_path = "models/qwen_vl_7b"

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"error": "Invalid JSON"}

def analyze_image(image_path):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": """You are a strict JSON generator.

Return ONLY valid JSON:

{
  "type": "",
  "objects": [],
  "risk_level": "",
  "description": ""
}"""}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.0
    )

    result = processor.batch_decode(output, skip_special_tokens=True)[0]

    return extract_json(result)
