#!/usr/bin/env python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import argparse
from pathlib import Path

# Try to import vision helper
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise SystemExit("Install qwen-vl-utils (package or repo providing qwen_vl_utils) before running.")

# --- CLI args to locate local image ---
parser = argparse.ArgumentParser()
parser.add_argument('--image', default='data/processed/WechatIMG74255_blurred.jpg', help='Path to local image file')
parser.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct', help='Model name')
parser.add_argument('--max-new', type=int, default=128, help='Max new tokens to generate')
args = parser.parse_args()

image_path = Path(args.image)
if not image_path.exists():
    raise SystemExit(f"Image not found: {image_path}")
print(f"Using image: {image_path}")

DEVICE = (
    'cuda' if torch.cuda.is_available() else
    ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
)
print(f"Device: {DEVICE}")

# Minimal load
processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(args.model, trust_remote_code=True)
model.to(DEVICE).eval()

# Messages (single turn)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path.as_posix()},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Build inputs
chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[chat_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(DEVICE)

# Generate
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=args.max_new)
trimmed = out[:, inputs.input_ids.shape[1]:]
text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Output:\n", text.strip())
