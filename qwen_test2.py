#!/usr/bin/env python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import argparse
from pathlib import Path
from models.prompts import get_prompt  # added
from PIL import Image  # new import

# Try to import vision helper
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise SystemExit("Install qwen-vl-utils (package or repo providing qwen_vl_utils) before running.")

# --- CLI args to locate local image ---
parser = argparse.ArgumentParser()
parser.add_argument('--image', default='data/processed/WechatIMG74255_blurred.jpg', help='Path to local image file')
parser.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct', help='Model name')
parser.add_argument('--max-new', type=int, default=1024, help='Max new tokens to generate')
parser.add_argument('--prompt-key', default='generalize', help='Prompt key defined in models/prompts.py')
parser.add_argument('--max-side', type=int, default=0, help='If >0, downscale so the longer side == this value (maintains aspect ratio)')  # resize arg
parser.add_argument('--max-pixels', type=int, default=0, help='If >0, downscale so W*H <= this (after any max-side)')  # optional pixel budget
args = parser.parse_args()

image_path = Path(args.image)
if not image_path.exists():
    raise SystemExit(f"Image not found: {image_path}")
print(f"Using image: {image_path}")

# Load image
img = Image.open(image_path).convert('RGB')
orig_w, orig_h = img.size
resized = False

# First constraint: max side
if args.max_side and max(orig_w, orig_h) > args.max_side:
    scale = args.max_side / float(max(orig_w, orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    resized = True
    print(f"[Resize] Max side: {orig_w}x{orig_h} -> {new_w}x{new_h}")
else:
    new_w, new_h = orig_w, orig_h

# Second constraint: max pixels
if args.max_pixels and (new_w * new_h) > args.max_pixels:
    # compute uniform scale
    scale = (args.max_pixels / float(new_w * new_h)) ** 0.5
    final_w = max(1, int(round(new_w * scale)))
    final_h = max(1, int(round(new_h * scale)))
    img = img.resize((final_w, final_h), Image.LANCZOS)
    resized = True
    print(f"[Resize] Pixel budget: {new_w}x{new_h} -> {final_w}x{final_h} (<= {args.max_pixels} px)")
    new_w, new_h = final_w, final_h

if not resized:
    print(f"[Resize] No resizing applied ({orig_w}x{orig_h})")

DEVICE = (
    'cuda' if torch.cuda.is_available() else
    ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
)
print(f"Device: {DEVICE}")

# Load prompt text
system_prompt = get_prompt(args.prompt_key)

# Minimal load
processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(args.model, trust_remote_code=True)
model.to(DEVICE).eval()

# Messages include system role with privacy prompt
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},  # pass resized PIL image
            {"type": "text", "text": "Return JSON now."},
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
