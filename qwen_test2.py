#!/usr/bin/env python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import argparse
from pathlib import Path
from models.prompts import get_prompt  # added
from PIL import Image

# Try to import vision helper
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise SystemExit("Install qwen-vl-utils (package or repo providing qwen_vl_utils) before running.")

# --- CLI args to locate local image ---
parser = argparse.ArgumentParser()
parser.add_argument('--image', default='data/raw/WechatIMG74255.jpg', help='Path to local image file')
parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct', help='Model name')
parser.add_argument('--max-new', type=int, default=1024, help='Max new tokens to generate')
parser.add_argument('--prompt-key', default='detect_descriptive', help='Prompt key defined in models/prompts.py')
# Decoding / stability controls
parser.add_argument('--temperature', type=float, default=0.3, help='Temperature (>0 enables sampling)')
parser.add_argument('--top-p', type=float, default=0.9, help='Top-p nucleus sampling')
parser.add_argument('--repetition-penalty', type=float, default=1.0, help='Repetition penalty')
parser.add_argument('--force-fp16', action='store_true', help='Force fp16 even on devices where it may be unstable')
parser.add_argument('--force-fp32', action='store_true', help='Force fp32 precision')
parser.add_argument('--retry-fp32', action='store_true', help='On NaN/inf prob error, reload model in fp32 and retry')
# Debug / fallback
parser.add_argument('--debug', action='store_true', help='Verbose diagnostics')
parser.add_argument('--force-pil', action='store_true', help='Force PIL image ingestion instead of path reference')
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

# Heuristic: avoid fp16 on mps (can cause NaNs), allow override
if args.force_fp32:
    dtype = torch.float32
elif args.force_fp16:
    dtype = torch.float16
else:
    if DEVICE == 'mps':
        dtype = torch.float32  # safer
    elif DEVICE == 'cuda':
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32
print(f"Precision dtype: {dtype}")

# Load prompt text
system_prompt = get_prompt(args.prompt_key)

# Decide user suffix depending on prompt style
if args.prompt_key == 'detect_descriptive':
    user_tail = (
        "Describe ALL plausible privacy-relevant elements (faces, text docs, cards, plates, screens, IDs, barcodes, etc). "
        "Include uncertain possibilities; only say no content if you carefully checked every category."
    )
else:
    user_tail = "Return ONLY valid minified JSON now."

print('[Load] Loading processor/model...')
processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype)
model.to(DEVICE).eval()
print('[Load] Done.')

# Prepare image reference or PIL object
if args.force_pil:
    pil_image = Image.open(image_path).convert('RGB')
    image_content = pil_image
else:
    image_content = image_path.as_posix()

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [
        {"type": "image", "image": image_content},
        {"type": "text", "text": user_tail},
    ]}
]

print(f"[Debug] Processing image: {image_path}")

chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
if args.debug:
    print(f"[Debug] image_inputs len={len(image_inputs)} video_inputs len={len(video_inputs)}")
    if len(image_inputs):
        try:
            # Some versions store tensors; inspect shape
            first = image_inputs[0]
            if torch.is_tensor(first):
                print(f"[Debug] First image tensor shape: {tuple(first.shape)} dtype={first.dtype}")
        except Exception:
            pass

# If no image inputs and we did NOT force PIL, retry with PIL fallback automatically
if len(image_inputs) == 0 and not args.force_pil:
    print('[Warn] No image tensors produced. Retrying with PIL ingestion...')
    pil_image = Image.open(image_path).convert('RGB')
    messages[1]['content'][0]['image'] = pil_image
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    print(f"[Debug] After PIL fallback: image_inputs len={len(image_inputs)}")

inputs = processor(
    text=[chat_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(DEVICE)

if args.debug:
    if 'pixel_values' in inputs:
        pv = inputs['pixel_values']
        print(f"[Debug] pixel_values shape={tuple(pv.shape)} dtype={pv.dtype}")
    print(f"[Debug] pad_token_id={getattr(processor.tokenizer, 'pad_token_id', None)} eos_token_id={getattr(model.config, 'eos_token_id', None)}")

gen_kwargs = {
    'max_new_tokens': args.max_new,
    'repetition_penalty': args.repetition_penalty,
    'pad_token_id': getattr(processor.tokenizer, 'pad_token_id', None) or getattr(model.config, 'pad_token_id', None) or getattr(model.config, 'eos_token_id', None),
}
if args.temperature > 0:
    gen_kwargs.update({'do_sample': True, 'temperature': args.temperature, 'top_p': args.top_p})
else:
    gen_kwargs.update({'do_sample': False})
print(f"[Gen] kwargs: {gen_kwargs}")

attempts = 2 if args.retry_fp32 else 1
for attempt in range(1, attempts + 1):
    try:
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("Output:\n", text.strip())
        # Remove old truncation heuristic for descriptive mode (not reliable)
        if args.prompt_key != 'detect_descriptive' and not text.strip().endswith(']'):
            print('[Warning] Output does not end with "]". Possible early stop or truncation.')
        # Optional hint if model claimed none but we expect faces
        if args.prompt_key == 'detect_descriptive' and 'no privacy' in text.lower() and args.debug:
            print('[Debug] Model reported none. Consider raising temperature (e.g., 0.5) or using JSON prompt with few-shots.')
        break
    except RuntimeError as e:
        msg = str(e)
        print(f"[Error] Generation failed (attempt {attempt}): {msg}")
        nan_issue = 'probability tensor contains either' in msg or 'inf' in msg.lower()
        if nan_issue and args.retry_fp32 and attempt < attempts:
            print('[Recover] Reloading model in fp32 and retrying...')
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            model = AutoModelForVision2Seq.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float32)
            model.to(DEVICE).eval()
            continue
        else:
            raise
