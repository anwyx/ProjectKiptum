import os, sys, json
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

RAW_DIR = Path('data/raw')
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

system_prompt = """
        You are a privacy protection assistant. Your job is to identify all privacy-relevant parts in a given image.

        Pay special attention to:
        - Human faces (including children)
        - ID cards, passports, driver's licenses
        - Credit cards, bank cards
        - License plates
        - Documents with readable text
        - Computer screens
        - Barcodes, QR codes
        - Handwritten signatures
        - Location-revealing text (street signs, addresses)

        Valid categories: face, child, license_plate, id_card, credit_card, document, screen, barcode, signature, location_text, other

        Output ONLY valid minified JSON (no markdown, no extra text) with this schema:
        {"items":[{"category":"<category>","reason":"short reason","bbox":[x,y,w,h]}]}

        - Bounding boxes: integer pixel values [x,y,w,h], origin at top-left. If unsure, use [0,0,0,0].
        - If nothing is found, output {"items":[]}.
        - Do not output any explanation or extra text.

        Example:
        {"items":[{"category":"face","reason":"adult male face","bbox":[12,34,56,78]}]}
        """

# Collect target images
if len(sys.argv) > 1:
    targets = [Path(p) for p in sys.argv[1:] if Path(p).suffix.lower() in ALLOWED_EXT]
else:
    targets = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXT] if RAW_DIR.exists() else []

if not targets:
    print('No images to process.')
    sys.exit(0)

# Device selection
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'[Init] Torch {torch.__version__} | device={device}')

model_name = os.getenv('QWEN_VL_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct')
print(f'[Init] Loading model: {model_name}...')
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
dtype = torch.float16 if device in ('cuda', 'mps') else torch.float32
model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype)
model.to(device)
model.eval()
print('[Init] Model loaded.')

MAX_NEW = int(os.getenv('QWEN_MAX_NEW', '256'))

for idx, img_path in enumerate(targets, 1):
    try:
        print(f'\n[Progress] Processing image {idx}/{len(targets)}: {img_path}')
        image = Image.open(img_path).convert('RGB')
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Return JSON now."}
            ]}
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=[image], text=prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW)
        raw = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        print('[Response] Raw output:')
        print(raw)
        # Attempt JSON extraction
        if not raw.startswith('{'):
            first = raw.find('{'); last = raw.rfind('}')
            if first != -1 and last != -1 and last > first:
                candidate = raw[first:last+1]
            else:
                candidate = raw
        else:
            candidate = raw
        try:
            parsed = json.loads(candidate)
            print('[Response] Parsed JSON:')
            print(json.dumps(parsed, indent=2))
        except Exception:
            print('[Response] Could not parse JSON, showing raw output above.')
    except Exception as e:
        print(f'  Error processing {img_path}: {e}')
