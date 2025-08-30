from pathlib import Path
import argparse
import json
from models.qwen import get_qwen_model
from models.prompts import get_prompt
from PIL import Image
import torch

RAW_DIR = Path("data/raw")

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="Single image path (if omitted, iterate over data/raw)")
    p.add_argument("--prompt-key", default="generalize", help="Prompt key defined in models/prompts.py")
    p.add_argument("--max-new", type=int, default=256, help="Max new tokens")
    return p


def run_qwen_on_image(proc, model, device, image_path: Path, prompt_key: str, max_new: int):
    image = Image.open(image_path).convert("RGB")
    system_prompt = get_prompt(prompt_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Return JSON now."}
        ]},
    ]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(images=[image], text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new)
    raw = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return raw


def main():
    args = build_parser().parse_args()
    proc, model, device = get_qwen_model()

    if args.image:
        targets = [Path(args.image)]
    else:
        targets = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
    if not targets:
        print("No images found.")
        return

    for idx, img_path in enumerate(targets, 1):
        try:
            print(f"[{idx}/{len(targets)}] {img_path}")
            result = run_qwen_on_image(proc, model, device, img_path, args.prompt_key, args.max_new)
            # Try JSON extract
            parsed = None
            if not result.startswith('{'):
                first = result.find('{'); last = result.rfind('}')
                if first != -1 and last != -1 and last > first:
                    sub = result[first:last+1]
                else:
                    sub = result
            else:
                sub = result
            try:
                parsed = json.loads(sub)
            except Exception:
                pass
            if parsed is not None:
                print(json.dumps(parsed, indent=2))
            else:
                print(result)
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
