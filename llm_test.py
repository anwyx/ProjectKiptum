from pathlib import Path
import argparse
import json
import time
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
    p.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress and LLM thinking process")
    p.add_argument("--show-raw", action="store_true", help="Show raw LLM response before JSON parsing")
    return p


def run_qwen_on_image(proc, model, device, image_path: Path, prompt_key: str, max_new: int, verbose: bool = False):
    if verbose:
        print(f"  📸 Loading image: {image_path}")
        print(f"  🖼️  Image size: ", end="", flush=True)
    
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    
    if verbose:
        print(f"{image.size}")
        print(f"  📝 Using prompt key: '{prompt_key}'")
    
    system_prompt = get_prompt(prompt_key)
    
    if verbose:
        print(f"  🧠 System prompt length: {len(system_prompt)} characters")
        print(f"  💭 Preparing messages for LLM...")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Return JSON now."}
        ]},
    ]
    
    if verbose:
        print(f"  🔄 Applying chat template...")
    
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    
    if verbose:
        print(f"  📊 Processing inputs (moving to {device})...")
    
    inputs = proc(images=[image], text=prompt, return_tensors="pt").to(device)
    
    if verbose:
        print(f"  🚀 Starting LLM generation (max_new_tokens={max_new})...")
        generation_start = time.time()
    
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new)
    
    if verbose:
        generation_time = time.time() - generation_start
        print(f"  ⚡ Generation completed in {generation_time:.2f}s")
        print(f"  🔤 Decoding response...")
    
    raw = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    
    total_time = time.time() - start_time
    if verbose:
        print(f"  ✅ Total processing time: {total_time:.2f}s")
        print(f"  📏 Raw response length: {len(raw)} characters")
    
    return raw


def main():
    args = build_parser().parse_args()
    
    print("🔧 Initializing Qwen model...")
    model_start_time = time.time()
    proc, model, device = get_qwen_model()
    model_load_time = time.time() - model_start_time
    print(f"✅ Model loaded in {model_load_time:.2f}s")

    if args.image:
        targets = [Path(args.image)]
        print(f"🎯 Processing single image: {args.image}")
    else:
        targets = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
        print(f"📁 Found {len(targets)} images in {RAW_DIR}")
    
    if not targets:
        print("❌ No images found.")
        return

    print(f"🚀 Starting batch processing with prompt key: '{args.prompt_key}'")
    print("=" * 60)
    
    total_start_time = time.time()
    successful_processes = 0
    failed_processes = 0

    for idx, img_path in enumerate(targets, 1):
        try:
            print(f"\n📋 [{idx}/{len(targets)}] Processing: {img_path.name}")
            if args.verbose:
                print(f"  📂 Full path: {img_path}")
            
            result = run_qwen_on_image(proc, model, device, img_path, args.prompt_key, args.max_new, args.verbose)
            
            if args.verbose or args.show_raw:
                print(f"  🔍 Raw LLM Response:")
                print(f"  {'─' * 50}")
                print(f"  {result}")
                print(f"  {'─' * 50}")
            
            # Try JSON extract
            if args.verbose:
                print(f"  🔧 Attempting JSON parsing...")
            
            parsed = None
            if not result.startswith('{'):
                if args.verbose:
                    print(f"  🔍 Response doesn't start with '{{', searching for JSON...")
                first = result.find('{'); last = result.rfind('}')
                if first != -1 and last != -1 and last > first:
                    sub = result[first:last+1]
                    if args.verbose:
                        print(f"  ✂️  Extracted JSON substring from position {first} to {last}")
                else:
                    sub = result
                    if args.verbose:
                        print(f"  ⚠️  No JSON braces found, using full response")
            else:
                sub = result
                if args.verbose:
                    print(f"  ✅ Response starts with '{{', using as-is")
            
            try:
                parsed = json.loads(sub)
                if args.verbose:
                    print(f"  ✅ JSON parsing successful!")
                successful_processes += 1
            except Exception as json_error:
                if args.verbose:
                    print(f"  ❌ JSON parsing failed: {json_error}")
                failed_processes += 1
            
            print(f"  📄 Final Result:")
            if parsed is not None:
                print(json.dumps(parsed, indent=2))
            else:
                print(f"  ⚠️  Non-JSON response:")
                print(f"  {result}")
                
        except Exception as e:
            print(f"  💥 Error processing {img_path.name}: {e}")
            if args.verbose:
                import traceback
                print(f"  📚 Full traceback:")
                traceback.print_exc()
            failed_processes += 1

    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"🏁 Batch processing completed!")
    print(f"✅ Successful: {successful_processes}")
    print(f"❌ Failed: {failed_processes}")
    print(f"📊 Total time: {total_time:.2f}s")
    if successful_processes > 0:
        print(f"⚡ Average time per successful image: {total_time/successful_processes:.2f}s")

if __name__ == "__main__":
    main()
