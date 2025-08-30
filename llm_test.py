from pathlib import Path
import argparse
import json
import time
import traceback
from models.qwen import get_qwen_model
from models.prompts import get_prompt
from PIL import Image
import torch

# Configuration
RAW_DIR = Path("data/raw")

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Single image path (if omitted, iterate over data/raw)")
parser.add_argument("--prompt-key", default="generalize", help="Prompt key defined in models/prompts.py")
parser.add_argument("--max-new", type=int, default=256, help="Max new tokens")
parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress and LLM thinking process")
parser.add_argument("--show-raw", action="store_true", help="Show raw LLM response before JSON parsing")
args = parser.parse_args()

# Initialize model
print("🔧 Initializing Qwen model...")
model_start_time = time.time()
proc, model, device = get_qwen_model()
model_load_time = time.time() - model_start_time
print(f"✅ Model loaded in {model_load_time:.2f}s")

# Determine target images
if args.image:
    targets = [Path(args.image)]
    print(f"🎯 Processing single image: {args.image}")
else:
    targets = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
    print(f"📁 Found {len(targets)} images in {RAW_DIR}")

if not targets:
    print("❌ No images found.")
    exit(1)

print(f"🚀 Starting batch processing with prompt key: '{args.prompt_key}'")
print("=" * 60)

# Processing statistics
total_start_time = time.time()
successful_processes = 0
failed_processes = 0

# Process each image
for idx, img_path in enumerate(targets, 1):
    try:
        print(f"\n📋 [{idx}/{len(targets)}] Processing: {img_path.name}")
        if args.verbose:
            print(f"  📂 Full path: {img_path}")
        
        # === IMAGE PROCESSING START ===
        if args.verbose:
            print(f"  📸 Loading image: {img_path}")
            print(f"  🖼️  Image size: ", end="", flush=True)
        
        start_time = time.time()
        image = Image.open(img_path).convert("RGB")
        
        if args.verbose:
            print(f"{image.size}")
            print(f"  📝 Using prompt key: '{args.prompt_key}'")
        
        # Get system prompt
        system_prompt = get_prompt(args.prompt_key)
        
        if args.verbose:
            print(f"  🧠 System prompt length: {len(system_prompt)} characters")
            print(f"  💭 Preparing messages for LLM...")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Return JSON now."}
            ]},
        ]
        
        if args.verbose:
            print(f"  🔄 Applying chat template...")
        
        # Apply chat template
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        
        if args.verbose:
            print(f"  📊 Processing inputs (moving to {device})...")
        
        # Process inputs
        inputs = proc(images=[image], text=prompt, return_tensors="pt").to(device)
        
        if args.verbose:
            print(f"  🚀 Starting LLM generation (max_new_tokens={args.max_new})...")
            generation_start = time.time()
        
        # Generate response
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new)
        
        if args.verbose:
            generation_time = time.time() - generation_start
            print(f"  ⚡ Generation completed in {generation_time:.2f}s")
            print(f"  🔤 Decoding response...")
        
        # Decode response
        raw_response = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        
        total_time = time.time() - start_time
        if args.verbose:
            print(f"  ✅ Total processing time: {total_time:.2f}s")
            print(f"  📏 Raw response length: {len(raw_response)} characters")
        
        # === IMAGE PROCESSING END ===
        
        # Show raw response if requested
        if args.verbose or args.show_raw:
            print(f"  🔍 Raw LLM Response:")
            print(f"  {'─' * 50}")
            print(f"  {raw_response}")
            print(f"  {'─' * 50}")
        
        # Try to parse JSON
        if args.verbose:
            print(f"  🔧 Attempting JSON parsing...")
        
        parsed_json = None
        if not raw_response.startswith('{'):
            if args.verbose:
                print(f"  🔍 Response doesn't start with '{{', searching for JSON...")
            first = raw_response.find('{')
            last = raw_response.rfind('}')
            if first != -1 and last != -1 and last > first:
                json_substring = raw_response[first:last+1]
                if args.verbose:
                    print(f"  ✂️  Extracted JSON substring from position {first} to {last}")
            else:
                json_substring = raw_response
                if args.verbose:
                    print(f"  ⚠️  No JSON braces found, using full response")
        else:
            json_substring = raw_response
            if args.verbose:
                print(f"  ✅ Response starts with '{{', using as-is")
        
        # Parse JSON
        try:
            parsed_json = json.loads(json_substring)
            if args.verbose:
                print(f"  ✅ JSON parsing successful!")
            successful_processes += 1
        except Exception as json_error:
            if args.verbose:
                print(f"  ❌ JSON parsing failed: {json_error}")
            failed_processes += 1
        
        # Display final result
        print(f"  📄 Final Result:")
        if parsed_json is not None:
            print(json.dumps(parsed_json, indent=2))
        else:
            print(f"  ⚠️  Non-JSON response:")
            print(f"  {raw_response}")
            
    except Exception as e:
        print(f"  💥 Error processing {img_path.name}: {e}")
        if args.verbose:
            print(f"  📚 Full traceback:")
            traceback.print_exc()
        failed_processes += 1

# Final summary
total_time = time.time() - total_start_time
print("\n" + "=" * 60)
print(f"🏁 Batch processing completed!")
print(f"✅ Successful: {successful_processes}")
print(f"❌ Failed: {failed_processes}")
print(f"📊 Total time: {total_time:.2f}s")
if successful_processes > 0:
    print(f"⚡ Average time per successful image: {total_time/successful_processes:.2f}s")
