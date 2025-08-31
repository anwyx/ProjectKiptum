import torch
from maestro.trainer.models.qwen_2_5_vl.checkpoints import load_model, OptimizationStrategy
from maestro.trainer.models.qwen_2_5_vl.inference import predict_with_inputs
from maestro.trainer.models.qwen_2_5_vl.loaders import format_conversation
import supervision as sv
from PIL import Image
import numpy as np
import json, re, os, cv2
from typing import Optional, Tuple

# ==================================================
# Workaround: disable HuggingFace allocator warmup
# ==================================================
import transformers.modeling_utils as modeling_utils
modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

# ==================================================
# Device setup
# ==================================================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

MODEL_ID_OR_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# Smaller pixel limits = less VRAM
MIN_PIXELS = 224 * 14 * 14
MAX_PIXELS = 1024 * 14 * 14

# ==================================================
# Load model
# ==================================================
processor, model = load_model(
    model_id_or_path=MODEL_ID_OR_PATH,
    optimization_strategy=OptimizationStrategy.NONE,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS
)

# ==================================================
# process_vision_info (with fallback)
# ==================================================
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("⚠️ qwen_vl_utils not installed, using fallback")

    def process_vision_info(conversation):
        images = [turn["image"] for turn in conversation if "image" in turn]
        return images, None

# ==================================================
# Inference function
# ==================================================
def run_qwen_2_5_vl_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    system_message: Optional[str] = None,
    max_new_tokens: int = 512,
) -> str:

    conversation = format_conversation(
        image=image, prefix=prompt, system_message=system_message
    )
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(conversation)

    inputs = processor(text=text, images=image_inputs, return_tensors="pt").to(device)

    response = predict_with_inputs(
        **inputs,
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=max_new_tokens,
    )[0]

    return response

# ==================================================
# Parse and clean JSON response
# ==================================================
def parse_bboxes(response: str):
    match = re.search(r"\[.*\]", response, re.S)
    if not match:
        return []

    fixed = match.group()
    # Try JSON parse
    try:
        data = json.loads(fixed)
    except:
        # Trim to last complete object
        last_bracket = fixed.rfind("}")
        if last_bracket != -1:
            try:
                data = json.loads(fixed[:last_bracket+1] + "]")
            except:
                return []
        else:
            return []

    # Deduplicate
    unique, seen = [], set()
    for det in data:
        if "bbox_2d" in det and len(det["bbox_2d"]) == 4:
            bbox = tuple(det["bbox_2d"])
            if bbox not in seen:
                unique.append(det)
                seen.add(bbox)
    return unique

# ==================================================
# Mosaic Blur helper
# ==================================================
def mosaic_blur_image(image: Image.Image, detections, mosaic_size: int = 15) -> Image:
    img = np.array(image.convert("RGB"))  # ensure 3-channel

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox_2d"])
        # Ensure within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        h, w = roi.shape[:2]
        if h < 10 or w < 10:
            continue

        roi_small = cv2.resize(
            roi,
            (max(1, w // mosaic_size), max(1, h // mosaic_size)),
            interpolation=cv2.INTER_LINEAR,
        )
        roi_mosaic = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = roi_mosaic
        print(f"Applied mosaic to bbox: {det['bbox_2d']}")

    return Image.fromarray(img)

# ==================================================
# Run test
# ==================================================
IMAGE_PATH = "data/raw/WechatIMG240.jpg"
PROMPT = "Outline the position of each card, and output all the coordinates in JSON format."

image = Image.open(IMAGE_PATH)

response = run_qwen_2_5_vl_inference(
    model=model,
    processor=processor,
    image=image,
    prompt=PROMPT,
    system_message=None,
)

print("Model response:\n", response)

detections = parse_bboxes(response)
print("Parsed detections:", detections)

# Apply mosaic blur
image = mosaic_blur_image(image=image, detections=detections, mosaic_size=20)
image.thumbnail((800, 800))
image.show()

# ==================================================
# Dynamic filename with sensitive_ prefix
# ==================================================
if detections:
    labels = sorted({det["label"] for det in detections})
    label_str = "_".join(labels)
else:
    label_str = "none"

save_path = os.path.expanduser(f"~/Downloads/sensitive_{label_str}.jpg")
image.save(save_path)

print(f"✅ Blurred image saved to {save_path}")