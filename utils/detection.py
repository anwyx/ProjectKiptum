from PIL import Image
from typing import Optional, Tuple

from maestro.trainer.models.qwen_2_5_vl.inference import predict_with_inputs
from maestro.trainer.models.qwen_2_5_vl.loaders import format_conversation
from maestro.trainer.common.utils.device import parse_device_spec
from src.utils.qwen_vl_utils import process_vision_info

def run_qwen_2_5_vl_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    system_message: Optional[str] = None,
    device: str = "auto",
    max_new_tokens: int = 1024,
) -> Tuple[str, Tuple[int, int]]:
    device = parse_device_spec(device)
    conversation = format_conversation(image=image, prefix=prompt, system_message=system_message)
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversation)

    inputs = processor(
        text=text,
        images=image_inputs,
        return_tensors="pt",
    )

    input_h = inputs['image_grid_thw'][0][1] * 14
    input_w = inputs['image_grid_thw'][0][2] * 14

    response = predict_with_inputs(
        **inputs,
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=max_new_tokens
    )[0]

    return response, (int(input_w), int(input_h))

def detect_faces(model, processor, image_path, prompt, system_message=None, max_new_tokens=1024):
    """
    Detects faces in the given image using the Qwen VL model.
    
    Returns:
        tuple: (response, input_wh, resolution_wh, image)
    """
    image = Image.open(image_path)
    resolution_wh = image.size
    
    response, input_wh = run_qwen_2_5_vl_inference(
        model=model,
        processor=processor,
        image=image,
        prompt=prompt,
        system_message=system_message,
        max_new_tokens=max_new_tokens
    )
    
    return response, input_wh, resolution_wh, image