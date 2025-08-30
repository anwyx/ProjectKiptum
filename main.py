import os
import argparse

from config.model_config import MODEL_CONFIG, DEFAULT_PROMPTS, DEFAULT_SYSTEM_MESSAGE
from src.models.qwen_model import QwenVLModel
from src.inference.face_detection import detect_faces
from src.visualization.annotation import process_vlm_detections
from src.utils.image_utils import resize_image, save_image

def parse_args():
    parser = argparse.ArgumentParser(description="Face detection using Qwen VL model")
    parser.add_argument("--image_path", type=str, default="data/raw/WechatIMG74255.jpg", 
                        help="Path to the input image")
    parser.add_argument("--output_path", type=str, default="outputs/annotated_image.jpg",
                        help="Path to save the annotated image")
    parser.add_argument("--prompt", type=str, 
                        default=DEFAULT_PROMPTS["face_detection"],
                        help="Prompt for the model")
    parser.add_argument("--max_size", type=int, default=800,
                        help="Maximum size for the output image")
    parser.add_argument("--save", action="store_true", 
                        help="Whether to save the annotated image")
    parser.add_argument("--display", action="store_true", 
                        help="Whether to display the annotated image")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if args.save:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load the model
    qwen_model = QwenVLModel(
        model_id_or_path=MODEL_CONFIG["MODEL_ID_OR_PATH"],
        min_pixels=MODEL_CONFIG["MIN_PIXELS"],
        max_pixels=MODEL_CONFIG["MAX_PIXELS"]
    )
    processor, model = qwen_model.load()
    
    # Run inference
    response, input_wh, resolution_wh, image = detect_faces(
        model=model,
        processor=processor,
        image_path=args.image_path,
        prompt=args.prompt,
        system_message=DEFAULT_SYSTEM_MESSAGE,
        max_new_tokens=MODEL_CONFIG["MAX_NEW_TOKENS"]
    )
    
    print("Model response:")
    print(response)
    
    # Process detections and annotate the image
    annotated_image = process_vlm_detections(
        response=response,
        input_wh=input_wh,
        resolution_wh=resolution_wh,
        image=image
    )
    
    # Resize for display
    annotated_image = resize_image(annotated_image, (args.max_size, args.max_size))
    
    # Save the annotated image if requested
    if args.save:
        save_image(annotated_image, args.output_path)
        print(f"Annotated image saved to {args.output_path}")
    
    # Display the image if requested
    if args.display:
        annotated_image.show()
    
    return annotated_image

if __name__ == "__main__":
    main()