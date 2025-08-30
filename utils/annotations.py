import supervision as sv
from PIL import Image

def annotate_image(image: Image.Image, detections: sv.Detections) -> Image.Image:
    """Annotate an image with detections."""
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    if detections.mask is not None:
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        image = mask_annotator.annotate(image, detections)
    else:
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness)
        image = box_annotator.annotate(image, detections)

    label_annotator = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness - 1,
        smart_position=True
    )
    image = label_annotator.annotate(image, detections)
    return image

def process_vlm_detections(response, input_wh, resolution_wh, image):
    """Process VLM detections and annotate the image."""
    detections = sv.Detections.from_vlm(
        vlm=sv.VLM.QWEN_2_5_VL,
        result=response,
        input_wh=input_wh,
        resolution_wh=resolution_wh
    )
    
    annotated_image = annotate_image(image=image, detections=detections)
    return annotated_image