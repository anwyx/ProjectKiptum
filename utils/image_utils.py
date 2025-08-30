from PIL import Image

def load_image(image_path):
    """Load an image from the given path."""
    return Image.open(image_path)

def resize_image(image, max_size=(800, 800)):
    """Resize an image while maintaining its aspect ratio."""
    image_copy = image.copy()
    image_copy.thumbnail(max_size)
    return image_copy

def save_image(image, output_path):
    """Save an image to the given path."""
    image.save(output_path)