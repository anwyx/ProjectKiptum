# Configuration for the Qwen VL model
MODEL_CONFIG = {
    "MODEL_ID_OR_PATH": "Qwen/Qwen2.5-VL-7B-Instruct",
    "MIN_PIXELS": 512 * 28 * 28,
    "MAX_PIXELS": 2048 * 28 * 28,
    "MAX_NEW_TOKENS": 1024
}

# Default prompts and system messages
DEFAULT_PROMPTS = {
    "face_detection": "Outline the position of each face and output all the coordinates in JSON format."
}

DEFAULT_SYSTEM_MESSAGE = None