import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

def get_qwen_model(model_name: str | None = None, device: str | None = None):
    model_name = model_name or os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    # Prefer MPS (Apple Silicon), then CUDA, then CPU
    if device:
        device_sel = device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_sel = "mps"
    elif torch.cuda.is_available():
        device_sel = "cuda"
    else:
        device_sel = "cpu"
    print(f"[Qwen Model] Using device: {device_sel}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if device_sel == "cuda" else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device_sel)
    return processor, model, device_sel
