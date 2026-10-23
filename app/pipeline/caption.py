"""Describe the input image (its visual style and its scene) with a small VLM (Qwen3-VL)."""

from __future__ import annotations

import torch
from PIL import Image

from .gpu import DEVICE, DTYPE, free_gpu
from .prompting import INSTRUCTION, parse_caption


def caption_image(image: Image.Image, model_id: str, max_new_tokens: int) -> tuple[str, str]:
    """Returns (style, scene), see prompting.parse_caption."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    image = image.copy()
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=DTYPE, device_map=str(DEVICE))
    processor = AutoProcessor.from_pretrained(model_id)
    try:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": INSTRUCTION}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
    finally:
        del model, processor
        free_gpu()
    return parse_caption(text)
