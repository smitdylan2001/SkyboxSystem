"""Describe the input photo as a 360° scene prompt with a small VLM (Qwen3-VL)."""

from __future__ import annotations

import re

import torch
from PIL import Image

from .gpu import DEVICE, DTYPE, free_gpu

INSTRUCTION = (
    "You write prompts for an AI that generates 360-degree panoramic skyboxes. "
    "Describe the ENVIRONMENT in this photo and plausibly extend it to every direction "
    "beyond the frame: type of place, landscape or architecture, ground, vegetation, sky "
    "and weather, time of day, lighting, colours and mood. Start directly with the place, "
    "for example 'a rocky beach at sunset with ...'; do not address the viewer and do not "
    "start with 'The image' or 'You'. Never mention people, crowds, animals, vehicles, text, "
    "logos, the camera, or that it is a photo. Reply with one paragraph of at most 60 words."
)

# Sentences mentioning people are dropped (the generator should produce empty scenes).
_PEOPLE = re.compile(
    r"\b(people|persons?|man|men|woman|women|child(ren)?|kids?|boys?|girls?|crowds?|"
    r"pedestrians?|tourists?|someone|somebody|figures?|couples?|famil(y|ies)|visitors?|"
    r"hikers?|walkers?|riders?|cyclists?|players?|audience|spectators?|selfie)\b",
    re.IGNORECASE,
)


def caption_image(image: Image.Image, model_id: str, max_new_tokens: int) -> str:
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
    return clean_caption(text)


def clean_caption(text: str) -> str:
    text = " ".join(text.replace('"', "").split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not _PEOPLE.search(s)]
    return " ".join(kept) if kept else _PEOPLE.sub("", text)


def build_prompt(template: str, description: str) -> str:
    """Insert the description into the LoRA's prompt template ('... The image shows {prompt}')."""
    description = description.strip()
    first_word = description.split(" ", 1)[0]
    # "A misty lake ..." -> "a misty lake ..." (but keep acronyms like "NYC" intact).
    if first_word[:1].isupper() and (len(first_word) == 1 or first_word[1:].islower()):
        description = description[0].lower() + description[1:]
    return template.format(prompt=description).strip()
