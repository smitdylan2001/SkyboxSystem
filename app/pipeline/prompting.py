"""Caption instruction, caption parsing and prompt building.

Kept free of ML imports so it can be unit tested anywhere (see tests/test_prompting.py).

The captioner reports two things: the visual STYLE of the input (its medium and rendering, e.g. "stylized 3D
animated film render" or "realistic photograph") and the SCENE (the environment, extended beyond the frame). The
style leads the FLUX prompt and is repeated at its end, so a frame from an animated film becomes an animated
panorama instead of a photorealistic one: the 360 LoRA was trained on real panoramas and drifts towards photos
unless the prompt says otherwise.
"""

from __future__ import annotations

import re

INSTRUCTION = (
    "You write prompts for an AI that generates 360-degree panoramic skyboxes. The skybox must look as if it "
    "belongs to the same film, game, artwork or photo series as this image. Answer with exactly two lines:\n"
    "STYLE: the visual style of the image as a short noun phrase of at most 15 words that names the medium and "
    "how it is rendered, for example 'realistic photograph', 'stylized 3D animated film render with soft lighting', "
    "'cel-shaded anime illustration', 'watercolor painting', 'low-poly video game screenshot' or 'pixel art'. "
    "Describe only how it looks, not what it shows.\n"
    "SCENE: the environment, plausibly extended to every direction beyond the frame: type of place, landscape or "
    "architecture, ground, vegetation, sky and weather, time of day, lighting, colours and mood. Start directly "
    "with the place, for example 'a rocky beach at sunset with ...'; do not start with 'The image'. One paragraph "
    "of at most 60 words. Never mention people, characters, crowds, animals, vehicles, text, logos or the camera."
)

# Sentences (scene) or comma-separated parts (style) mentioning people are dropped: the generator should
# produce empty places.
_PEOPLE = re.compile(
    r"\b(people|persons?|man|men|woman|women|child(ren)?|kids?|boys?|girls?|crowds?|"
    r"pedestrians?|tourists?|someone|somebody|figures?|couples?|famil(y|ies)|visitors?|"
    r"hikers?|walkers?|riders?|cyclists?|players?|audience|spectators?|selfie|characters?)\b",
    re.IGNORECASE,
)

# "STYLE: ...", "**Style:** ...", "- scene: ..." (small models like markdown).
_LABEL = re.compile(r"^[\s*#>_\-]*(style|scene)[\s*_]*:[\s*_]*(.*)$", re.IGNORECASE)

# "The image shows ...", "This frame depicts ..." at the start of a scene description.
_LEAD_IN = re.compile(
    r"^(the|this)\s+(image|picture|photo|photograph|frame|scene|shot)\s+(shows|depicts|features|is of|contains)\s+",
    re.IGNORECASE,
)

MAX_STYLE_WORDS = 20


def parse_caption(text: str) -> tuple[str, str]:
    """Split the captioner's answer into (style, scene). Unlabelled text counts as scene, without a style."""
    style_parts: list[str] = []
    scene_parts: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        match = _LABEL.match(line)
        if match:
            current = style_parts if match.group(1).lower() == "style" else scene_parts
            current.append(match.group(2))
        elif line.strip():
            (current if current is not None else scene_parts).append(line)
    return clean_style(" ".join(style_parts)), clean_scene(" ".join(scene_parts))


def clean_style(text: str) -> str:
    """'A Stylized 3D animated render, with cartoon characters.' -> 'stylized 3D animated render'."""
    text = _squash(text).rstrip(" .;:")
    text = re.sub(r"^(it is|it's|this is|the image is)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in text.split(",")]
    text = ", ".join(p for p in parts if p and not _PEOPLE.search(p))
    words = text.split()
    if len(words) > MAX_STYLE_WORDS:
        text = " ".join(words[:MAX_STYLE_WORDS]).rstrip(",;")
    return _lower_first(text)


def clean_scene(text: str) -> str:
    """Drop sentences about people and a leading 'The image shows'."""
    text = _LEAD_IN.sub("", _squash(text))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not _PEOPLE.search(s)]
    text = " ".join(kept) if kept else _squash(_PEOPLE.sub("", text))
    return _lower_first(text)


def compose_description(scene: str, style: str = "", extra: str = "") -> str:
    """
    The text inserted into the prompt template. The style leads (CLIP only reads the first 77 tokens) and is
    repeated at the end, so it is not lost in a long scene description:
    'a stylized 3D animated film render of a sunny meadow with ... <extra> Everything is in the same
    stylized 3D animated film render style.'
    """
    scene, style, extra = scene.strip().rstrip("."), style.strip(), extra.strip()
    if not scene and extra:
        scene, extra = extra.rstrip("."), ""
    scene = _lower_first(scene)
    if style and scene:
        parts = [f"{_article(style)} {style} of {scene}."]
    elif style:
        parts = [f"{_article(style)} {style}."]
    else:
        parts = [f"{scene}." if scene else ""]
    if extra:
        parts.append(extra if extra[-1] in ".!?" else extra + ".")
    if style:
        parts.append(f"Everything is in the same {style} style.")
    return " ".join(p for p in parts if p)


def build_prompt(template: str, description: str, style: str = "") -> str:
    """Insert the description into the LoRA's prompt template ('... The image shows {prompt}', '{style}' optional)."""
    return template.format_map(_Fields(prompt=description.strip(), style=style.strip())).strip()


def inpaint_prompt(description: str) -> str:
    return f"{description} The place is completely empty, nobody is there.".strip()


class _Fields(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _squash(text: str) -> str:
    return " ".join(text.replace('"', "").replace("**", "").split())


def _lower_first(text: str) -> str:
    """'A misty lake' -> 'a misty lake', but acronyms like 'NYC' or '3D' stay as they are."""
    first_word = text.split(" ", 1)[0] if text else ""
    if first_word[:1].isupper() and (len(first_word) == 1 or first_word[1:].islower()):
        return text[0].lower() + text[1:]
    return text


def _article(phrase: str) -> str:
    return "an" if phrase[:1].lower() in "aeiou" else "a"
