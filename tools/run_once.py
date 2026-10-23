"""Run the full pipeline once on an image, without the API or queue (for debugging/benchmarking).

Inside the container:
    docker compose exec skybox python tools/run_once.py --image /data/test.jpg --out /data/test_out

Use --vram-gb to cap PyTorch's allocator and simulate a smaller card (e.g. 11 on a 16 GB GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("AUTH_DISABLED", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", default=Path("run_once_out"), type=Path)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--style", type=float, default=-1.0, help="Redux style strength 0..1")
    ap.add_argument("--visual-style", default="", help="e.g. 'stylized 3D animated film render'; empty = detected")
    ap.add_argument("--keep-people", action="store_true")
    ap.add_argument("--vram-gb", type=float, default=0.0, help="cap the CUDA allocator (simulate a smaller GPU)")
    args = ap.parse_args()

    import torch

    if args.vram_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb / total))
        print(f"VRAM capped at {args.vram_gb:.1f} GiB of {total:.1f} GiB")

    from app.config import settings
    from app.pipeline import JobContext
    from app.pipeline.skybox import SkyboxPipeline

    pipe = SkyboxPipeline(settings)
    t0 = time.time()
    pipe.prefetch()
    print(f"prefetch: {time.time() - t0:.1f}s")

    params = {
        "prompt": args.prompt,
        "use_caption": not args.no_caption,
        "seed": args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1),
        "steps": args.steps or settings.steps,
        "depth_mode": settings.depth_mode,
        "remove_people": not args.keep_people,
        "style_strength": args.style if args.style >= 0 else settings.redux_strength,
        "style": args.visual_style,
    }
    last = [""]

    def progress(stage: str, fraction: float) -> None:
        if stage != last[0]:
            print(f"[{time.time() - t1:6.1f}s] {stage}")
            last[0] = stage

    ctx = JobContext(
        id="run_once",
        params=params,
        input_path=args.image,
        out_dir=args.out,
        progress=progress,
        set_text=lambda caption, prompt: print(f"caption: {caption}\nprompt:  {prompt}"),
    )
    t1 = time.time()
    result = pipe.run(ctx)
    print(json.dumps({"seed": params["seed"], "total_seconds": round(time.time() - t1, 1), **result}, indent=2))


if __name__ == "__main__":
    main()
