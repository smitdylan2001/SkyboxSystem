"""Command-line client (Python stdlib only): submit an image, wait in the queue, download results.

    python tools/client.py --url https://sky.dylansmit.dev --key YOUR_KEY photo.jpg
    python tools/client.py --url http://192.168.1.10:8790 --key YOUR_KEY frame.png --prompt "at sunset" --out results
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def call(method: str, url: str, key: str, body: bytes | None = None, content_type: str | None = None) -> bytes:
    headers = {"X-API-Key": key}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:
        sys.exit(f"{method} {url} -> HTTP {err.code}: {err.read().decode(errors='replace')}")


def multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{file_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n".encode()
        + file_path.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--url", required=True, help="server base URL")
    ap.add_argument("--key", required=True, help="API key")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--depth-mode", default="")
    ap.add_argument("--keep-people", action="store_true")
    ap.add_argument("--style", type=float, default=-1.0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--png", action="store_true", help="also download the lossless skybox PNG")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    body, ctype = multipart(
        {
            "prompt": args.prompt,
            "use_caption": "false" if args.no_caption else "true",
            "seed": str(args.seed),
            "depth_mode": args.depth_mode,
            "remove_people": "false" if args.keep_people else "true",
            "style_strength": str(args.style),
        },
        args.image,
    )
    job = json.loads(call("POST", f"{base}/v1/jobs", args.key, body, ctype))
    print(f"job {job['id']} queued (position {job['position']})")

    last = None
    while job["status"] not in ("done", "failed", "cancelled"):
        time.sleep(2)
        job = json.loads(call("GET", f"{base}/v1/jobs/{job['id']}", args.key))
        line = f"{job['status']:<9} {job['stage']:<18} {job['progress'] * 100:5.1f}%"
        if job["status"] == "queued":
            line += f"  position {job['position']}"
        if job["eta_seconds"] >= 0:
            line += f"  eta {job['eta_seconds']}s"
        if line != last:
            print(line)
            last = line
    if job["status"] != "done":
        sys.exit(f"job {job['status']}: {job['error']}")

    print(f"caption: {job['caption']}\nprompt:  {job['prompt']}")
    out = args.out / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    files = {"skybox.jpg": job["skybox_url"], "depth.png": job["depth_url"], "depth.r16": job["depth_raw_url"]}
    if args.png:
        files["skybox.png"] = job["skybox_png_url"]
    for name, path in files.items():
        (out / name).write_bytes(call("GET", base + path, args.key))
    (out / "job.json").write_text(json.dumps(job, indent=2))
    print(f"saved to {out}  ({job['width']}x{job['height']}, depth {job['depth_mode']}, "
          f"near {job['depth_near']} m, far {job['depth_far']} m)")


if __name__ == "__main__":
    main()
