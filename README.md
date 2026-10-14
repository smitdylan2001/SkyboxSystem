# Skybox Generator

A self-hosted API that turns a photo (or a video frame) into a **360° equirectangular skybox** plus a
matching **16-bit depth map**, for a Unity shader that pushes skybox vertices outward by depth.

```
photo ─► Qwen3-VL caption ─► FLUX.1-dev + DiT360 panorama LoRA (2048×1024, seamless)
      ─► people check (Mask R-CNN) ─► inpaint people away (FLUX)
      ─► sky segmentation (SegFormer) + depth (MoGe-2, 12-view panorama merge)
      ─► Real-ESRGAN ×2 ─► skybox 4096×2048 + depth 4096×2048 (near = white, sky = black)
```

* The panorama is **newly generated, inspired by** the photo (scene, lighting, mood), and the generator is
  asked for an empty scene; anyone who still appears is detected and painted out.
* Jobs go into a **persistent FIFO queue** (SQLite): first in, first out, and it survives restarts.
* Built for an **RTX 4070 Super 12 GB shared with other containers and 16 GB of system RAM**. Each stage
  loads its model **entirely onto the GPU** (no CPU offload), runs, and frees the VRAM before the next
  stage. Nothing large is parked in RAM, and between jobs only the ~0.5 GB CUDA context stays on the GPU.

## Measured performance

Tested end to end (through the API, two queued jobs) on an RTX 5070 Ti with PyTorch's allocator
**capped at 11 GiB** to mimic the 4070 Super.

| Stage | Model | Peak VRAM | Time |
|---|---|---|---|
| Caption | Qwen3-VL-2B (bf16) | 4.1 GB | ~4 s |
| Prompt encoding | CLIP-L + T5-XXL (NF4) | 4.5 GB | ~5 s |
| Panorama, 28 steps (last 4 as seam ensemble) | FLUX.1-dev (NF4) + DiT360 LoRA | **7.3 GB** | ~115 s |
| Decode | FLUX VAE (wrap-around padding) | 6.0 GB | ~1 s |
| People detection | Mask R-CNN v2 | 0.9 GB | ~1 s |
| People removal (only if found) | FLUX.1-dev (NF4) inpaint, 512–1024 px crops | 7.2 GB | ~50 s |
| Sky mask | SegFormer-B5 ADE20K | 3.0 GB | ~1 s |
| Depth | MoGe-2 ViT-L (12 views) + CPU merge | 5.3 GB | ~11–20 s |
| Upscale ×2 | Real-ESRGAN x2plus | 1.4 GB | ~1 s |
| **Total per job** | | **max 7.3 GB** | **~140 s** (~190 s with people removal) |

Expect the 4070 Super to be roughly 20–30 % slower in the FLUX stage, so about 3 minutes per job. The
first job after a container start pays ~15 s of one-time CUDA warm-up, and the very first start
downloads the models (~20 GB).

---

## 1. Deploy on OpenMediaVault 8

The Docker image is built by GitHub Actions ([`.github/workflows/docker-image.yml`](.github/workflows/docker-image.yml))
on every push and published to GitHub Container Registry as **`ghcr.io/smitdylan2001/skyboxsystem`**:

| Tag | Built from |
|---|---|
| `latest` | `main` |
| `<branch>` | any other branch |
| `1.2.0` | a git tag `v1.2.0` |
| `sha-abc1234` | every commit (pin this for reproducible deploys) |

Each build is smoke-tested on the runner (all ML imports plus a full API/queue round trip in mock mode)
before you pull it. The server needs **no source code and no local build**, just the compose file and
your settings.

Prerequisites: Docker with NVIDIA GPU access (`docker run --rm --gpus all ubuntu nvidia-smi` works).
The image uses CUDA 13.0, so the host driver must be **580 or newer** (`nvidia-smi` shows the version).

1. **Folders**: create two folders on a data disk, e.g. `/srv/dev-disk-by-uuid-XXXX/appdata/skybox/models`
   and `.../skybox/data`. Put `models` on an SSD if you can, because every job reads ~20 GB of weights from
   disk (the 16 GB RAM can't cache them all).
2. **Compose file**: in OMV go to Services → Compose → Files → Add, name it `skybox`, and paste
   [`docker-compose.yml`](docker-compose.yml).
3. **Settings**: paste the contents of [`.env.example`](.env.example) into the file's Environment field and set:
   * `API_KEYS=`: a long random string (`openssl rand -hex 32`)
   * `MODELS_PATH=` and `DATA_PATH=`: the two absolute folder paths from step 1
4. **Start**: click Up (or, over SSH next to the files, `docker compose up -d`). Docker pulls the image
   (~10 GB, once) and the container downloads the models (~20 GB, once). Follow it with:
   ```bash
   docker logs -f skybox-generator
   ```
   Wait for `all models present`, then `curl http://localhost:8790/health` should show `"worker":"idle"`.

**Updating** to the newest build: in OMV press Pull and then Up, or over SSH:
```bash
docker compose pull && docker compose up -d
```
Queued jobs survive the restart. To pin a specific build, set `SKYBOX_TAG=sha-abc1234`.

**Building locally instead** (e.g. for a driver older than 580), from a checkout of this repo:
```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml build --build-arg BASE_IMAGE=pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime
```
Then start it with `docker compose -f docker-compose.yml -f docker-compose.build.yml up -d`.

### Nginx Proxy Manager (`sky.dylansmit.dev`)

1. DNS: an `A` record `sky` pointing to your public IP (as for your other subdomains).
2. NPM → Hosts → Proxy Hosts → Add:
   * Domain: `sky.dylansmit.dev`, scheme `http`, forward host = the server's LAN IP, port `8790`
     (or `skybox-generator` / port `8000` if NPM shares a Docker network with it)
   * Enable *Block Common Exploits*
   * SSL tab: request a Let's Encrypt certificate, enable *Force SSL* and *HTTP/2*
3. Advanced tab (optional; NPM's defaults already allow large uploads):
   ```nginx
   client_max_body_size 30m;
   proxy_read_timeout 120s;
   ```
   Don't add a `location /` block there, because NPM then drops its own proxy location.

Jobs are asynchronous (submit, then poll), so no request has to stay open for the whole generation.

### Testing without a GPU or models

Set `MOCK_PIPELINE=true` to get a synthetic but geometrically exact scene (a ground plane with a 2 m checker
grid and mountains at 150 m) after a few seconds. Use it to wire up Unity and tune your displacement shader:
the grid lines should look straight once displaced.

---

## 2. API

All `/v1` endpoints need the header `X-API-Key: <key>` (or `Authorization: Bearer <key>`).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | No auth. Worker state and queue length |
| `POST` | `/v1/jobs` | Queue an image (multipart form), returns `202` and the job JSON |
| `POST` | `/v1/test-jobs` | **Test run without generation**: same queue, worker, result files and URLs, but returns a checkerboard test scene in seconds (works without GPU). `image` is optional |
| `GET` | `/v1/jobs/{id}` | Status, queue position, ETA, result URLs |
| `GET` | `/v1/jobs/{id}/skybox.jpg` | 4096×2048 skybox (JPEG q92) |
| `GET` | `/v1/jobs/{id}/skybox.png` | Same, lossless |
| `GET` | `/v1/jobs/{id}/depth.r16` | Raw depth, little-endian uint16, **bottom row first** (Unity layout) |
| `GET` | `/v1/jobs/{id}/depth.png` | Same depth as a 16-bit grayscale PNG (top row first, for tools) |
| `DELETE` | `/v1/jobs/{id}` | Cancel a queued/running job, or delete a finished one |
| `GET` | `/v1/jobs?limit=20` | Recent jobs |

**`POST /v1/jobs` form fields**

| Field | Default | Meaning |
|---|---|---|
| `image` (file) | required | JPG/PNG/WebP, max `MAX_UPLOAD_MB` |
| `prompt` | `""` | Extra description appended to the auto caption ("at sunset", "in winter") |
| `use_caption` | `true` | `false` = ignore the photo's content and use only `prompt` |
| `seed` | `-1` | `-1` = random; the same seed and prompt give the same panorama |
| `steps` | server `STEPS` | 8–60 |
| `depth_mode` | server `DEPTH_MODE` | `linear` / `inverse` / `log` (see below) |
| `remove_people` | `true` | Detect and inpaint people |
| `style_strength` | server `REDUX_STRENGTH` | 0–1, feed the photo's look in via FLUX Redux (needs `HF_TOKEN`) |

**Job JSON** (flat and null-free, so Unity's `JsonUtility` reads it directly):

```json
{
  "id": "3f9c...", "status": "done", "stage": "done", "progress": 1.0,
  "position": -1, "eta_seconds": -1, "error": "",
  "caption": "A grassy plain with rolling hills ...", "prompt": "This is a panorama. The image shows ...",
  "seed": 42,
  "skybox_url": "/v1/jobs/3f9c.../skybox.jpg", "depth_raw_url": "/v1/jobs/3f9c.../depth.r16", "...": "...",
  "width": 4096, "height": 2048, "depth_width": 4096, "depth_height": 2048,
  "depth_mode": "linear", "depth_near": 5.5, "depth_far": 365.3,
  "depth_decode": "distance = far - v * (far - near)", "people_removed": 0
}
```

`status` is one of `queued`, `running`, `done`, `failed`, `cancelled`. While queued, `position` counts the
jobs ahead (0 = next). `eta_seconds` comes from the average of recent jobs (`-1` until one has finished).

**Test runs (`/v1/test-jobs`).** To check the whole chain (API key, proxy, upload, queue, polling, downloads,
16-bit depth in Unity, the displacement shader) without waiting for or depending on the GPU pipeline, post
to `/v1/test-jobs` instead of `/v1/jobs`. It accepts the same form fields and returns a normal job (with
`"test": true`). The result is a synthetic scene with exact depth:

* a 2 m black/white checkerboard ground (its lines must stay straight and evenly spaced once displaced),
* a ring of mountains at 150 m, and sky with grid lines every 30°,
* your uploaded image on a yellow-framed panel 6 m in front of the camera (equirect centre = front),
* N / E / S / W markers on the horizon.

Test jobs run even when the GPU pipeline failed to start. `/health` then shows `"pipeline_ready": false` and
the reason in `worker_error`, and real jobs fail immediately with that message instead of waiting.

**Quick test from any PC** (Python standard library only):

```bash
python tools/client.py --url https://sky.dylansmit.dev --key YOUR_KEY photo.jpg
```

Add `--test` for a checkerboard test run (the image is then optional).

---

## 3. Unity (6.x)

Copy `unity/SkyboxGenerator/` into your project's `Assets/`. The client was compile-checked and run
against Unity 6000.7.0b4.

* **`SkyboxGeneratorComponent`** is drop-in: set *Base Url* and *Api Key*, assign a *Source Image*
  (a `Texture2D`, or the `RenderTexture` of a `VideoPlayer` / camera to send the current frame), pick the
  *Target Renderer* (your displaced sphere), and set the texture property names your shader uses. Then use
  *Generate From Source Image* from the component's context menu, or call it from code:
  ```csharp
  bool ok = await generator.GenerateAsync(videoPlayer.targetTexture);
  ```
* **`SkyboxApiClient`** is the plain async API (`SubmitAsync`, `WaitForCompletionAsync`,
  `DownloadSkyboxAsync`, `DownloadDepthAsync`, `CancelAsync`) for your own flow.

The depth map is loaded from the raw `.r16` as a linear `TextureFormat.R16` texture (16-bit, samples as
0..1). Unity's `LoadImage` would reduce a 16-bit PNG to 8-bit. On devices without R16 support (some GLES3
phones) the client falls back to `RFloat`. Both textures use *Repeat* horizontally and *Clamp* vertically.

### Depth map format

* The value is the **distance along the view ray** (not planar z), which is exactly what you need to push a
  sphere vertex outward along its direction.
* **1.0 (white) = nearest, 0 = sky.** Real geometry is always ≥ 1/65535, so `d == 0` means sky.
* `depth_near` / `depth_far` are the distances (approx. meters) that map to 1 and ~0, taken from the
  0.5 / 99.5 percentiles. Mapping per `depth_mode`:

| mode | `v` → distance | When to use |
|---|---|---|
| `linear` (default) | `far - v·(far - near)` | Physically proportional; a shader that lerps radius by `v` gets correct perspective |
| `inverse` | `1 / (1/far + v·(1/near - 1/far))` | More precision for nearby detail (disparity-like) |
| `log` | `far · (near/far)^v` | In between; good for large outdoor ranges |

Minimal vertex displacement for a unit-sphere mesh, sampling both maps with the same UVs as your color
texture:

```hlsl
sampler2D _DepthTex;
float _SkyRadius;    // radius used for sky (d == 0), e.g. 1000
float _NearRadius;   // radius for the nearest geometry (d == 1), e.g. 5
float _FarRadius;    // radius for the farthest geometry, e.g. 400

v2f vert (appdata v)
{
    float d = tex2Dlod(_DepthTex, float4(v.uv, 0, 0)).r;
    float3 dir = normalize(v.vertex.xyz);
    float radius = d > 0 ? lerp(_FarRadius, _NearRadius, d) : _SkyRadius;   // linear mode
    v.vertex.xyz = dir * radius;
    ...
}
```

For true metric scale, feed `depth_near` / `depth_far` into the shader (the component can set two float
properties for you) and use `radius = far - d * (far - near)`. The camera should sit at the sphere's
centre at roughly human eye height, because that is how the panoramas are generated.

---

## 4. Configuration

Everything is set through environment variables; see `.env.example` for the full annotated list.
The most useful ones:

| Variable | Default | Notes |
|---|---|---|
| `API_KEYS` | required | Comma-separated |
| `OUT_WIDTH` | `4096` | Output width (height = width/2). 8192 runs the ×2 upscaler twice |
| `STEPS` | `28` | 20 is noticeably faster and slightly softer |
| `DEPTH_MODE` | `linear` | Per-job override with the `depth_mode` field |
| `REMOVE_PEOPLE` | `true` | |
| `REDUX_STRENGTH` | `0` | Try 0.2–0.5 for a closer match to the photo's look (needs `HF_TOKEN`; this path is not covered by the test runs) |
| `PEOPLE_MAX_REGIONS` | `8` | Max inpainting crops per job (each crop also covers every other person inside it) |
| `SEAM_ROLL_TOKENS` / `SEAM_ENSEMBLE_STEPS` | `8` / `4` | Seam handling (see `app/pipeline/flux.py`) |
| `MAX_QUEUE` / `RESULT_TTL_HOURS` | `50` / `168` | Queue limit (429 when full) and result retention |
| `MOCK_PIPELINE` | `false` | Fake results without GPU |

Debug a single image inside the container, bypassing the queue:

```bash
docker compose exec skybox python tools/run_once.py --image /data/test.jpg --out /data/test_out
```

## 5. Troubleshooting

* **`CUDA out of memory`**: another container was using the GPU at that moment. The job is marked failed
  with a clear message; resubmit it. The largest stage needs ~7.3 GB plus ~0.5 GB of CUDA context.
* **Worker `stopped` in `/health`**: read `worker_error`. Usually the GPU isn't visible in the container
  (missing `deploy.resources` / NVIDIA runtime) or the driver is too old for CUDA 13.
* **401 on gated models**: only FLUX Redux (`REDUX_STRENGTH > 0`) is gated. Accept its license on Hugging Face
  and set `HF_TOKEN`.
* **Slow jobs**: most of the overhead is loading weights from disk. Move `MODELS_PATH` to an SSD.

## Licenses of the models

FLUX.1-dev (and its NF4 conversion): **FLUX.1 [dev] Non-Commercial License**. DiT360 LoRA: MIT.
Qwen3-VL: Apache-2.0. MoGe-2: MIT. SegFormer: NVIDIA research license (non-commercial; the MIT
alternative `openmmlab/upernet-convnext-small` can be set via `SKY_MODEL`). Real-ESRGAN: BSD-3.
Mask R-CNN (torchvision): BSD-3. The service's own code has no restrictions.
