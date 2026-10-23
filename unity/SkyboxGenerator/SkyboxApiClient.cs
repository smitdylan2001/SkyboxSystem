using System;
using System.Globalization;
using System.Threading;
using UnityEngine;
using UnityEngine.Networking;

namespace SkyboxGenerator
{
    public class SkyboxApiException : Exception
    {
        public long StatusCode { get; }
        public SkyboxApiException(long statusCode, string message) : base(message) => StatusCode = statusCode;
    }

    /// <summary>
    /// Thin async client for the skybox server (Unity 6+, uses Awaitable).
    /// All methods must be called from the main thread.
    /// </summary>
    public class SkyboxApiClient
    {
        public string BaseUrl { get; }
        public int RequestTimeoutSeconds { get; set; } = 120;

        readonly string _apiKey;

        public SkyboxApiClient(string baseUrl, string apiKey)
        {
            BaseUrl = baseUrl.TrimEnd('/');
            _apiKey = apiKey;
        }

        // ------------------------------------------------------------------ jobs

        /// <summary>Queue a new job. imageBytes = JPG/PNG/WebP file contents.</summary>
        public async Awaitable<SkyboxJob> SubmitAsync(byte[] imageBytes, string fileName = "input.jpg",
            string mimeType = "image/jpeg", SkyboxRequest options = null, CancellationToken ct = default)
        {
            options ??= new SkyboxRequest();
            var form = new WWWForm();
            form.AddBinaryData("image", imageBytes, fileName, mimeType);
            form.AddField("prompt", options.prompt ?? "");
            form.AddField("use_caption", options.useCaption ? "true" : "false");
            form.AddField("seed", options.seed.ToString(CultureInfo.InvariantCulture));
            form.AddField("steps", options.steps.ToString(CultureInfo.InvariantCulture));
            form.AddField("depth_mode", options.depthMode ?? "");
            form.AddField("remove_people", options.removePeople ? "true" : "false");
            form.AddField("style_strength", options.styleStrength.ToString(CultureInfo.InvariantCulture));
            if (!string.IsNullOrWhiteSpace(options.style)) form.AddField("style", options.style);

            using var req = UnityWebRequest.Post(BaseUrl + "/v1/jobs", form);
            await SendAsync(req, ct);
            return JsonUtility.FromJson<SkyboxJob>(req.downloadHandler.text);
        }

        public async Awaitable<SkyboxJob> GetJobAsync(string jobId, CancellationToken ct = default)
        {
            using var req = UnityWebRequest.Get($"{BaseUrl}/v1/jobs/{jobId}");
            await SendAsync(req, ct);
            return JsonUtility.FromJson<SkyboxJob>(req.downloadHandler.text);
        }

        /// <summary>Cancel a queued/running job, or delete a finished one (and its files).</summary>
        public async Awaitable CancelAsync(string jobId, CancellationToken ct = default)
        {
            using var req = UnityWebRequest.Delete($"{BaseUrl}/v1/jobs/{jobId}");
            req.downloadHandler = new DownloadHandlerBuffer();
            await SendAsync(req, ct);
        }

        /// <summary>Poll until the job is done. Throws SkyboxApiException if it failed or was cancelled.</summary>
        public async Awaitable<SkyboxJob> WaitForCompletionAsync(string jobId, Action<SkyboxJob> onUpdate = null,
            float pollSeconds = 2f, CancellationToken ct = default)
        {
            while (true)
            {
                var job = await GetJobAsync(jobId, ct);
                onUpdate?.Invoke(job);
                if (job.IsDone) return job;
                if (job.IsFinished)
                    throw new SkyboxApiException(0, $"Job {job.status}: {job.error}");
                await Awaitable.WaitForSecondsAsync(pollSeconds, ct);
            }
        }

        // ------------------------------------------------------------------ results

        /// <summary>Downloads the 4096x2048 skybox (sRGB, decoded off the main thread).</summary>
        public async Awaitable<Texture2D> DownloadSkyboxAsync(SkyboxJob job, bool png = false, CancellationToken ct = default)
        {
            var url = BaseUrl + (png ? job.skybox_png_url : job.skybox_url);
            using var req = UnityWebRequestTexture.GetTexture(url, nonReadable: true);
            await SendAsync(req, ct);
            var tex = DownloadHandlerTexture.GetContent(req);
            tex.name = $"Skybox_{job.id}";
            tex.wrapModeU = TextureWrapMode.Repeat; // equirect wraps horizontally
            tex.wrapModeV = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            tex.anisoLevel = 4;
            return tex;
        }

        /// <summary>Downloads the raw 16-bit depth map. Values: 1 = nearest, 0 = sky.</summary>
        public async Awaitable<Texture2D> DownloadDepthAsync(SkyboxJob job, CancellationToken ct = default)
        {
            using var req = UnityWebRequest.Get(BaseUrl + job.depth_raw_url);
            await SendAsync(req, ct);
            var tex = CreateDepthTexture(req.downloadHandler.data, job.depth_width, job.depth_height);
            tex.name = $"Depth_{job.id}";
            return tex;
        }

        // ------------------------------------------------------------------ helpers

        /// <summary>
        /// Builds a linear single-channel texture from raw little-endian uint16 data (bottom row first).
        /// Uses R16 (16-bit UNorm, samples as 0..1) where supported, otherwise RFloat (e.g. some GLES3 devices).
        /// </summary>
        public static Texture2D CreateDepthTexture(byte[] r16, int width, int height)
        {
            if (r16 == null || r16.Length != width * height * 2)
                throw new ArgumentException($"Depth data is {r16?.Length ?? 0} bytes, expected {width * height * 2}.");

            Texture2D tex;
            if (SystemInfo.SupportsTextureFormat(TextureFormat.R16))
            {
                tex = new Texture2D(width, height, TextureFormat.R16, false, true);
                tex.LoadRawTextureData(r16);
            }
            else
            {
                tex = new Texture2D(width, height, TextureFormat.RFloat, false, true);
                var values = new float[width * height];
                for (int i = 0; i < values.Length; i++)
                    values[i] = (r16[2 * i] | (r16[2 * i + 1] << 8)) / 65535f;
                tex.SetPixelData(values, 0);
            }
            tex.wrapModeU = TextureWrapMode.Repeat;
            tex.wrapModeV = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            tex.Apply(false, true);
            return tex;
        }

        /// <summary>
        /// Encodes any texture (Texture2D, RenderTexture from a VideoPlayer or camera, non-readable assets...)
        /// to JPG, downscaled so the longest side is at most maxSize.
        /// </summary>
        public static byte[] EncodeToJpg(Texture source, int quality = 90, int maxSize = 2048)
        {
            float scale = Mathf.Min(1f, (float)maxSize / Mathf.Max(source.width, source.height));
            int w = Mathf.Max(1, Mathf.RoundToInt(source.width * scale));
            int h = Mathf.Max(1, Mathf.RoundToInt(source.height * scale));

            var rt = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
            var previous = RenderTexture.active;
            Graphics.Blit(source, rt);
            RenderTexture.active = rt;
            var readable = new Texture2D(w, h, TextureFormat.RGB24, false, false);
            readable.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            readable.Apply();
            RenderTexture.active = previous;
            RenderTexture.ReleaseTemporary(rt);

            byte[] jpg = readable.EncodeToJPG(quality);
            UnityEngine.Object.Destroy(readable);
            return jpg;
        }

        async Awaitable SendAsync(UnityWebRequest req, CancellationToken ct)
        {
            req.SetRequestHeader("X-API-Key", _apiKey);
            req.timeout = RequestTimeoutSeconds;
            using (ct.Register(req.Abort))
                await req.SendWebRequest();
            ct.ThrowIfCancellationRequested();

            if (req.result != UnityWebRequest.Result.Success)
            {
                string body = null;
                try { body = req.downloadHandler?.text; }
                catch (Exception) { /* texture handlers may not expose text */ }
                string detail = string.IsNullOrEmpty(body) ? req.error : body;
                throw new SkyboxApiException(req.responseCode, $"{req.method} {req.url} -> {req.responseCode}: {detail}");
            }
        }
    }
}
