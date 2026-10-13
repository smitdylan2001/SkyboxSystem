using System;
using System.Threading;
using UnityEngine;
using UnityEngine.Events;

namespace SkyboxGenerator
{
    /// <summary>
    /// Drop-in component: sends an image (or a video frame via a RenderTexture) to the skybox server,
    /// waits in the queue, and assigns the resulting skybox + depth textures to your displacement material.
    /// </summary>
    public class SkyboxGeneratorComponent : MonoBehaviour
    {
        [Header("Server")]
        public string baseUrl = "https://sky.dylansmit.dev";
        [Tooltip("One of the API_KEYS from the server's .env. Don't commit real keys to public repos.")]
        public string apiKey = "";
        [Min(0.5f)] public float pollIntervalSeconds = 2f;

        [Header("Input")]
        [Tooltip("Texture2D, or a RenderTexture (e.g. VideoPlayer target) to send the current frame.")]
        public Texture sourceImage;
        public SkyboxRequest request = new SkyboxRequest();
        public bool generateOnStart;

        [Header("Output")]
        [Tooltip("Renderer of the displaced skybox mesh. Its material instance gets the textures.")]
        public Renderer targetRenderer;
        [Tooltip("Used when no renderer is set (e.g. a shared material or RenderSettings.skybox).")]
        public Material targetMaterial;
        public string colorProperty = "_MainTex";
        public string depthProperty = "_DepthTex";
        [Tooltip("Optional float properties that receive the near/far distance (approx. meters) of the depth encoding.")]
        public string depthNearProperty = "";
        public string depthFarProperty = "";

        [Header("Events")]
        public UnityEvent<SkyboxJob> onStatusChanged = new UnityEvent<SkyboxJob>();
        public UnityEvent<Texture2D, Texture2D> onCompleted = new UnityEvent<Texture2D, Texture2D>();
        public UnityEvent<string> onFailed = new UnityEvent<string>();

        [Header("State (read only)")]
        [SerializeField] string jobId;
        [SerializeField] string status;
        [SerializeField, Range(0, 1)] float progress;

        public Texture2D SkyboxTexture { get; private set; }
        public Texture2D DepthTexture { get; private set; }
        public SkyboxJob LastJob { get; private set; }
        public bool IsBusy { get; private set; }

        SkyboxApiClient _client;
        SkyboxApiClient Client => _client ??= new SkyboxApiClient(baseUrl, apiKey);

        async void Start()
        {
            if (generateOnStart && sourceImage != null)
                await GenerateAsync(sourceImage);
        }

        [ContextMenu("Generate From Source Image")]
        void GenerateFromInspector()
        {
            if (sourceImage != null) _ = GenerateAsync(sourceImage);
            else Debug.LogWarning("[Skybox] No source image set.", this);
        }

        [ContextMenu("Cancel Current Job")]
        public async void CancelCurrentJob()
        {
            if (string.IsNullOrEmpty(jobId)) return;
            try { await Client.CancelAsync(jobId, destroyCancellationToken); }
            catch (Exception e) { Debug.LogWarning($"[Skybox] Cancel failed: {e.Message}", this); }
        }

        /// <summary>Send a texture (Texture2D, RenderTexture, video frame...). Returns true on success.</summary>
        public Awaitable<bool> GenerateAsync(Texture source) =>
            GenerateAsync(SkyboxApiClient.EncodeToJpg(source), "frame.jpg", "image/jpeg");

        /// <summary>Send an encoded image file (JPG/PNG/WebP bytes). Returns true on success.</summary>
        public async Awaitable<bool> GenerateAsync(byte[] imageBytes, string fileName, string mimeType)
        {
            if (IsBusy)
            {
                Debug.LogWarning("[Skybox] A job is already running on this component.", this);
                return false;
            }
            IsBusy = true;
            CancellationToken ct = destroyCancellationToken;
            try
            {
                var job = await Client.SubmitAsync(imageBytes, fileName, mimeType, request, ct);
                Report(job);
                job = await Client.WaitForCompletionAsync(job.id, Report, pollIntervalSeconds, ct);
                await LoadResultsAsync(job, ct);
                return true;
            }
            catch (OperationCanceledException)
            {
                return false;
            }
            catch (Exception e)
            {
                status = "failed";
                Debug.LogError($"[Skybox] {e.Message}", this);
                onFailed.Invoke(e.Message);
                return false;
            }
            finally
            {
                IsBusy = false;
            }
        }

        /// <summary>Load the results of an existing (finished) job, e.g. after an app restart.</summary>
        public async Awaitable LoadResultsAsync(string existingJobId)
        {
            var job = await Client.GetJobAsync(existingJobId, destroyCancellationToken);
            Report(job);
            if (!job.IsDone) throw new SkyboxApiException(0, $"Job is {job.status}");
            await LoadResultsAsync(job, destroyCancellationToken);
        }

        async Awaitable LoadResultsAsync(SkyboxJob job, CancellationToken ct)
        {
            var sky = await Client.DownloadSkyboxAsync(job, png: false, ct);
            var depth = await Client.DownloadDepthAsync(job, ct);

            if (SkyboxTexture != null) Destroy(SkyboxTexture);
            if (DepthTexture != null) Destroy(DepthTexture);
            SkyboxTexture = sky;
            DepthTexture = depth;
            Apply(job);
            onCompleted.Invoke(sky, depth);
        }

        void Apply(SkyboxJob job)
        {
            Material mat = targetRenderer != null ? targetRenderer.material : targetMaterial;
            if (mat == null) return;
            if (!string.IsNullOrEmpty(colorProperty)) mat.SetTexture(colorProperty, SkyboxTexture);
            if (!string.IsNullOrEmpty(depthProperty)) mat.SetTexture(depthProperty, DepthTexture);
            if (!string.IsNullOrEmpty(depthNearProperty)) mat.SetFloat(depthNearProperty, job.depth_near);
            if (!string.IsNullOrEmpty(depthFarProperty)) mat.SetFloat(depthFarProperty, job.depth_far);
        }

        void Report(SkyboxJob job)
        {
            LastJob = job;
            jobId = job.id;
            status = job.status == "queued" ? $"queued (#{job.position + 1})"
                   : job.status == "running" ? job.stage
                   : job.status;
            progress = job.progress;
            onStatusChanged.Invoke(job);
        }

        void OnDestroy()
        {
            if (SkyboxTexture != null) Destroy(SkyboxTexture);
            if (DepthTexture != null) Destroy(DepthTexture);
        }
    }
}
