using System;

namespace SkyboxGenerator
{
    /// <summary>Job status as returned by GET /v1/jobs/{id}. Field names match the JSON exactly.</summary>
    [Serializable]
    public class SkyboxJob
    {
        // ReSharper disable InconsistentNaming
        public string id;
        public string status;          // queued | running | done | failed | cancelled
        public string stage;           // captioning, generating, estimating_depth, ...
        public float progress;         // 0..1
        public int position = -1;      // jobs ahead of this one (0 = running / next)
        public int eta_seconds = -1;   // estimated seconds until done (-1 = unknown yet)
        public string error;
        public string caption;
        public string prompt;
        public int seed;
        public string skybox_url;      // 4096x2048 JPG (relative URL)
        public string skybox_png_url;  // lossless PNG
        public string depth_url;       // 16-bit grayscale PNG (for tools, not for Unity runtime loading)
        public string depth_raw_url;   // raw little-endian uint16, Unity row order -> TextureFormat.R16
        public int width;
        public int height;
        public int depth_width;
        public int depth_height;
        public string depth_mode;      // linear | inverse | log
        public float depth_near;       // distance (approx. m) that maps to 1.0 (white)
        public float depth_far;        // distance (approx. m) that maps to ~0.0; sky is exactly 0
        public string depth_decode;    // formula to turn a sample back into a distance
        public int people_removed;
        // ReSharper restore InconsistentNaming

        public bool IsDone => status == "done";
        public bool IsFinished => status == "done" || status == "failed" || status == "cancelled";
    }

    /// <summary>Optional parameters for a new job.</summary>
    [Serializable]
    public class SkyboxRequest
    {
        public string prompt = "";          // extra description appended to the auto caption
        public bool useCaption = true;      // false = only use `prompt`
        public int seed = -1;               // -1 = random
        public int steps = 0;               // 0 = server default (28)
        public string depthMode = "";       // "" = server default (linear)
        public bool removePeople = true;
        public float styleStrength = -1f;   // 0..1 FLUX Redux photo style, -1 = server default
    }
}
