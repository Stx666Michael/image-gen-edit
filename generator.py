"""Shared FLUX.2 Klein pipeline loading + generation.

Used by both the CLI (``main.py``) and the web UI (``app.py``) so the model
is loaded the same way in both contexts.
"""
from __future__ import annotations

import gc
import threading
from typing import Iterable, Optional

import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from huggingface_hub import get_token
from PIL import Image

# ---------------------------------------------------------------------------
# Available models — Klein models use Flux2KleinPipeline with a built-in
# text encoder and run fully on-device.
# ---------------------------------------------------------------------------
MODELS = {
    "flux2-klein-4b": {"repo": "black-forest-labs/FLUX.2-klein-4B"},
    "flux2-klein-9b": {"repo": "black-forest-labs/FLUX.2-klein-9B"},
}

DEFAULTS = {"steps": 4, "guidance": 1.0, "width": 1024, "height": 1024, "seed": 42, "dtype": torch.float16}


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()

# Cache a single pipeline instance across calls (UI keeps it warm between
# messages; CLI uses it once per invocation).
_pipe = None
_pipe_repo: Optional[str] = None
_pipe_lock = threading.Lock()


def get_pipeline(model_key: str) -> Flux2KleinPipeline:
    """Return a cached pipeline for ``model_key``, loading/swapping if needed."""
    global _pipe, _pipe_repo
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choices: {list(MODELS)}")
    repo_id = MODELS[model_key]["repo"]
    with _pipe_lock:
        if _pipe is None or _pipe_repo != repo_id:
            # Drop any prior pipeline and free memory before loading a new one.
            _pipe = None
            _pipe_repo = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            if DEVICE == "cuda":
                # Use device_map="cuda" so accelerate loads each weight shard
                # directly onto the GPU and immediately releases the CPU copy.
                # This keeps system RAM usage to a few hundred MB instead of a
                # full fp16 copy (~8 GB for 4B), which is what crashed the T4
                # runtime (Colab caps system RAM at ~12 GB).
                #
                # For the 9B model (18 GB weights > T4's 16 GB VRAM) we fall
                # back to device_map="auto" so accelerate can spill the excess
                # to CPU RAM — still tight on T4, so an L4/A100 is recommended
                # for 9B.
                free, total = torch.cuda.mem_get_info()
                vram_gb = total / 1e9
                model_gb = 18.0 if "9b" in model_key.lower() else 8.0
                dmap = "cuda" if vram_gb >= model_gb + 1.0 else "auto"
                pipe = Flux2KleinPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=DEFAULTS["dtype"],
                    token=get_token(),
                    device_map=dmap,
                )
                # Slicing/tiling keeps VAE decode VRAM manageable at 1024x1024.
                if hasattr(pipe, "vae") and pipe.vae is not None:
                    if hasattr(pipe.vae, "enable_slicing"):
                        pipe.vae.enable_slicing()
                    if hasattr(pipe.vae, "enable_tiling"):
                        pipe.vae.enable_tiling()
            else:
                # MPS / CPU: load to CPU then offload to MPS on demand.
                # device_map is not reliably supported on MPS, so we keep the
                # original enable_model_cpu_offload() path.
                pipe = Flux2KleinPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=DEFAULTS["dtype"],
                    token=get_token(),
                    low_cpu_mem_usage=True,
                )
                pipe.enable_model_cpu_offload()

            _pipe = pipe
            _pipe_repo = repo_id
        return _pipe


_MAX_INPUT_DIM = 512


def _fit_image(img):
    """Resize *img* so its larger dimension is at most ``_MAX_INPUT_DIM`` px."""
    w, h = img.size
    if max(w, h) <= _MAX_INPUT_DIM:
        return img
    scale = _MAX_INPUT_DIM / max(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def generate(
    *,
    model: str,
    prompt: str,
    images: Optional[Iterable] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: int = DEFAULTS["seed"],
    width: int = DEFAULTS["width"],
    height: int = DEFAULTS["height"],
    on_step=None,
):
    """Run the pipeline and return the first generated PIL image.

    ``images`` may be a list of PIL images, file paths, or URLs (anything
    accepted by ``diffusers.utils.load_image``), or ``None`` for text-to-image.
    ``on_step``, if provided, is called as ``on_step(step, total)`` after each
    denoising step so callers can track progress.
    """
    steps = steps if steps is not None else DEFAULTS["steps"]
    guidance = guidance if guidance is not None else DEFAULTS["guidance"]

    input_images = None
    if images:
        input_images = [img if hasattr(img, "size") else load_image(img) for img in images]
        input_images = [_fit_image(img) for img in input_images]

    def _step_callback(pipe, step_index, timestep, callback_kwargs):
        if on_step is not None:
            on_step(step_index + 1, steps)
        return callback_kwargs

    pipe = get_pipeline(model)
    with _pipe_lock:
        # Use a CPU generator — when device_map is active diffusers places
        # tensors itself and a device-specific generator can cause a mismatch.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            image=input_images,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
            callback_on_step_end=_step_callback,
        )
    return result.images[0]
