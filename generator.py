"""Shared FLUX.2 Klein pipeline loading + generation.

Used by both the CLI (``main.py``) and the web UI (``app.py``) so the model
is loaded the same way in both contexts.
"""
from __future__ import annotations

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
            # Drop any prior pipeline before loading a new one.
            _pipe = None
            _pipe_repo = None
            pipe = Flux2KleinPipeline.from_pretrained(
                repo_id, torch_dtype=DEFAULTS["dtype"], token=get_token()
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
        result = pipe(
            prompt=prompt,
            image=input_images,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=torch.Generator(device=DEVICE).manual_seed(seed),
            callback_on_step_end=_step_callback,
        )
    return result.images[0]
