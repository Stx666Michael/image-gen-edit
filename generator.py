"""Shared pipeline loading + generation for multiple diffusion models.

Used by both the CLI (``main.py``) and the web UI (``app.py``) so models 
are loaded the same way in both contexts.
"""
from __future__ import annotations

import threading
from typing import Iterable, Optional

import torch
from diffusers import Flux2KleinPipeline, StableDiffusion3Pipeline
from diffusers.utils import load_image
from huggingface_hub import get_token
from PIL import Image

# ---------------------------------------------------------------------------
# Available models — each with its pipeline class, dtype, and repo ID.
# Klein models use Flux2KleinPipeline with a built-in text encoder.
# SD 3.5 models use StableDiffusion3Pipeline with separate text encoders.
# ---------------------------------------------------------------------------
MODELS = {
    "flux2-klein-4b": {
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "pipeline_cls": Flux2KleinPipeline,
        "dtype": torch.float16,
        "defaults": {"steps": 4, "guidance": 1.0},
    },
    "flux2-klein-9b": {
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "pipeline_cls": Flux2KleinPipeline,
        "dtype": torch.float16,
        "defaults": {"steps": 4, "guidance": 1.0},
    },
    "sd-3.5-medium": {
        "repo": "stabilityai/stable-diffusion-3.5-medium",
        "pipeline_cls": StableDiffusion3Pipeline,
        "dtype": torch.bfloat16,
        "defaults": {"steps": 40, "guidance": 4.5},
    },
}


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()

# Global defaults shared across all models
GLOBAL_DEFAULTS = {"width": 1024, "height": 1024, "seed": 42, "dtype": torch.float16}


def get_model_defaults(model_key: str) -> dict:
    """Get default parameters for a specific model (steps, guidance, width, height, seed)."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choices: {list(MODELS)}")
    model_defaults = MODELS[model_key]["defaults"].copy()
    model_defaults.update({
        "width": GLOBAL_DEFAULTS["width"],
        "height": GLOBAL_DEFAULTS["height"],
        "seed": GLOBAL_DEFAULTS["seed"],
    })
    return model_defaults


# Cache a single pipeline instance across calls (UI keeps it warm between
# messages; CLI uses it once per invocation).
_pipe = None
_pipe_repo: Optional[str] = None
_pipe_lock = threading.Lock()


def get_pipeline(model_key: str):
    """Return a cached pipeline for ``model_key``, loading/swapping if needed."""
    global _pipe, _pipe_repo
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choices: {list(MODELS)}")
    model_cfg = MODELS[model_key]
    repo_id = model_cfg["repo"]
    with _pipe_lock:
        if _pipe is None or _pipe_repo != repo_id:
            # Drop any prior pipeline before loading a new one.
            _pipe = None
            _pipe_repo = None
            pipe = model_cfg["pipeline_cls"].from_pretrained(
                repo_id, torch_dtype=model_cfg["dtype"], token=get_token()
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
    seed: int = GLOBAL_DEFAULTS["seed"],
    width: int = GLOBAL_DEFAULTS["width"],
    height: int = GLOBAL_DEFAULTS["height"],
    on_step=None,
):
    """Run the pipeline and return the first generated PIL image.

    ``images`` may be a list of PIL images, file paths, or URLs (anything
    accepted by ``diffusers.utils.load_image``), or ``None`` for text-to-image.
    ``on_step``, if provided, is called as ``on_step(step, total)`` after each
    denoising step so callers can track progress.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}. Choices: {list(MODELS)}")
    model_defaults = MODELS[model]["defaults"]
    steps = steps if steps is not None else model_defaults["steps"]
    guidance = guidance if guidance is not None else model_defaults["guidance"]

    input_images = None
    if images:
        input_images = [img if hasattr(img, "size") else load_image(img) for img in images]
        input_images = [_fit_image(img) for img in input_images]

    def _step_callback(pipe, step_index, timestep, callback_kwargs):
        if on_step is not None:
            on_step(step_index + 1, steps)
        return callback_kwargs

    pipe = get_pipeline(model)
    pipeline_cls = MODELS[model]["pipeline_cls"]
    call_kwargs: dict = dict(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=torch.Generator(device=DEVICE).manual_seed(seed),
        callback_on_step_end=_step_callback,
    )
    if input_images is not None:
        call_kwargs["image"] = input_images
    if pipeline_cls is StableDiffusion3Pipeline:
        call_kwargs["max_sequence_length"] = 512
    with _pipe_lock:
        result = pipe(**call_kwargs)
    return result.images[0]
