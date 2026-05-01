# Image Generation & Editing

Local image generation and editing using multiple SOTA models. Runs on CUDA (Linux/Windows), Apple Silicon MPS (macOS), or CPU.

## Requirements

- Python 3.10+
- GPU recommended: NVIDIA CUDA, or Apple Silicon MPS
- 16 GB VRAM/RAM for the 4B model; 24 GB for the 9B model
- Google Colab (T4/L4/A100) is supported via [`colab.ipynb`](colab.ipynb)

## Models
- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (generation/editing)
- [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) (generation/editing)
- [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) (generation only)

### Small Decoder (Klein models only)

Klein models optionally support the [FLUX.2-small-decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder), a distilled VAE decoder that is a drop-in replacement for the standard FLUX.2 decoder:

- ~1.4× faster decoding
- ~1.4× less VRAM at decode time (enables higher resolutions without running out of memory)
- ~28 M decoder parameters vs ~50 M in the full decoder
- Minimal to zero quality loss

Enable it with `--small-decoder` on the CLI, or tick the **Small decoder** checkbox in the web UI config panel (visible only when a Klein model is selected).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Log in to HuggingFace (one-time):

```bash
huggingface-cli login
```

You must also accept the model licenses on HuggingFace before downloading.

## Usage

```bash
python main.py [--model MODEL] [--prompt TEXT] [--image PATH ...] [--size PX] [--steps N] [--guidance F] [--seed N] [--output PATH]
```

### Web UI

A small chat-style web UI is also available. It supports multiple sessions
(each with its own model/steps/guidance/seed/size), accepts text or
text-with-uploaded-image inputs, returns the generated image as the reply,
and persists every session under `sessions/` so they reload on next launch.

```bash
pip install -r requirements.txt   # installs Flask
python app.py                     # http://127.0.0.1:5000
python app.py --host 0.0.0.0 --port 8000   # expose on the LAN
```

When no image is attached, by default the UI feeds the previous generated
output back in as the input image, letting you iterate on a result by chatting.
Untick the toggle in the composer to disable that and do pure text-to-image
on every turn.

### Run on Google Colab

No local GPU? Open [`colab.ipynb`](colab.ipynb) in Google
Colab and run the cells top to bottom:

1. **Runtime → Change runtime type → GPU** (T4 is sufficient for the 4B model).
2. The notebook clones this repo, installs dependencies, prompts for your
   Hugging Face token, and starts the same Flask web UI as `python app.py`.
3. A public `https://*.trycloudflare.com` URL is printed — click it to open
   the UI from any browser. Stop the cell to shut the server and tunnel down.

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `flux2-klein-4b` | `flux2-klein-4b`, `flux2-klein-9b`, or `sd-3.5-medium` |
| `--prompt` | *(hermit crab scene)* | Text prompt |
| `--image` | *(none)* | One or more input images for editing (local path or URL) |
| `--size` | `1024` | Output image size in pixels (square) |
| `--steps` | `4` | Number of inference steps |
| `--guidance` | `1.0` | Guidance scale |
| `--seed` | `42` | Random seed for reproducibility |
| `--output` | `<model>.png` | Output file path |
| `--small-decoder` | *(off)* | Use [FLUX.2-small-decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) for Klein models (~1.4× faster decode, ~1.4× lower VRAM) |

### Text-to-Image

```bash
# Quick run with 4B model at 512px (faster, less memory)
python main.py --model flux2-klein-4b --size 512 --output output/result.png

# High-res with 9B model
python main.py --model flux2-klein-9b --size 1024 --output output/klein9b.png

# Use small decoder for faster decode and lower VRAM (Klein models only)
python main.py --model flux2-klein-4b --small-decoder --output output/result_small_dec.png
python main.py --model flux2-klein-9b --small-decoder --size 1024 --output output/klein9b_small_dec.png

# Custom prompt
python main.py --model flux2-klein-4b --prompt "A futuristic city at sunset, cinematic lighting"

# More inference steps for higher quality
python main.py --model flux2-klein-9b --steps 8 --size 768 --output output/hq.png

# Using Stable Diffusion 3.5 Medium
python main.py --model sd-3.5-medium --prompt "A serene mountain landscape" --steps 40
```

### Image Editing

Pass one or more images with `--image` to use them as reference context. The model will edit or transform them according to the prompt.

```bash
# Edit a local image
python main.py --image photo.jpg --prompt "Make it look like a painting"

# Edit using a URL
python main.py --image https://example.com/cat.jpg --prompt "Add a hat to the cat"

# Multi-image reference (compose from multiple images)
python main.py --image img1.png img2.png --prompt "Combine these two scenes into one"
```

## Model Cache

Models are downloaded once and cached at `~/.cache/huggingface/hub/`. To store them on an external drive:

```bash
export HF_HOME=/path/to/external/drive/hf_cache
python main.py ...
```

Add the `export` line to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.) to make it permanent.
