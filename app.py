"""Web UI for FLUX.2 Klein image generation.

A small Flask app that exposes a chat-like interface with multiple sessions.
Each session keeps its own configuration (model / steps / guidance / seed /
size), a history of user messages (text + optional uploaded image), and the
generated image responses. Sessions are persisted to ``sessions/`` so they
survive restarts.

Run with::

    python app.py            # http://127.0.0.1:5000
    python app.py --port 8000 --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from generator import DEFAULTS, MODELS, generate

# Suppress Werkzeug access-log noise for the high-frequency progress endpoint.
class _NoProgressLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/progress" not in record.getMessage()

logging.getLogger("werkzeug").addFilter(_NoProgressLog())

# ---------------------------------------------------------------------------
# Storage layout
#   sessions/
#     <session_id>/
#       session.json     # metadata, config, message history
#       <message_id>_in_<n>.<ext>     # uploaded input images
#       <message_id>_out.png          # generated output image
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# One generation at a time — the pipeline isn't thread-safe and GPU memory is
# tight. Each request waits its turn.
_gen_lock = threading.Lock()

# Progress tracking: {session_id: {"step": int, "total": int, "done": bool}}
_progress: dict[str, dict] = {}
_progress_lock = threading.Lock()


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _session_path(sid: str) -> Path:
    # Defend against path traversal — only accept hex/uuid-shaped ids.
    if not sid or not all(c.isalnum() or c in "-_" for c in sid):
        abort(400, "invalid session id")
    return SESSIONS_DIR / sid


def _load_session(sid: str) -> dict:
    path = _session_path(sid) / "session.json"
    if not path.exists():
        abort(404, "session not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_session(data: dict) -> None:
    path = _session_path(data["id"]) / "session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _new_session(name: str, config: dict) -> dict:
    sid = uuid.uuid4().hex[:12]
    data = {
        "id": sid,
        "name": name or f"Session {sid[:6]}",
        "created_at": _now(),
        "updated_at": _now(),
        "config": _normalize_config(config),
        "messages": [],
    }
    _session_path(sid).mkdir(parents=True, exist_ok=True)
    _save_session(data)
    return data


def _normalize_config(cfg: Optional[dict]) -> dict:
    cfg = cfg or {}
    model = cfg.get("model", "flux2-klein-4b")
    if model not in MODELS:
        abort(400, f"unknown model: {model}")
    return {
        "model": model,
        "steps": int(cfg.get("steps") or DEFAULTS["steps"]),
        "guidance": float(cfg.get("guidance") or DEFAULTS["guidance"]),
        "seed": int(cfg.get("seed") if cfg.get("seed") is not None else DEFAULTS["seed"]),
        "width": int(cfg.get("width") or DEFAULTS["width"]),
        "height": int(cfg.get("height") or DEFAULTS["height"]),
    }


def _list_sessions() -> list[dict]:
    items = []
    for d in SESSIONS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta = d / "session.json"
        if not meta.exists():
            continue
        try:
            with meta.open("r", encoding="utf-8") as f:
                data = json.load(f)
            items.append({
                "id": data["id"],
                "name": data.get("name", data["id"]),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "message_count": len(data.get("messages", [])),
                "config": data.get("config", {}),
            })
        except (OSError, json.JSONDecodeError):
            continue
    items.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return items


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload cap


@app.route("/")
def index():
    ui_defaults = {k: v for k, v in DEFAULTS.items() if k != "dtype"}
    return render_template("index.html", models=list(MODELS.keys()), defaults=ui_defaults)


@app.get("/api/sessions")
def api_list_sessions():
    return jsonify(_list_sessions())


@app.post("/api/sessions")
def api_create_session():
    body = request.get_json(silent=True) or {}
    data = _new_session(body.get("name", ""), body.get("config"))
    return jsonify(data), 201


@app.get("/api/sessions/<sid>")
def api_get_session(sid):
    return jsonify(_load_session(sid))


@app.patch("/api/sessions/<sid>")
def api_update_session(sid):
    data = _load_session(sid)
    body = request.get_json(silent=True) or {}
    if "name" in body:
        data["name"] = str(body["name"])[:200]
    if "config" in body:
        merged = {**data.get("config", {}), **(body.get("config") or {})}
        data["config"] = _normalize_config(merged)
    data["updated_at"] = _now()
    _save_session(data)
    return jsonify(data)


@app.delete("/api/sessions/<sid>")
def api_delete_session(sid):
    path = _session_path(sid)
    if not path.exists():
        abort(404, "session not found")
    # Only delete files we created (session.json + image files in the dir).
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
    path.rmdir()
    return ("", 204)


@app.get("/api/sessions/<sid>/files/<path:filename>")
def api_get_file(sid, filename):
    sdir = _session_path(sid)
    if not sdir.exists():
        abort(404)
    # send_from_directory guards against path traversal.
    return send_from_directory(sdir, filename)


@app.post("/api/sessions/<sid>/messages")
def api_send_message(sid):
    """Send a user message. Form fields:
        prompt         (required, str)
        images         (optional, multiple file uploads)
        use_last_output (optional, "1" to chain previous output as input)
        rewind_to      (optional, message id — truncate from that message onward)
    Returns the appended user message + generated assistant message.
    """
    data = _load_session(sid)
    sdir = _session_path(sid)

    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        abort(400, "prompt is required")

    # Rewind: drop the target message and everything after it.
    rewind_to = (request.form.get("rewind_to") or "").strip()
    if rewind_to:
        idx = next((i for i, m in enumerate(data["messages"]) if m.get("id") == rewind_to), None)
        if idx is not None:
            data["messages"] = data["messages"][:idx]

    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        abort(400, "prompt is required")

    msg_id = uuid.uuid4().hex[:8]
    saved_inputs: list[str] = []
    uploaded = request.files.getlist("images")
    for idx, f in enumerate(uploaded):
        if not f or not f.filename:
            continue
        ext = Path(f.filename).suffix.lower() or ".png"
        if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            ext = ".png"
        name = f"{msg_id}_in_{idx}{ext}"
        f.save(sdir / name)
        saved_inputs.append(name)

    # If no upload but session has prior generated images, optionally chain
    # the last output as the input image so the user can keep editing it.
    chain = request.form.get("use_last_output") == "1"
    if not saved_inputs and chain:
        for prev in reversed(data["messages"]):
            if prev.get("role") == "assistant" and prev.get("image"):
                saved_inputs = [prev["image"]]
                break

    # Append the user message immediately so the UI can render it even if
    # generation fails.
    user_msg = {
        "id": msg_id,
        "role": "user",
        "text": prompt,
        "images": saved_inputs,
        "created_at": _now(),
    }
    data["messages"].append(user_msg)
    data["updated_at"] = _now()
    _save_session(data)

    # Run the model (single global lock — only one generation at a time).
    cfg = data["config"]
    pil_inputs = [Image.open(sdir / n).convert("RGB") for n in saved_inputs] if saved_inputs else None
    started = time.time()
    with _progress_lock:
        _progress[sid] = {"step": 0, "total": cfg["steps"], "done": False}
    try:
        def _on_step(step, total):
            with _progress_lock:
                _progress[sid] = {"step": step, "total": total, "done": False}

        with _gen_lock:
            image = generate(
                model=cfg["model"],
                prompt=prompt,
                images=pil_inputs,
                steps=cfg["steps"],
                guidance=cfg["guidance"],
                seed=cfg["seed"],
                width=cfg["width"],
                height=cfg["height"],
                on_step=_on_step,
            )
    except Exception as e:  # surface failure to the UI
        with _progress_lock:
            _progress.pop(sid, None)
        err_msg = {
            "id": uuid.uuid4().hex[:8],
            "role": "assistant",
            "error": str(e),
            "created_at": _now(),
        }
        data["messages"].append(err_msg)
        data["updated_at"] = _now()
        _save_session(data)
        return jsonify({"user": user_msg, "assistant": err_msg}), 500

    with _progress_lock:
        _progress.pop(sid, None)
    out_name = f"{msg_id}_out.png"
    image.save(sdir / out_name)

    asst_msg = {
        "id": uuid.uuid4().hex[:8],
        "role": "assistant",
        "image": out_name,
        "elapsed_s": round(time.time() - started, 2),
        "config_snapshot": dict(cfg),  # includes width + height
        "created_at": _now(),
    }
    data["messages"].append(asst_msg)
    data["updated_at"] = _now()
    _save_session(data)

    return jsonify({"user": user_msg, "assistant": asst_msg})


@app.get("/api/sessions/<sid>/progress")
def api_progress(sid):
    with _progress_lock:
        info = _progress.get(sid)
    if info is None:
        return jsonify({"step": 0, "total": 0, "done": True})
    return jsonify(info)


@app.get("/api/models")
def api_models():
    return jsonify({
        "models": list(MODELS.keys()),
        "defaults": {k: v for k, v in DEFAULTS.items() if k != "dtype"},
    })


def main():
    parser = argparse.ArgumentParser(description="FLUX.2 Klein chat UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
