"""The model servers of the offline evaluation (``python -m vira.eval.run``): first views as files and the
OpenAI-compatible client.

Images reach the vLLM servers as file:// URLs at exactly the size recorded in the results (multiples of the patch
size, so the processor does not resize them again).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

from vira import agent as A
from vira import common as C
from vira import images as I


def view(path: str, budget: int, cache: str) -> tuple:
    """The whole image resized under `budget` pixels, as a cached PNG. Returns (file, (w, h), (W, H)). An image
    smaller than common.MIN_PIXELS is only rounded to the patch grid here; the server's processor enlarges it."""
    key = hashlib.sha1(f"{os.path.realpath(path)}|{budget}".encode()).hexdigest()[:16]
    dst = f"{cache}/views/{budget}/{key}.png"
    meta = dst + ".json"
    if os.path.exists(meta):
        with open(meta) as f:
            m = json.load(f)
        return dst, tuple(m["size"]), tuple(m["orig"])
    im = I.load_original(path)
    W, H = im.size
    size = C.resized_size(W, H, budget, min_pixels=4 * C.PATCH * C.PATCH)
    save_image(I.resize_to(im, size), dst)
    tmp = f"{meta}.{os.getpid()}_{threading.get_ident()}.tmp"   # several processes may build the same view at once
    with open(tmp, "w") as f:                                   # meta last: its presence means the PNG is complete
        json.dump({"size": list(size), "orig": [W, H], "src": path}, f)
    os.replace(tmp, meta)
    return dst, size, (W, H)


def save_image(im, path: str) -> str:
    """Save an image shown to the model (atomically); returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}_{threading.get_ident()}.tmp.png"
    im.save(tmp)
    os.replace(tmp, path)
    return path


def image_part(path: str) -> dict:
    return {"type": "image_url", "image_url": {"url": "file://" + os.path.abspath(path)}}


def server_index(key: str, n: int) -> int:
    """The server of an episode: fixed per key, so its turns hit the same prefix cache."""
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % n


class Caller:
    """call(key, messages) -> (text, usage): one reply from the vLLM servers, greedy, with the generation settings of
    vira.agent."""

    def __init__(self, urls, model_name):
        from openai import OpenAI

        self.clients = [OpenAI(base_url=u, api_key="EMPTY", timeout=1800) for u in urls]
        self.model = model_name

    def __call__(self, key, messages):
        cli = self.clients[server_index(key, len(self.clients))]
        t0 = time.time()
        r = cli.chat.completions.create(model=self.model, messages=messages, temperature=0.0, **A.GENERATION)
        return r.choices[0].message.content or "", {
            "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
            "sec": round(time.time() - t0, 3), "finish": r.choices[0].finish_reason}
