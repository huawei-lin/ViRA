"""Full-resolution photos (decoded once per process, small LRU), first views and zoom crops under the pixel budget x
(sizes from vira.protocol). A 108 MP photo decodes to about 324 MB: VIRA_ORIG_CACHE (default 12) bounds the cache. All
rollouts of one training row run in one verl AgentLoopWorker, so each version is decoded once."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

from PIL import Image

from . import protocol as P

Image.MAX_IMAGE_PIXELS = None


class _LRU:
    def __init__(self, cap: int):
        self.cap, self.d, self.lock = cap, OrderedDict(), threading.Lock()

    def get(self, k):
        with self.lock:
            v = self.d.get(k)
            if v is not None:
                self.d.move_to_end(k)
            return v

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)


_ORIG = _LRU(int(os.environ.get("VIRA_ORIG_CACHE", "12")))
_VIEW = _LRU(int(os.environ.get("VIRA_VIEW_CACHE", "48")))


def set_original_cache(n: int) -> None:
    """Keep up to n decoded photos (the evaluation runs many episodes in one process)."""
    _ORIG.cap = int(n)


def load_original(path: str) -> Image.Image:
    im = _ORIG.get(path)
    if im is None:
        with Image.open(path) as f:
            im = f.convert("RGB")
            im.load()
        _ORIG.put(path, im)
    return im


def resize_to(im: Image.Image, size) -> Image.Image:
    size = (int(size[0]), int(size[1]))
    return im.copy() if im.size == size else im.resize(size, Image.BICUBIC)


def first_view(path: str, x_pixels: int) -> Image.Image:
    """The whole photo under the pixel budget x."""
    key = (path, int(x_pixels))
    v = _VIEW.get(key)
    if v is None:
        im = load_original(path)
        v = resize_to(im, P.view_size(im.size, x_pixels))
        _VIEW.put(key, v)
    return v


def crop(path: str, ob, x_pixels: int) -> tuple:
    """(the crop shown to the model, its record) for a box in pixels of the photo, already clamped by
    protocol.frame_box_to_orig. The crop is cut from the full-resolution photo and resized as protocol.crop_record
    says."""
    rec = P.crop_record(ob, x_pixels)
    im = load_original(path).crop(tuple(int(v) for v in ob))
    return resize_to(im, rec["fed"]), rec
