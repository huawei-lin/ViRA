"""The interaction protocol of ViRA, framework-free: one copy for training (verl, vira.train), evaluation
(vira.eval) and inference (vira.infer): the prompts, the tool calls, the box conversion and crop sizes of the zoom
tool, and the reward.

Prompts. The system prompt and the answer format are SYSTEM_PROMPT_V2 and USER_PROMPT_V2 of DeepEyes
(github.com/Visual-Agent/DeepEyes, verl/workers/agent/envs/mm_process_engine/prompt.py), with three additions: the box
description states the 0-1000 scale, the tool takes `source` (a box can be drawn on an earlier crop), and two
sentences ask the agent to zoom in step by step. A tool result is a user turn
"<tool_response>observation_k:<image>...</tool_response>" that asks the question again; an error is "Error: ...".

The numbers (pixel budget, sizes, coverage, answer parsing) come from vira/common.py.
"""

from __future__ import annotations

import ast
import json
import math
import re

from vira import common as C

LET = "ABCDEFGH"
TOOL_NAME = "image_zoom_in_tool"
MAX_CALLS = 6                                  # zoom calls of a rollout; one more call ends it without an answer
CHAT_KWARGS = {"enable_thinking": False}       # chat-template options: the models run with thinking disabled

# ------------------------------------------------------------------------------------------------------------- prompts
_TOOL = {"type": "function", "function": {
    "name": TOOL_NAME,
    "description": ("Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an "
                    "optional object label. You can zoom in step by step: zoom in again on the latest observation "
                    "with a tighter box, until the detail you need is clearly visible. If the question needs several "
                    "parts of the image, zoom in on each of them in the original image."),
    "parameters": {"type": "object", "properties": {
        "bbox_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": (
            "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner "
            "and (x2, y2) is the bottom-right corner. Coordinates are relative to the image named by source, on a "
            "0-1000 scale.")},
        "source": {"type": "string", "description": (
            'The image to zoom in on: "original_image" (the first image, the default) or "observation_i", the i-th '
            "zoom-in image returned so far (observation_1 is the first). Zooming in on an observation looks closer "
            "at part of it; every crop is cut from the full-resolution original.")},
        "label": {"type": "string", "description": (
            "The name or label of the object in the specified bounding box (optional).")}},
        "required": ["bbox"]}}}          # "required" is kept from DeepEyes, which names bbox instead of bbox_2d
# the lines of DeepEyes' example end with two spaces (kept, so the prompt matches its prompt.py byte for byte)
SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n"
    "# Tools\n"
    "You may call one or more functions to assist with the user query.\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n" + json.dumps(_TOOL, separators=(",", ":")) + "\n</tools>\n\n"
    "# How to call a tool\n"
    "Return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "**Example**:  \n"
    "<tool_call>  \n"
    '{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  \n'
    "</tool_call>")
ANSWER_FORMAT = ("\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  "
                 "<think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> ")
OPEN_TAIL = "\nAnswer the question using a single word or phrase."
ZOOM_RESULT = ("\nThis is observation_{obs}, a zoom-in of your last box. If the detail you need is still small or "
               'unclear, zoom in again on it ("source": "observation_{obs}") with a tighter box; repeat until you can '
               "see it clearly. If the question also needs another part of the image, zoom in on that part of the "
               'original image ("source": "original_image"). When you have seen what you need, answer the question.'
               "\nQuestion: {question}")


def question_block(question: str, options=None) -> str:
    """The question as the agent reads it: with its lettered options, or, for an open question (no options), with
    the request for a short answer."""
    if options:
        return question.strip() + "\n" + "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(options))
    return question.strip() + OPEN_TAIL


def question_text(question: str, options=None) -> str:
    """The text of the first user turn (after the image): the question and the answer format."""
    return question_block(question, options) + ANSWER_FORMAT


def tool_result_parts(image_part, obs: int, question: str) -> list:
    """The user turn after a zoom: the crop, named observation_<obs>, a hint on how to go on, the question again
    (question_block) and the answer format."""
    return [{"type": "text", "text": f"<tool_response>observation_{obs}:"}, image_part,
            {"type": "text", "text": ZOOM_RESULT.format(obs=obs, question=question) + ANSWER_FORMAT + "</tool_response>"}]


def tool_error_text(msg: str) -> str:
    return msg if msg.startswith("Error") else f"Error: {msg}"


def tool_error_turn(msg: str) -> str:
    """The user turn after a failed call. It is wrapped in <tool_response>: the Qwen3 chat templates treat a bare user
    turn as a new question."""
    return f"<tool_response>{tool_error_text(msg)}</tool_response>"


def open_answer(text: str) -> str | None:
    """The text inside the last <answer>...</answer> (None if there is none or it is empty)."""
    if not C.has_answer(text or ""):
        return None
    a = text.split("<answer>")[-1].split("</answer>")[0].strip()
    return a or None


# ----------------------------------------------------------------------------------------------------------- actions
_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _literal(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)


def parse_action(text: str) -> dict:
    """What a reply asks for: an <answer> ends the rollout; otherwise the LAST <tool_call>, written as the JSON object
    {"name", "arguments"} of the prompt or in Qwen3's XML form <function=..><parameter=..>.
      {"type": "answer"} | {"type": "tool", "name": str, "args": dict} | {"type": "error", "msg": str} | {"type": "none"}"""
    text = text or ""
    if C.has_answer(text):
        return {"type": "answer"}
    if "<tool_call>" in text and "</tool_call>" not in text:
        text = text + "</tool_call>"           # generation cut right before the closing tag
    calls = _TOOL_RE.findall(text)
    if not calls:
        return {"type": "none"}
    body = calls[-1].strip()
    if body.startswith("{"):
        try:
            obj = _literal(body)
            name = obj.get("name") or obj.get("function")
            args = obj.get("arguments", obj.get("parameters", {k: v for k, v in obj.items() if k not in ("name", "function")}))
            if isinstance(args, str):
                args = _literal(args)
            if not isinstance(args, dict):
                raise ValueError("arguments is not an object")
            return {"type": "tool", "name": str(name), "args": args}
        except Exception as e:  # noqa: BLE001
            return {"type": "error", "msg": f"Invalid tool call format: {body[:200]}. Error: {e}"}
    m = _FUNC_RE.search(body)
    if m:
        args = {}
        for k, v in _PARAM_RE.findall(m.group(2)):
            try:
                args[k.strip()] = _literal(v)
            except Exception:
                args[k.strip()] = v.strip()
        return {"type": "tool", "name": m.group(1).strip(), "args": args}
    return {"type": "error", "msg": f"Invalid tool call format: {body[:200]}."}


def bbox_arg(args: dict):
    b = args.get("bbox_2d", args.get("bbox"))
    if isinstance(b, str):
        b = _literal(b)
    if isinstance(b, dict):
        b = b.get("bbox_2d") or b.get("bbox")
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        raise ValueError(f"bbox_2d must be a list of 4 numbers, got {b!r}")
    b = [float(v) for v in b]
    if not all(math.isfinite(v) for v in b):
        raise ValueError(f"bbox_2d has non-finite values {b!r}")
    return b


def source_arg(args: dict, n_obs: int) -> int:
    """The image a zoom box is drawn on: 0 = the first view ("original_image", the default), i = observation_i."""
    s = str(args.get("source") or "original_image").strip().lower()
    if s == "original_image":
        return 0
    m = re.fullmatch(r"observation_(\d+)", s)
    if not m:
        raise ValueError(f'source must be "original_image" or "observation_i", got {s!r}')
    i = int(m.group(1))
    if not 1 <= i <= n_obs:
        raise ValueError(f"{s} does not exist: {n_obs} observation(s) so far")
    return i


# ------------------------------------------------------------------------------------------------------------ zoom tool
def view_size(orig_size, x_pixels: int) -> tuple:
    """(w, h) of the first view: the photo under the pixel budget x."""
    return C.resized_size(int(orig_size[0]), int(orig_size[1]), min(int(x_pixels), C.MAX_PIXELS))


def _clamp(b, w, h):
    """A box inside a w x h image: sorted, clipped, each side at least one visual token (widened about its center)."""
    x0, y0, x1, y1 = b
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
    if x1 - x0 < C.PATCH:
        c = (x0 + x1) / 2
        x0, x1 = max(0.0, c - C.PATCH / 2), min(float(w), c + C.PATCH / 2)
    if y1 - y0 < C.PATCH:
        c = (y0 + y1) / 2
        y0, y1 = max(0.0, c - C.PATCH / 2), min(float(h), c + C.PATCH / 2)
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError(f"empty box {list(b)}")
    if max(x1 - x0, y1 - y0) / min(x1 - x0, y1 - y0) > 100:   # as DeepEyes' validate_bbox
        raise ValueError(f"aspect ratio of box {list(b)} is above 100")
    return [x0, y0, x1, y1]


def frame_box_to_orig(box, frame) -> list:
    """A box of the model (0-1000 scale) on an image it saw -> pixels of the photo. frame = (region, size): the region
    of the photo that the image shows and the size it was shown at; the first view is ((0, 0, W, H), view size),
    observation_i is (its crop box, its shown size). The box is scaled to pixels of that image, clamped there, mapped
    to the region and rounded outward."""
    (rx0, ry0, rx1, ry1), (fw, fh) = frame
    vb = _clamp([box[0] * fw / 1000, box[1] * fh / 1000, box[2] * fw / 1000, box[3] * fh / 1000], fw, fh)
    ob = C.view_to_orig(vb, (fw, fh), (rx1 - rx0, ry1 - ry0))
    return [int(math.floor(ob[0] + rx0)), int(math.floor(ob[1] + ry0)),
            int(math.ceil(ob[2] + rx0)), int(math.ceil(ob[3] + ry0))]


def crop_record(ob, x_pixels: int) -> dict:
    """The record of one zoom call: its box in the photo ("bbox"), the size the crop is shown at ("fed": enlarged to
    fill the budget by at most common.MAGNIFY per side, or shrunk to the budget) and its linear scale ("mag", shown /
    crop)."""
    cw, ch = ob[2] - ob[0], ob[3] - ob[1]
    fw, fh = C.magnified_size(cw, ch, min(int(x_pixels), C.MAX_PIXELS))
    mag = math.sqrt((fw * fh) / (cw * ch)) if cw > 0 and ch > 0 else 1.0
    return {"bbox": list(ob), "fed": [fw, fh], "mag": round(mag, 4)}


# ------------------------------------------------------------------------------------------------------------- reward
def crop_coverage(calls: list, object_box) -> dict:
    """Logged diagnostics of a rollout: the best and the last share of the object's box inside its crops."""
    cov = [C.coverage(c["bbox"], object_box) for c in calls if c.get("bbox")] if object_box is not None else []
    return {"cov_max": max(cov) if cov else 0.0, "cov_last": cov[-1] if cov else 0.0}


def episode_score(final_text: str, gold: str, calls: list, object_box) -> dict:
    """The reward of a rollout on a multiple-choice question: "acc", the correctness of the answer, is 1 if the option
    letter in the last <answer> is the correct one and 0 otherwise (also 0 without an answer)."""
    letter = C.parse_answer(final_text or "")
    acc = float(bool(letter) and letter == str(gold).strip().upper()[:1])
    return {"acc": acc, "answered": float(bool(letter)), "letter": letter, **crop_coverage(calls, object_box)}
