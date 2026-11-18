"""One rollout of the zoom agent over an OpenAI-compatible chat API (a vLLM server): the loop that the offline
evaluation (vira.eval.run) and inference (vira.infer) share. Training runs the same protocol inside verl
(vira.train.agent_loop).

The agent sees the image resized to the pixel budget x and replies with a zoom call or an answer. A zoom call is
answered with the crop, cut from the full-resolution image and shown under the same budget, a failed call with
"Error: ...". After protocol.MAX_CALLS calls the next reply is final: one more call ends the rollout without an
answer.
"""

from __future__ import annotations

from . import images as I
from . import protocol as P

STOP = "</tool_call>"
# every reply: at most 4,096 new tokens, generation stops at </tool_call> (a written call is executed, not followed
# by an answer in the same reply), thinking disabled in the chat template
GENERATION = {"max_tokens": 4096, "stop": [STOP], "extra_body": {"chat_template_kwargs": dict(P.CHAT_KWARGS)}}


def rollout(chat, question: str, options, image_path: str, photo_size, x_pixels: int, first_part, first_size,
            crop_part) -> dict:
    """chat(messages) -> the text of the next reply (a chat completion with GENERATION).
    question, options: options = None or [] for an open question.
    image_path, photo_size: the full-resolution image and its (W, H); x_pixels: the pixel budget.
    first_part, first_size: the content part of the first view (the image resized to the budget) and its (w, h).
    crop_part(image, k) -> the content part of the k-th crop (a PIL image).
    Returns {"replies": the replies in order, "zooms": the record of every call (protocol.crop_record + turn,
    box_model = the box as the model wrote it, source; a failed call is {"turn", "error"}), "status": "ok" (the last
    reply has an answer), "no_answer_no_tool", or "cap" (a call after the last allowed one)}."""
    asked = P.question_block(question, options)
    messages = [{"role": "system", "content": P.SYSTEM_PROMPT},
                {"role": "user", "content": [first_part, {"type": "text", "text": P.question_text(question, options)}]}]
    frames = [((0, 0, int(photo_size[0]), int(photo_size[1])), tuple(first_size))]   # images shown: region, size
    replies, zooms, status = [], [], "ok"
    for turn in range(P.MAX_CALLS + 1):
        text = chat(messages)
        if "<tool_call>" in text and STOP not in text.split("<tool_call>")[-1]:
            text += STOP                               # the server does not return the stop string
        replies.append(text)
        act = P.parse_action(text)
        if act["type"] == "answer":
            break
        if act["type"] == "none":
            status = "no_answer_no_tool"
            break
        if turn == P.MAX_CALLS:
            status = "cap"
            break
        # reasoning_content: the reply stays as generated (the chat template would move its <think> block)
        messages.append({"role": "assistant", "content": text, "reasoning_content": ""})
        try:
            if act["type"] == "error":
                raise ValueError(act["msg"])
            if act["name"] != P.TOOL_NAME:
                raise ValueError(f"Unknown tool name: {act['name']}")
            box = P.bbox_arg(act["args"])
            src = P.source_arg(act["args"], len(frames) - 1)
            ob = P.frame_box_to_orig(box, frames[src])
            crop, rec = I.crop(image_path, ob, x_pixels)
        except Exception as e:  # noqa: BLE001 - the model's mistake: it is told "Error: ..." and may try again
            zooms.append({"turn": turn, "error": f"{type(e).__name__}: {e}"[:300]})
            messages.append({"role": "user", "content": [{"type": "text", "text": P.tool_error_turn(str(e))}]})
            continue
        rec.update(turn=turn, box_model=box, source=src)
        zooms.append(rec)
        frames.append((tuple(ob), tuple(rec["fed"])))
        k = len(frames) - 1
        messages.append({"role": "user", "content": P.tool_result_parts(crop_part(crop, k), k, asked)})
    return {"replies": replies, "zooms": zooms, "status": status}
