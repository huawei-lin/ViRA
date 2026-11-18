"""Ask ViRA one question about one image, with the prompts and the zoom tool of training and evaluation.

Serve the model with vLLM (an OpenAI-compatible server), for example

    vllm serve huaweilin/ViRA-9B --served-model-name ViRA-9B --max-model-len 128000 \\
        --limit-mm-per-prompt '{"image": 10}' --generation-config vllm

and ask:

    python -m vira.infer --image photo.jpg --question "What does the small sign above the door show?" \\
        --options "An arrow pointing up" "An arrow pointing left" "An arrow pointing right" "A cross" --budget 1

Without --options the question is open and the answer is a word or a phrase. --budget is the pixel budget in MP of
the first view and of every crop (0.5, 1, 2, 4, or max = 16.8 MP). In Python:

    from openai import OpenAI
    from vira.infer import answer_question
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    out = answer_question(client, "ViRA-9B", "photo.jpg", "What ...?", ["...", "...", "...", "..."], budget="1")
    out["answer"], out["letter"], out["zooms"]

The agent sees the image resized to the budget. It may call the zoom tool up to six times; each crop is cut from the
full-resolution image and enlarged by up to 4 times within the budget. The answer is the last <answer> span.
"""

from __future__ import annotations

import argparse
import base64
import io
import json

from . import agent as A
from . import common as C
from . import images as I
from . import protocol as P


def image_part(im) -> dict:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def answer_question(client, model: str, image_path: str, question: str, options=None, budget: str = "max") -> dict:
    """One rollout with greedy decoding. Returns {"answer": the text of the last <answer> span (None without an
    answer), "letter": its option letter (multiple choice), "zooms": the boxes in pixels of the image, "replies"}."""
    x = C.BUDGETS[str(budget)]
    view = I.first_view(image_path, x)

    def chat(messages):
        r = client.chat.completions.create(model=model, messages=messages, temperature=0, **A.GENERATION)
        return r.choices[0].message.content or ""

    out = A.rollout(chat, question, options, image_path, I.load_original(image_path).size, x, image_part(view),
                    view.size, lambda im, k: image_part(im))
    answer = P.open_answer(out["replies"][-1]) if out["status"] == "ok" else None
    return {"answer": answer, "letter": C.parse_answer(out["replies"][-1]) if options and answer else None,
            "zooms": [z["bbox"] for z in out["zooms"] if "bbox" in z], "replies": out["replies"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--options", nargs="*", default=None, help="the options of a multiple-choice question")
    ap.add_argument("--budget", default="max", choices=list(C.BUDGETS), help="pixel budget in MP")
    ap.add_argument("--model", default="ViRA-9B", help="the served model name")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--verbose", action="store_true", help="print every reply of the agent")
    a = ap.parse_args()
    from openai import OpenAI

    out = answer_question(OpenAI(base_url=a.url, api_key=a.api_key), a.model, a.image, a.question, a.options, a.budget)
    if a.verbose:
        for i, r in enumerate(out["replies"]):
            print(f"--- reply {i + 1} ---\n{r}")
    print(json.dumps({k: out[k] for k in ("answer", "letter", "zooms")}))


if __name__ == "__main__":
    main()
