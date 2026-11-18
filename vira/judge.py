"""Scoring of open answers (rows with ``extra_info.open``: questions on single images without options), as Mini-o3
does.

judge(question, gold, pred, key_file) -> (acc, failed): every answer goes to GPT-4o (gpt-4o-2024-11-20 on OpenRouter)
with Mini-o3's judge prompts verbatim (Mini-o3 verl/utils/reward_score/general_qa_tool.py: SYSTEM_PROMPT +
QUERY_PROMPT, text only, temperature 0, retried at 0.2 x attempt as Mini-o3 does); acc = 1 if the reply says
"Score: 1". If every attempt fails, acc = 0 and failed = 1 (logged as train/judge_failed). The agent loop calls it only for
a non-empty <answer>. The API key is read from ``key_file`` if one is given, else from OPENROUTER_API_KEY.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request

MODEL = "openai/gpt-4o-2024-11-20"
URL = "https://openrouter.ai/api/v1/chat/completions"
ATTEMPTS, TIMEOUT = 5, 45

# Mini-o3's judge prompts, verbatim
SYSTEM_PROMPT = "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\nYour task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n- Focus on the meaningful match between the predicted answer and the correct answer.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate the correctness of the prediction compared to the answer."
QUERY_PROMPT = 'I will give you a question related to an image and the following text as inputs:\n\n1. **Question Related to the Image**: {question}\n2. **Ground Truth Answer**: {ground_truth}\n3. **Model Predicted Answer**: {prediction}\n\nYour task is to evaluate the model\'s predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:\n- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?\n- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:\n(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.\n(2) If the ground truth answer is a definitive answer, strictly compare the model\'s prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model\'s prediction should be deemed correct.\n**Output Format**:\nYour response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means the model\'s prediction strictly aligns with the ground truth, while 0 means it does not.\nThe format should be "Score: 0 or 1"'

_KEY: dict = {}
_LOCK = threading.Lock()


def _key(path: str) -> str:
    if not path:
        return os.environ["OPENROUTER_API_KEY"]
    with _LOCK:
        if path not in _KEY:
            with open(os.path.expanduser(path)) as f:
                _KEY[path] = f.read().strip()
        return _KEY[path]


def score_reply(text: str) -> int:
    m = re.search(r"Score\W*\s*([01])", text or "")
    return int(m.group(1)) if m else int("1" in (text or ""))


def gpt4o(question: str, gold: str, pred: str, key_file: str) -> tuple[int, bool]:
    """(score, ok): Mini-o3's GPT-4o judge on OpenRouter; ok False when every attempt failed."""
    body = {"model": MODEL, "max_tokens": 256, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUERY_PROMPT.format(question=question, ground_truth=gold, prediction=pred)}]}
    hdr = {"Authorization": f"Bearer {_key(key_file)}", "Content-Type": "application/json"}
    for attempt in range(ATTEMPTS):
        body["temperature"] = min(0.2 * attempt, 1.0)
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), hdr)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                txt = json.load(r)["choices"][0]["message"]["content"] or ""
            if txt.strip():
                return score_reply(txt), True
        except Exception:  # noqa: BLE001 - network / rate limit / bad reply: retry
            pass
        time.sleep(2.0 * (attempt + 1))
    return 0, False


def judge(question: str, gold: str, pred: str, key_file: str = "") -> tuple[float, float]:
    """(acc, failed) of one open answer: GPT-4o's verdict (see the module docstring)."""
    s, ok = gpt4o(question, gold, str(pred).strip()[:300], key_file)
    return float(s), float(not ok)
