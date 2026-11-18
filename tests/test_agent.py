"""vira.agent (one rollout over a chat API) and vira.infer, with scripted replies."""
import pytest
from PIL import Image

from vira import agent as A
from vira import infer
from vira import protocol as P

CALL = '<think>look</think>\n<tool_call>\n{"name": "image_zoom_in_tool", "arguments": %s}\n'


class Client:
    """An OpenAI client that replays `replies` and records the requests."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = self.completions = self

    def create(self, **kw):
        self.requests.append(dict(kw, messages=list(kw["messages"])))
        msg = type("M", (), {"content": self.replies[len(self.requests) - 1]})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


@pytest.fixture
def photo(tmp_path):
    p = str(tmp_path / "photo.png")
    Image.new("RGB", (4000, 3000), "white").save(p)
    return p


def test_rollout_zooms_errors_and_answer(photo):
    client = Client([CALL % '{"bbox_2d": [400, 400, 500, 500]}',
                     CALL % '{"bbox_2d": [250, 250, 750, 750], "source": "observation_1"}',
                     CALL % '{"bbox_2d": [0, 0, 10, 10], "source": "observation_9"}',
                     "<think>ok</think><answer>B. x</answer>"])
    out = infer.answer_question(client, "m", photo, "Q?", ["w", "x", "y", "z"], budget="1")
    assert out["answer"] == "B. x" and out["letter"] == "B" and len(out["replies"]) == 4
    assert out["zooms"] == [[1600, 1200, 2000, 1500], [1700, 1275, 1900, 1425]]      # the second box is on the first crop
    assert all(r.endswith("</tool_call>") for r in out["replies"][:3])                # the stop string is put back
    msgs = client.requests[-1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
    assert msgs[0]["content"] == P.SYSTEM_PROMPT
    assert msgs[1]["content"][1]["text"] == P.question_text("Q?", ["w", "x", "y", "z"])
    assert msgs[2] == {"role": "assistant", "content": out["replies"][0], "reasoning_content": ""}
    assert msgs[3]["content"][0]["text"] == "<tool_response>observation_1:"
    assert msgs[5]["content"][0]["text"] == "<tool_response>observation_2:"
    assert "Question: Q?\nA. w" in msgs[3]["content"][2]["text"]
    assert msgs[7]["content"][0]["text"] == "<tool_response>Error: observation_9 does not exist: 2 observation(s) so far</tool_response>"
    req = client.requests[0]
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert req["temperature"] == 0 and req["max_tokens"] == 4096 and req["stop"] == ["</tool_call>"]


def test_rollout_records_and_status(photo):
    replies = iter([CALL % '{"bbox_2d": [100, 100, 190, 220]}', CALL % '{"bbox_2d": [1, 2]}', "no tags"])
    shown = []
    out = A.rollout(lambda messages: next(replies), "Q?", ["a", "b"], photo, (4000, 3000), 4_014_080, {"type": "first"},
                    (2304, 1728), lambda im, k: shown.append((k, im.size)) or {"type": "crop", "k": k})
    assert out["status"] == "no_answer_no_tool" and len(out["replies"]) == 3
    z, err = out["zooms"]
    assert z["turn"] == 0 and z["source"] == 0 and z["box_model"] == [100.0, 100.0, 190.0, 220.0]
    assert z["bbox"] == P.frame_box_to_orig([100, 100, 190, 220], ((0, 0, 4000, 3000), (2304, 1728)))
    assert z["bbox"][:3] == [400, 300, 760] and 3.9 < z["mag"] <= 4.0                 # enlarged by at most 4 x
    assert shown == [(1, tuple(z["fed"]))] and z["fed"][0] * z["fed"][1] <= 4_014_080
    assert err["turn"] == 1 and err["error"].startswith("ValueError: bbox_2d must be a list of 4 numbers")


def test_open_question_and_call_cap(photo):
    client = Client(["<think>t</think><answer>a red door</answer>"])
    out = infer.answer_question(client, "m", photo, "What?", budget="0.5")
    assert out["answer"] == "a red door" and out["letter"] is None and out["zooms"] == []
    assert client.requests[0]["messages"][1]["content"][1]["text"] == P.question_text("What?")
    # six zooms are allowed; a seventh call ends the rollout without an answer
    out = infer.answer_question(Client([CALL % '{"bbox_2d": [0, 0, 500, 500]}'] * 7), "m", photo, "Q?", ["a", "b"])
    assert out["answer"] is None and len(out["replies"]) == 7 and len(out["zooms"]) == 6
