"""vira.protocol: prompts, tool calls, the box conversion and crop sizes of the zoom tool, the reward."""
import json

import pytest

from vira import common as C
from vira import protocol as P


def test_system_prompt():
    s = P.SYSTEM_PROMPT
    assert s.startswith("You are a helpful assistant.\n\n# Tools\n") and s.count("<tools>\n") == 1
    tool = json.loads(s.split("<tools>\n")[1].split("\n</tools>")[0])
    props = tool["function"]["parameters"]["properties"]
    assert tool["function"]["name"] == P.TOOL_NAME == "image_zoom_in_tool" and set(props) == {"bbox_2d", "source", "label"}
    assert "0-1000 scale" in props["bbox_2d"]["description"] and "observation_i" in props["source"]["description"]
    assert "zoom in step by step" in tool["function"]["description"]
    assert s.endswith('"label": "the apple on the desk"}}  \n</tool_call>')


def test_question_and_tool_result_text():
    q = P.question_text(" What color is the sign? ", ["red", "green"])
    assert q.startswith("What color is the sign?\nA. red\nB. green\nThink first, call **image_zoom_in_tool** if needed")
    assert P.question_text("What is written?").startswith("What is written?\nAnswer the question using a single word or phrase.\nThink")
    assert P.question_text("Q?", []) == P.question_text("Q?")                      # no options: an open question
    parts = P.tool_result_parts({"type": "image"}, 2, P.question_block("Q?", ["a", "b"]))
    assert parts[0] == {"type": "text", "text": "<tool_response>observation_2:"} and parts[1] == {"type": "image"}
    t = parts[2]["text"]
    assert t.startswith("\nThis is observation_2, a zoom-in of your last box.") and '("source": "observation_2")' in t
    assert "\nQuestion: Q?\nA. a\nB. b\nThink first" in t and t.endswith("<answer>...</answer> </tool_response>")
    assert P.tool_error_turn("bbox_2d must be a list") == "<tool_response>Error: bbox_2d must be a list</tool_response>"
    assert P.tool_error_turn("Error: x") == "<tool_response>Error: x</tool_response>"


def test_parse_action():
    call = '<think>t</think> <tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [1, 2, 3, 4]}}\n</tool_call>'
    assert P.parse_action(call) == {"type": "tool", "name": "image_zoom_in_tool", "args": {"bbox_2d": [1, 2, 3, 4]}}
    assert P.parse_action(call.replace("</tool_call>", ""))["type"] == "tool"          # cut before the closing tag
    assert P.parse_action(call + " <answer>B</answer>") == {"type": "answer"}          # an answer ends the rollout
    assert P.parse_action("just text") == {"type": "none"}
    xml = "<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n[5, 6, 7, 8]\n</parameter>\n" \
          "<parameter=source>\nobservation_1\n</parameter>\n</function>\n</tool_call>"
    assert P.parse_action(xml) == {"type": "tool", "name": "image_zoom_in_tool",
                                   "args": {"bbox_2d": [5, 6, 7, 8], "source": "observation_1"}}
    bad = P.parse_action("<tool_call>{oops</tool_call>")
    assert bad["type"] == "error" and bad["msg"].startswith("Invalid tool call format: {oops")
    two = call + call.replace("[1, 2, 3, 4]", "[9, 9, 20, 20]")
    assert P.parse_action(two)["args"]["bbox_2d"] == [9, 9, 20, 20]                    # the last call counts
    assert P.parse_action('<tool_call>{"name": "f", "arguments": "{\\"bbox_2d\\": [1, 2, 3, 4]}"}</tool_call>')["args"] \
        == {"bbox_2d": [1, 2, 3, 4]}


def test_bbox_and_source_arguments():
    assert P.bbox_arg({"bbox_2d": [1, 2, 3, 4]}) == [1.0, 2.0, 3.0, 4.0] and P.bbox_arg({"bbox": "[1, 2, 3, 4]"}) == [1.0, 2.0, 3.0, 4.0]
    for bad in ({}, {"bbox_2d": [1, 2, 3]}, {"bbox_2d": [1, 2, 3, float("nan")]}, {"bbox_2d": "x"}):
        with pytest.raises(Exception):
            P.bbox_arg(bad)
    assert P.source_arg({}, 0) == 0 and P.source_arg({"source": "original_image"}, 3) == 0
    assert P.source_arg({"source": "Observation_2"}, 2) == 2
    for bad in ({"source": "observation_3"}, {"source": "observation_0"}, {"source": "crop"}):
        with pytest.raises(ValueError):
            P.source_arg(bad, 2)


def test_box_conversion_and_crop_record():
    W, H, x = 6000, 4000, 1_000_000
    vsize = P.view_size((W, H), x)
    assert vsize == C.resized_size(W, H, x) == (1216, 800)
    first = ((0, 0, W, H), vsize)
    ob = P.frame_box_to_orig([400, 400, 500, 500], first)
    assert ob == [2400, 1600, 3000, 2000]
    rec = P.crop_record(ob, x)
    assert rec["bbox"] == ob and rec["fed"] == list(C.magnified_size(600, 400, x)) and 1.9 < rec["mag"] < 2.1
    # a box on observation_1 is mapped through that crop
    inner = P.frame_box_to_orig([250, 250, 750, 750], (tuple(ob), tuple(rec["fed"])))
    assert inner == [2550, 1700, 2850, 1900]
    # reversed corners are sorted; a box outside the image is clipped; a tiny box is widened to one visual token
    assert P.frame_box_to_orig([500, 500, 400, 400], first) == ob
    assert P.frame_box_to_orig([-50, -50, 100, 100], first) == [0, 0, 600, 400]
    tiny = P.frame_box_to_orig([500, 500, 501, 501], first)
    assert (tiny[2] - tiny[0]) * vsize[0] / W == pytest.approx(C.PATCH, abs=1.5)
    with pytest.raises(ValueError):
        P.frame_box_to_orig([0, 0, 1000, 1], ((0, 0, 60000, 300), (3200, 32)))      # aspect ratio above 100
    assert P.crop_record([0, 0, 4000, 3000], x)["mag"] < 1                           # above the budget: shrunk


def test_episode_score():
    box = [100, 100, 200, 200]
    calls = [{"bbox": [0, 0, 150, 200]}, {"error": "x"}, {"bbox": [90, 90, 210, 210]}]
    r = P.episode_score("<think>t</think><answer>B. blue</answer>", "B", calls, box)
    assert r == {"acc": 1.0, "answered": 1.0, "letter": "B", "cov_max": 1.0, "cov_last": 1.0}
    assert P.episode_score("<answer>A</answer>", "B", calls[:1], box) == \
        {"acc": 0.0, "answered": 1.0, "letter": "A", "cov_max": 0.5, "cov_last": 0.5}
    assert P.episode_score("The answer is B.", "B", [], box)["acc"] == 0.0           # no <answer>: reward 0
    assert P.episode_score("<answer>B</answer>", "B", calls, None)["cov_max"] == 0.0  # no box of the object
    assert P.open_answer("<answer> a red door </answer>") == "a red door" and P.open_answer("<answer></answer>") is None
