"""verl ToolParser "vira": reads the tool call of a reply with vira.protocol.parse_action, the function that the
evaluation uses. A reply with an <answer> has no tool call (the rollout ends). A malformed call becomes a call with
"__error__", so the zoom tool answers "Error: ..." and the agent may try again."""

from __future__ import annotations

import json

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.ray_utils import get_event_loop

from .. import protocol as P


@ToolParser.register("vira")
class ViRAToolParser(ToolParser):
    _stop_ids: list = []

    @property
    def stop_token_ids(self) -> list:
        """Set by the agent loop: generation stops right after </tool_call>."""
        return list(self._stop_ids)

    async def extract_tool_calls(self, responses_ids, tools=None):
        loop = get_event_loop()
        text = await loop.run_in_executor(None, lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=True))
        return text, self.calls_from_text(text)

    @staticmethod
    def calls_from_text(text: str) -> list:
        act = P.parse_action(text)
        if act["type"] == "tool":
            if act["name"] != P.TOOL_NAME:
                return [FunctionCall(name=P.TOOL_NAME, arguments=json.dumps({"__error__": f"Unknown tool name: {act['name']}"}))]
            return [FunctionCall(name=P.TOOL_NAME, arguments=json.dumps(act["args"], ensure_ascii=False, default=str))]
        if act["type"] == "error":
            return [FunctionCall(name=P.TOOL_NAME, arguments=json.dumps({"__error__": act["msg"]}))]
        return []
