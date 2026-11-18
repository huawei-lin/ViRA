"""The zoom tool as a verl BaseTool. The state of the rollout is on agent_data.vira (set by the agent loop): the path
of the photo, the images shown so far, the pixel budget x and the list of calls. A call maps the model's box to pixels
of the photo (protocol.frame_box_to_orig), cuts the crop from the full-resolution photo and resizes it under x
(images.crop); every call is recorded."""

from __future__ import annotations

import asyncio
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .. import protocol as P
from ..images import crop as crop_image

# verl's registry entry; the model reads the tool in the system prompt (protocol.SYSTEM_PROMPT), not this schema
SCHEMA = {"type": "function", "function": {
    "name": P.TOOL_NAME, "description": "Zoom in on a region of an image.",
    "parameters": {"type": "object", "properties": {
        "bbox_2d": {"type": "array", "description": "[x1, y1, x2, y2]"},
        "source": {"type": "string", "description": "original_image or observation_i"},
        "label": {"type": "string", "description": "optional label"}}, "required": ["bbox_2d"]}}}


class ZoomTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        super().__init__(config, tool_schema or OpenAIFunctionToolSchema.model_validate(SCHEMA))

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        st = getattr(kwargs.get("agent_data"), "vira", None)
        if st is None:
            return ToolResponse(text=P.tool_error_text("zoom tool has no image context")), 0.0, {}
        st["n_calls"] += 1
        try:
            if "__error__" in parameters:
                raise ValueError(parameters["__error__"])
            box = P.bbox_arg(parameters)
            src = P.source_arg(parameters, len(st["frames"]) - 1)
            ob = P.frame_box_to_orig(box, st["frames"][src])
            loop = asyncio.get_running_loop()
            img, rec = await loop.run_in_executor(None, crop_image, st["path"], ob, st["x_pixels"])
        except Exception as e:  # noqa: BLE001 - the model's mistake: it is told "Error: ..." and may try again
            st["errors"] += 1
            st["calls"].append({"turn": st["n_calls"], "error": f"{type(e).__name__}: {e}"[:200]})
            return ToolResponse(text=P.tool_error_text(str(e))[:300]), 0.0, {}
        rec.update(turn=st["n_calls"], box_model=box, source=src)
        st["calls"].append(rec)
        st["frames"].append((tuple(ob), tuple(rec["fed"])))
        return ToolResponse(image=[img]), 0.0, {}
