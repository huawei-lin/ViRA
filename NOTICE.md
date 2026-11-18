# Third-party sources

This repository contains our own code. Some files adapt the prompts or rules of other projects. The header of each
file names its sources. No license text is copied here: for the license of each project, see the original repository.

| File | Source | License |
|---|---|---|
| `vira/protocol.py` | System and user prompts of DeepEyes (github.com/Visual-Agent/DeepEyes, `verl/workers/agent/envs/mm_process_engine/prompt.py`), with the additions described in the file | see the original repository |
| `vira/judge.py` | Judge prompts of Mini-o3 (`verl/utils/reward_score/general_qa_tool.py` in its repository) | see the original repository |
| `vira/common.py`, `vira/eval/vlm.py` | The resize rule follows `smart_resize` of `qwen_vl_utils` | see the original repository |
| `vira/train/` | Runs on verl, https://github.com/verl-project/verl at commit `174888f` (not included; `scripts/install_env.sh` clones it) | see the original repository |
