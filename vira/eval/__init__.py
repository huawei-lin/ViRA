"""Evaluation.

In training: arguments (``args``), answer parsing (``parsing``) and metrics (``metrics``); the evaluator itself is
``ViRAMixin.evaluate`` in ``vira/train/trainer.py``. Offline, a model behind vLLM: ``run`` (ViRA-Bench,
ViRA-Bench-Real and the general benchmarks under a pixel budget), the servers (``vlm``), tables (``summarize``).
"""
