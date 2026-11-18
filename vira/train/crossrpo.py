"""The CrossRPO advantage as a drop-in for verl's ``compute_advantage_for_multi_trajectories``.

The trainer (``trainer.ViRAMixin._compute_advantage``) swaps this in for the call that computes verl's advantages, so
everything around it in verl stays as is (rollout correction, writing the nested ``advantages`` / ``returns`` back to
TransferQueue). Only the final output of each ``{uid}_{session_id}`` session (one rollout) is scored, and its
advantage is assigned to every generated token of that rollout. The per-rollout values come from
:func:`vira.common.crossrpo_advantages`; they are divided by their root mean square over the scored rollouts with a
nonempty response (the batch scale): one positive number per step, so it changes the length of the update but not its
direction.
"""

from __future__ import annotations

import math

from vira import common as C


def make_crossrpo_fn(rows: list, lam: float = C.CROSSRPO_LAMBDA, metrics: dict | None = None):
    """rows[i] = (set key, version) of batch row i (``metrics.rollout_rows``) -> a function with the signature of
    verl's ``compute_advantage_for_multi_trajectories(data, batch_keys, ...)``."""

    def compute(data, batch_keys, **_ignored):
        import torch

        final: dict = {}
        row_session = []
        for i, key in enumerate(batch_keys):
            uid, session_id, index = str(key).rsplit("_", 2)
            sk = f"{uid}_{session_id}"
            row_session.append(sk)
            if sk not in final or final[sk][0] < int(index):
                final[sk] = (int(index), i)
        mask = data.batch["response_mask"]
        rewards = (data.batch["token_level_rewards"] * mask).sum(dim=-1)      # the correctness of each rollout
        sessions = list(final)
        fi = [final[s][1] for s in sessions]
        adv = C.crossrpo_advantages([rows[i][0] for i in fi], [rows[i][1] for i in fi],
                                    [float(rewards[i]) for i in fi], lam)
        live = [float(mask[i].sum()) > 0 for i in fi]
        sq = [a * a for a, ok in zip(adv, live) if ok]
        rms = math.sqrt(sum(sq) / len(sq)) if sq else 1.0
        adv = [a / (rms + 1e-6) for a in adv]
        by_session = dict(zip(sessions, adv))
        per_row = torch.tensor([by_session[s] for s in row_session], dtype=mask.dtype if mask.is_floating_point()
                               else torch.float32, device=mask.device)
        data.batch["advantages"] = data.batch["returns"] = per_row.unsqueeze(-1) * mask
        if metrics is not None:
            metrics["train/crossrpo_rms"] = rms
            if live:                  # share of the scored rollouts whose advantage is not zero
                metrics["train/crossrpo_nonzero"] = sum(abs(a) > 1e-9 for a, ok in zip(adv, live) if ok) / max(1, sum(live))
        return data

    return compute
