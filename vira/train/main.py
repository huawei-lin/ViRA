"""Entry point: verl's main_ppo with a task runner that registers the ViRA trainer first.

    python -m vira.train.main <hydra overrides>   (trainer.v1.trainer_mode=vira_sync)

The runner is verl.trainer.main_ppo.TaskRunnerV1 with one change: it imports vira.train.trainer inside the
Ray actor, so `vira_sync` is in the trainer registry there. The repository root (the `vira` package) must be on the
Ray workers' PYTHONPATH.
"""

from __future__ import annotations

import os
from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

import verl.trainer.main_ppo as mp

CONFIG_DIR = os.path.join(os.path.dirname(mp.__file__), "config")


@ray.remote
class ViRATaskRunner:
    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        self.agent_loop_manager = AgentLoopManagerTQ.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        mp.configure_verl_logging()
        import transfer_queue as tq

        import vira.train.trainer  # noqa: F401 - registers vira_sync
        from verl.trainer.ppo.v1 import get_trainer_cls

        trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config
        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path=CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    mp.auto_set_device(config)
    mp.validate_config(config=config, use_reference_policy=mp.need_reference_policy(config),
                       use_critic=mp.need_critic(config))
    assert config.trainer.use_v1, "ViRA runs on the v1 trainer"
    mp.run_ppo(config, task_runner_class=ViRATaskRunner)


if __name__ == "__main__":
    main()
