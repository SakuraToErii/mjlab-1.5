from mjlab.tasks.eff_velocity.rl import EffortVelocityOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_mha_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_rough_mha_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_mha_runner_cfg, unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Effort-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Effort-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Effort-Rough-Unitree-G1-MHA",
  env_cfg=unitree_g1_rough_mha_env_cfg(),
  play_env_cfg=unitree_g1_rough_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Effort-Flat-Unitree-G1-MHA",
  env_cfg=unitree_g1_flat_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortVelocityOnPolicyRunner,
)
