from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_navigation_baseline_env_cfg,
  unitree_g1_navigation_extension_env_cfg,
)
from .rl_cfg import navigation_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-29dof-Navigation-HRL-Extension",
  env_cfg=unitree_g1_navigation_extension_env_cfg(),
  play_env_cfg=unitree_g1_navigation_extension_env_cfg(play=True),
  # Source registration intentionally uses the compact-single-goal runner wiring.
  rl_cfg=navigation_ppo_runner_cfg(baseline=True),
)

register_mjlab_task(
  task_id="Unitree-G1-29dof-Navigation-HRL-Baseline",
  env_cfg=unitree_g1_navigation_baseline_env_cfg(),
  play_env_cfg=unitree_g1_navigation_baseline_env_cfg(play=True),
  rl_cfg=navigation_ppo_runner_cfg(baseline=True),
)
