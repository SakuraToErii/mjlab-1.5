from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.navigation.navigation_env_cfg import (
  make_navigation_v5_compact_single_goal_env_cfg,
  make_navigation_v5_mixed_obstacle_env_cfg,
)


def unitree_g1_navigation_extension_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_navigation_v5_mixed_obstacle_env_cfg(play=play)


def unitree_g1_navigation_baseline_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_navigation_v5_compact_single_goal_env_cfg(play=play)
