from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.navigation_loco.navigation_loco_env_cfg import make_low_level_env_cfg


def unitree_g1_low_level_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_low_level_env_cfg(sequential=True, play=play)


def unitree_g1_low_level_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_low_level_env_cfg(sequential=False, play=play)
