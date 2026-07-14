from __future__ import annotations

import torch

from ..obstacles import MixedObstacleLayoutCfg
from ..obstacles.fixed_mixed_obstacle_layout import FixedMixedObstacleLayout


def assign_fixed_mixed_arena_layout(env, env_ids, layout_cfg: MixedObstacleLayoutCfg):
  env_ids = (
    torch.arange(env.num_envs, device=env.device)
    if env_ids is None or isinstance(env_ids, slice)
    else env_ids
  )
  layout = getattr(env, "obstacle_layout", None)
  if not isinstance(layout, FixedMixedObstacleLayout):
    layout = FixedMixedObstacleLayout(env, layout_cfg)
    env.obstacle_layout = layout
  layout.load_fixed_templates(env_ids)
  if not hasattr(env, "obstacle_num_active"):
    env.obstacle_num_active = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
  env.obstacle_num_active[env_ids] = layout.num_active[env_ids]
