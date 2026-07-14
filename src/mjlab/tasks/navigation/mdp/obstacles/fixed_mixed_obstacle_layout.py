from __future__ import annotations

import torch

from .mixed_arena_templates import (
  get_fixed_mixed_arena_template,
  num_fixed_mixed_arena_templates,
)
from .mixed_obstacle_layout import MixedObstacleLayout, MixedObstacleLayoutCfg


class FixedMixedObstacleLayout(MixedObstacleLayout):
  """Mixed obstacle layout with per-env maps baked once from fixed arena templates."""

  def __init__(self, env, cfg: MixedObstacleLayoutCfg):
    super().__init__(env, cfg)
    self.template_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

  def apply_baked_template(self, env_ids: torch.Tensor, template_ids: torch.Tensor):
    """Copy a pre-baked template into per-environment layout buffers."""
    template_ids = template_ids.to(device=self.device, dtype=torch.long)
    for row, env_id in enumerate(env_ids.tolist()):
      level = int(template_ids[row].item()) % num_fixed_mixed_arena_templates()
      baked = get_fixed_mixed_arena_template(level, layout_cfg=self.cfg)
      placed = baked.num_active
      self.template_id[env_id] = level
      self.centers_xy[env_id] = 0.0
      self.active_mask[env_id] = False
      self.active_slot_ids[env_id] = -1
      if placed > 0:
        self.centers_xy[env_id, :placed] = baked.centers_xy[:placed].to(self.device)
        self.active_slot_ids[env_id, :placed] = baked.active_slot_ids[:placed].to(
          self.device
        )
        self.active_mask[env_id, :placed] = True
      self.num_active[env_id] = placed

  def load_fixed_templates(self, env_ids: torch.Tensor | None):
    """Assign each environment a sticky template and write obstacles to sim."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    elif not isinstance(env_ids, torch.Tensor):
      env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    preset = getattr(self.env, "arena_template_id", None)
    if preset is not None:
      template_ids = preset[env_ids]
    else:
      template_ids = env_ids % num_fixed_mixed_arena_templates()
    self.apply_baked_template(env_ids, template_ids)
    self.write_to_sim(env_ids)


def get_fixed_mixed_obstacle_layout(env) -> FixedMixedObstacleLayout | None:
  layout = getattr(env, "obstacle_layout", None)
  if isinstance(layout, FixedMixedObstacleLayout):
    return layout
  return None
