from __future__ import annotations

from dataclasses import dataclass

import torch

from .cylinder_layout import CylinderObstacleLayout, CylinderObstacleLayoutCfg
from .maze_templates import FIXED_MAZE_TEMPLATES, get_fixed_maze_template


@dataclass
class FixedMazeLayoutCfg(CylinderObstacleLayoutCfg):
  """Runtime parameters for fixed serpentine maze layouts."""

  max_obstacles: int = 81
  cylinder_radius: float = 0.35
  soft_margin: float = 0.4
  max_template_level: int = len(FIXED_MAZE_TEMPLATES) - 1
  dense_random_min_active: int = 58
  dense_random_max_active: int = 81
  dense_random_min_separation: float = 1.45
  dense_random_half_extent: float = 8.0
  dense_random_margin: float = 0.5


class FixedMazeLayout(CylinderObstacleLayout):
  """Cylinder layout backed by fixed maze templates instead of random sampling."""

  def __init__(self, env, cfg: FixedMazeLayoutCfg):
    super().__init__(env, cfg)
    self.cfg: FixedMazeLayoutCfg = cfg
    self.template_level = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.entrance_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self.exit_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self.entrance_yaw = torch.zeros(self.num_envs, device=self.device)
    self.exit_yaw = torch.zeros(self.num_envs, device=self.device)

  def _load_dense_random_arena(
    self, env_ids: torch.Tensor, template_level: int, template
  ):
    """Sample a unique dense obstacle field for each environment."""
    low = -self.cfg.dense_random_half_extent + self.cfg.dense_random_margin
    high = self.cfg.dense_random_half_extent - self.cfg.dense_random_margin
    min_sep_sq = self.cfg.dense_random_min_separation**2

    self.centers_xy[env_ids] = 0.0
    self.active_mask[env_ids] = False

    for _row, env_id in enumerate(env_ids.tolist()):
      target_count = int(
        torch.randint(
          self.cfg.dense_random_min_active,
          self.cfg.dense_random_max_active + 1,
          (1,),
          device=self.device,
        ).item()
      )
      target_count = min(target_count, self.max_obstacles)

      placed = 0
      for _ in range(self.cfg.max_resample_tries * max(target_count, 1)):
        if placed >= target_count:
          break
        candidate = torch.empty(2, device=self.device).uniform_(low, high)
        if placed > 0:
          delta = self.centers_xy[env_id, :placed] - candidate
          if torch.any(torch.sum(delta.square(), dim=1) < min_sep_sq):
            continue
        self.centers_xy[env_id, placed] = candidate
        self.active_mask[env_id, placed] = True
        placed += 1

      self.template_level[env_id] = template_level
      self.num_active[env_id] = placed
      self.active_mask[env_id, placed:] = False
      self.entrance_xy[env_id] = torch.tensor(
        template.entrance_xy, dtype=torch.float, device=self.device
      )
      self.exit_xy[env_id] = torch.tensor(
        template.exit_xy, dtype=torch.float, device=self.device
      )
      self.entrance_yaw[env_id] = template.entrance_yaw
      self.exit_yaw[env_id] = template.exit_yaw

  def load_template(self, env_ids: torch.Tensor, template_level: torch.Tensor | int):
    """Load fixed wall centers and endpoints for selected environments."""
    if isinstance(template_level, int):
      template_level = torch.full(
        (len(env_ids),), template_level, dtype=torch.long, device=self.device
      )
    else:
      template_level = template_level.to(device=self.device, dtype=torch.long)

    levels = template_level.tolist()
    if all(get_fixed_maze_template(int(level)).randomize_walls for level in levels):
      level = int(levels[0])
      template = get_fixed_maze_template(level)
      self._load_dense_random_arena(env_ids, level, template)
      self.write_to_sim(env_ids)
      return

    self.centers_xy[env_ids] = 0.0
    self.active_mask[env_ids] = False

    for row, env_id in enumerate(env_ids.tolist()):
      level = int(
        torch.clamp(template_level[row], 0, self.cfg.max_template_level).item()
      )
      template = get_fixed_maze_template(level)
      if template.randomize_walls:
        self._load_dense_random_arena(
          torch.tensor([env_id], device=self.device), level, template
        )
        continue
      wall_centers = torch.tensor(
        template.wall_centers_xy, dtype=torch.float, device=self.device
      )
      wall_count = min(wall_centers.shape[0], self.max_obstacles)

      self.template_level[env_id] = level
      self.num_active[env_id] = wall_count
      self.centers_xy[env_id, :wall_count] = wall_centers[:wall_count]
      self.active_mask[env_id, :wall_count] = True
      self.entrance_xy[env_id] = torch.tensor(
        template.entrance_xy, dtype=torch.float, device=self.device
      )
      self.exit_xy[env_id] = torch.tensor(
        template.exit_xy, dtype=torch.float, device=self.device
      )
      self.entrance_yaw[env_id] = template.entrance_yaw
      self.exit_yaw[env_id] = template.exit_yaw

    self.write_to_sim(env_ids)

  def endpoint_xy_w(
    self, env_ids: torch.Tensor, target_is_exit: torch.Tensor
  ) -> torch.Tensor:
    """Return entrance/exit endpoints in world coordinates."""
    endpoint_xy = torch.where(
      target_is_exit[:, None], self.exit_xy[env_ids], self.entrance_xy[env_ids]
    )
    return endpoint_xy + self.env.scene.env_origins[env_ids, :2]

  def endpoint_yaw(
    self, env_ids: torch.Tensor, target_is_exit: torch.Tensor
  ) -> torch.Tensor:
    """Return preferred yaw for entrance/exit endpoints."""
    return torch.where(
      target_is_exit, self.exit_yaw[env_ids], self.entrance_yaw[env_ids]
    )


def get_fixed_maze_layout(env) -> FixedMazeLayout | None:
  """Return the active fixed maze layout if this environment defines one."""
  layout = getattr(env, "cylinder_layout", None)
  if isinstance(layout, FixedMazeLayout):
    return layout
  return None
