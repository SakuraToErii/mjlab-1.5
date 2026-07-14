from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CylinderObstacleLayoutCfg:
  obstacle_asset_name: str = "cylinder_obstacle"
  max_obstacles: int = 8
  cylinder_radius: float = 0.4
  cylinder_height: float = 2.0
  soft_margin: float = 0.6
  min_center_separation: float = 2.0
  arena_half_extent: float = 28.0
  arena_margin: float = 3.0
  max_resample_tries: int = 128
  exclude_origin: bool = True
  hidden_local_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)


class CylinderObstacleLayout:
  def __init__(self, env, cfg: CylinderObstacleLayoutCfg):
    self.env, self.cfg, self.device, self.num_envs = env, cfg, env.device, env.num_envs
    self.max_obstacles = cfg.max_obstacles
    self.centers_xy = torch.zeros(
      self.num_envs, cfg.max_obstacles, 2, device=self.device
    )
    self.active_mask = torch.zeros(
      self.num_envs, cfg.max_obstacles, dtype=torch.bool, device=self.device
    )
    self.num_active = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.violation_rate = torch.zeros(self.num_envs, device=self.device)

  @property
  def clearance_radius(self):
    return self.cfg.cylinder_radius + self.cfg.soft_margin

  def sample_layout(self, env_ids, num_active):
    if isinstance(num_active, int):
      num_active = torch.full(
        (len(env_ids),), num_active, dtype=torch.long, device=self.device
      )
    self.centers_xy[env_ids] = 0
    self.active_mask[env_ids] = False
    low, high = (
      -self.cfg.arena_half_extent + self.cfg.arena_margin,
      self.cfg.arena_half_extent - self.cfg.arena_margin,
    )
    for row, env_id in enumerate(env_ids.tolist()):
      target, placed = min(int(num_active[row]), self.max_obstacles), 0
      for _ in range(self.cfg.max_resample_tries * max(target, 1)):
        if placed >= target:
          break
        candidate = torch.empty(2, device=self.device).uniform_(low, high)
        if (
          self.cfg.exclude_origin
          and candidate.square().sum() < (self.clearance_radius + 0.8) ** 2
        ):
          continue
        if placed and torch.any(
          (self.centers_xy[env_id, :placed] - candidate).square().sum(dim=-1)
          < self.cfg.min_center_separation**2
        ):
          continue
        self.centers_xy[env_id, placed] = candidate
        self.active_mask[env_id, placed] = True
        placed += 1
      self.num_active[env_id] = placed

  def write_to_sim(self, env_ids):
    origins = self.env.scene.env_origins[env_ids]
    for slot in range(self.max_obstacles):
      pose = torch.zeros(len(env_ids), 7, device=self.device)
      pose[:, 3] = 1
      pose[:, :3] = origins + torch.tensor(
        self.cfg.hidden_local_pos, device=self.device
      )
      active = self.active_mask[env_ids, slot]
      pose[active, :2] = origins[active, :2] + self.centers_xy[env_ids[active], slot]
      pose[active, 2] = self.cfg.cylinder_height * 0.5
      self.env.scene[
        f"{self.cfg.obstacle_asset_name}_{slot:03d}"
      ].write_mocap_pose_to_sim(pose, env_ids)

  def is_pose_free(self, xy_w, radius, env_ids):
    local = xy_w - self.env.scene.env_origins[env_ids, :2]
    dist = torch.norm(local[:, None] - self.centers_xy[env_ids], dim=-1)
    return ~torch.any(
      (dist < self.clearance_radius + radius) & self.active_mask[env_ids], dim=-1
    )

  def is_disk_free(self, xy_w, radius, env_ids):
    return self.is_pose_free(xy_w, radius, env_ids)

  def soft_proximity_penalty(self, xy_w):
    local = xy_w - self.env.scene.env_origins[:, :2]
    distance = (
      torch.norm(local[:, None] - self.centers_xy, dim=-1) - self.cfg.cylinder_radius
    )
    penalty = (
      torch.clamp(1 - distance / self.cfg.soft_margin, 0, 1).square()
      * self.active_mask.float()
    )
    result = penalty.max(dim=1).values
    self.violation_rate[:] = (result > 0).float()
    return result

  def draw_debug_vis(self, visualizer) -> None:
    """Draw active cylinder soft zones for the selected worlds."""
    for env_id in visualizer.get_env_indices(self.num_envs):
      origin = self.env.scene.env_origins[env_id].detach().cpu().numpy()
      for slot in self.active_mask[env_id].nonzero(as_tuple=False).flatten().tolist():
        center = origin.copy()
        center[:2] += self.centers_xy[env_id, slot].detach().cpu().numpy()
        center[2] = 0.01
        visualizer.add_cylinder(
          center,
          center + np.array([0.0, 0.0, 0.025]),
          self.clearance_radius,
          (1.0, 0.25, 0.15, 0.25),
          label="obstacle soft zone",
        )
