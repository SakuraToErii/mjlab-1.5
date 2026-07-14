from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

from .mixed_obstacle_collection import (
  V5_BOX_LOW_SIZE,
  V5_BOX_TALL_SIZE,
  V5_CYL_RADIUS_LARGE,
  V5_CYL_RADIUS_MEDIUM,
  V5_CYL_RADIUS_SMALL,
  V5_CYLINDER_HEIGHT,
  V5_MAX_MIXED_OBSTACLES,
  V5_NUM_BOX_LOW,
  V5_NUM_BOX_TALL,
  V5_NUM_CYL_LARGE,
  V5_NUM_CYL_MEDIUM,
  V5_NUM_CYL_SMALL,
)


class ObstacleSlotType(IntEnum):
  CYLINDER = 0
  BOX_LOW = 1
  BOX_TALL = 2


@dataclass
class MixedObstacleLayoutCfg:
  obstacle_asset_name: str = "mixed_obstacle"
  max_obstacles: int = V5_MAX_MIXED_OBSTACLES
  soft_margin: float = 0.4
  min_center_separation: float = 1.1
  arena_half_extent: float = 28.0
  arena_margin: float = 3.0
  max_resample_tries: int = 256
  exclude_origin: bool = False
  hidden_local_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)


def _build_slot_metadata(
  device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  specs: list[tuple[int, float, float, float, float]] = []
  specs += [
    (ObstacleSlotType.CYLINDER, V5_CYL_RADIUS_SMALL, 0.0, 0.0, V5_CYLINDER_HEIGHT)
  ] * V5_NUM_CYL_SMALL
  specs += [
    (ObstacleSlotType.CYLINDER, V5_CYL_RADIUS_MEDIUM, 0.0, 0.0, V5_CYLINDER_HEIGHT)
  ] * V5_NUM_CYL_MEDIUM
  specs += [
    (ObstacleSlotType.CYLINDER, V5_CYL_RADIUS_LARGE, 0.0, 0.0, V5_CYLINDER_HEIGHT)
  ] * V5_NUM_CYL_LARGE
  specs += [
    (
      ObstacleSlotType.BOX_LOW,
      max(V5_BOX_LOW_SIZE[:2]) * 0.5,
      V5_BOX_LOW_SIZE[0] * 0.5,
      V5_BOX_LOW_SIZE[1] * 0.5,
      V5_BOX_LOW_SIZE[2],
    )
  ] * V5_NUM_BOX_LOW
  specs += [
    (
      ObstacleSlotType.BOX_TALL,
      max(V5_BOX_TALL_SIZE[:2]) * 0.5,
      V5_BOX_TALL_SIZE[0] * 0.5,
      V5_BOX_TALL_SIZE[1] * 0.5,
      V5_BOX_TALL_SIZE[2],
    )
  ] * V5_NUM_BOX_TALL
  return (
    torch.tensor([s[0] for s in specs], dtype=torch.long, device=device),
    torch.tensor([s[1] for s in specs], device=device),
    torch.tensor([[s[2], s[3]] for s in specs], device=device),
    torch.tensor([s[4] for s in specs], device=device),
  )


class MixedObstacleLayout:
  """Per-world pose layout over a shared fixed 120-slot model topology."""

  def __init__(self, env, cfg: MixedObstacleLayoutCfg):
    self.env, self.cfg, self.device, self.num_envs = env, cfg, env.device, env.num_envs
    self.max_obstacles = min(cfg.max_obstacles, V5_MAX_MIXED_OBSTACLES)
    self.slot_type, self.footprint_radius, self.half_extents_xy, self.heights = (
      _build_slot_metadata(self.device)
    )
    self.centers_xy = torch.zeros(
      self.num_envs, self.max_obstacles, 2, device=self.device
    )
    self.active_mask = torch.zeros(
      self.num_envs, self.max_obstacles, dtype=torch.bool, device=self.device
    )
    self.active_slot_ids = torch.full(
      (self.num_envs, self.max_obstacles), -1, dtype=torch.long, device=self.device
    )
    self.num_active = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.violation_rate = torch.zeros(self.num_envs, device=self.device)

  def _slot_clearance(self, slot_id: int) -> float:
    return self.footprint_radius[slot_id].item() + self.cfg.soft_margin

  def _pair_separation_sq(self, slot_a: int, slot_b: int) -> float:
    sep = (
      self.footprint_radius[slot_a]
      + self.footprint_radius[slot_b]
      + self.cfg.min_center_separation
    )
    return float(sep.item() ** 2)

  def sample_layout(
    self, env_ids: torch.Tensor, num_active: torch.Tensor | int
  ) -> None:
    if isinstance(num_active, int):
      num_active = torch.full(
        (len(env_ids),), num_active, dtype=torch.long, device=self.device
      )
    else:
      num_active = num_active.to(device=self.device, dtype=torch.long)
    low = -self.cfg.arena_half_extent + self.cfg.arena_margin
    high = self.cfg.arena_half_extent - self.cfg.arena_margin
    self.centers_xy[env_ids] = 0.0
    self.active_mask[env_ids] = False
    self.active_slot_ids[env_ids] = -1
    self.num_active[env_ids] = torch.clamp(num_active, 0, self.max_obstacles)
    for env_id in env_ids.tolist():
      target = int(self.num_active[env_id])
      placed = 0
      for slot_id in torch.randperm(self.max_obstacles, device=self.device)[
        :target
      ].tolist():
        for _ in range(self.cfg.max_resample_tries):
          candidate = torch.empty(2, device=self.device).uniform_(low, high)
          if (
            self.cfg.exclude_origin
            and torch.sum(candidate.square())
            < (self._slot_clearance(slot_id) + 0.8) ** 2
          ):
            continue
          if placed:
            active_slots = self.active_slot_ids[env_id, :placed]
            dist_sq = torch.sum(
              (self.centers_xy[env_id, :placed] - candidate).square(), dim=1
            )
            sep_sq = torch.tensor(
              [
                self._pair_separation_sq(int(s), slot_id) for s in active_slots.tolist()
              ],
              device=self.device,
            )
            if torch.any(dist_sq < sep_sq):
              continue
          self.centers_xy[env_id, placed] = candidate
          self.active_slot_ids[env_id, placed] = slot_id
          self.active_mask[env_id, placed] = True
          placed += 1
          break
      self.num_active[env_id] = placed

  def write_to_sim(self, env_ids: torch.Tensor) -> None:
    origins = self.env.scene.env_origins[env_ids]
    poses = torch.zeros(len(env_ids), self.max_obstacles, 7, device=self.device)
    poses[..., 3] = 1.0
    poses[..., :3] = origins[:, None, :] + torch.tensor(
      self.cfg.hidden_local_pos, device=self.device
    )
    for row, env_id in enumerate(env_ids.tolist()):
      for i in range(int(self.num_active[env_id])):
        slot = int(self.active_slot_ids[env_id, i])
        poses[row, slot, :2] = origins[row, :2] + self.centers_xy[env_id, i]
        poses[row, slot, 2] = self.heights[slot] * 0.5
    for slot in range(self.max_obstacles):
      self.env.scene[
        f"{self.cfg.obstacle_asset_name}_{slot:03d}"
      ].write_mocap_pose_to_sim(poses[:, slot], env_ids)

  def _distance_to_obstacle_surface(self, query_xy, centers_xy, slot_ids):
    safe_slot_ids = slot_ids.clamp_min(0)
    delta = query_xy.unsqueeze(-2) - centers_xy
    cyl = torch.norm(delta, dim=-1) - self.footprint_radius[safe_slot_ids]
    half = self.half_extents_xy[safe_slot_ids]
    box = torch.sqrt(
      torch.relu(torch.abs(delta[..., 0]) - half[..., 0]).square()
      + torch.relu(torch.abs(delta[..., 1]) - half[..., 1]).square()
    )
    return torch.where(
      self.slot_type[safe_slot_ids] == ObstacleSlotType.CYLINDER, cyl, box
    )

  def is_pose_free(
    self, xy_w: torch.Tensor, radius: float, env_ids: torch.Tensor
  ) -> torch.Tensor:
    local = xy_w - self.env.scene.env_origins[env_ids, :2]
    distance = self._distance_to_obstacle_surface(
      local, self.centers_xy[env_ids], self.active_slot_ids[env_ids]
    )
    return ~torch.any(
      (distance < self.cfg.soft_margin + radius) & self.active_mask[env_ids], dim=-1
    )

  def is_disk_free(self, xy_w, radius, env_ids):
    return self.is_pose_free(xy_w, radius, env_ids)

  def soft_proximity_penalty(self, xy_w: torch.Tensor) -> torch.Tensor:
    local = xy_w - self.env.scene.env_origins[:, :2]
    distance = self._distance_to_obstacle_surface(
      local, self.centers_xy, self.active_slot_ids
    )
    penalty = (
      torch.clamp(1.0 - distance / self.cfg.soft_margin, 0.0, 1.0).square()
      * self.active_mask.float()
    )
    result = penalty.max(dim=1).values
    self.violation_rate[:] = (result > 0).float()
    return result

  def draw_debug_vis(self, visualizer) -> None:
    """Draw active mixed-obstacle soft zones for the selected worlds."""
    identity = np.eye(3)
    for env_id in visualizer.get_env_indices(self.num_envs):
      origin = self.env.scene.env_origins[env_id].detach().cpu().numpy()
      for idx in self.active_mask[env_id].nonzero(as_tuple=False).flatten().tolist():
        slot = int(self.active_slot_ids[env_id, idx])
        center = origin.copy()
        center[:2] += self.centers_xy[env_id, idx].detach().cpu().numpy()
        center[2] = 0.01
        if int(self.slot_type[slot]) == ObstacleSlotType.CYLINDER:
          radius = float(self.footprint_radius[slot]) + self.cfg.soft_margin
          visualizer.add_cylinder(
            center,
            center + np.array([0.0, 0.0, 0.025]),
            radius,
            (1.0, 0.25, 0.15, 0.25),
            label="obstacle soft zone",
          )
        else:
          half = self.half_extents_xy[slot].detach().cpu().numpy()
          visualizer.add_box(
            center,
            np.array(
              [half[0] + self.cfg.soft_margin, half[1] + self.cfg.soft_margin, 0.015]
            ),
            identity,
            (1.0, 0.25, 0.15, 0.25),
            label="obstacle soft zone",
          )


def get_obstacle_layout(env):
  return getattr(env, "obstacle_layout", getattr(env, "cylinder_layout", None))
