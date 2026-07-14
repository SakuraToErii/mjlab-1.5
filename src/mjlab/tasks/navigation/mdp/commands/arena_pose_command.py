from __future__ import annotations

from dataclasses import dataclass

import torch

from .ring_pose_command import RingPose2dCommand, RingPose2dCommandCfg


class ArenaPose2dCommand(RingPose2dCommand):
  """Sample 2D pose targets uniformly across the full navigable arena."""

  cfg: ArenaPose2dCommandCfg

  def __str__(self) -> str:
    msg = "ArenaPose2dCommand:\n"
    msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
    msg += f"\tArena half extent: {self.cfg.arena_half_extent}\n"
    msg += f"\tArena margin: {self.cfg.arena_margin}\n"
    msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
    return msg

  def _sample_goal_xy(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Sample world-frame goal positions uniformly inside the arena bounds."""
    origin = self._env.scene.env_origins[env_ids, :2]
    extent = self.cfg.arena_half_extent - self.cfg.arena_margin
    return origin + torch.empty(len(env_ids), 2, device=self.device).uniform_(
      -extent, extent
    )


@dataclass(kw_only=True)
class ArenaPose2dCommandCfg(RingPose2dCommandCfg):
  """Configuration for uniform arena-wide 2D pose targets."""

  arena_half_extent: float = 8.0
  """Half-size of the square arena in env-local coordinates (m)."""
  arena_margin: float = 0.5
  """Keep sampled goals away from the arena boundary (m)."""

  def build(self, env) -> ArenaPose2dCommand:
    return ArenaPose2dCommand(self, env)
