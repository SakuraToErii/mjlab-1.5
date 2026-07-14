from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from ..obstacles import FixedMazeLayout
from .ring_pose_command import RingPose2dCommand, RingPose2dCommandCfg


class MazeExitCommand(RingPose2dCommand):
  """Command the robot between fixed maze endpoints instead of random ring goals."""

  cfg: MazeExitCommandCfg

  def __init__(self, cfg: MazeExitCommandCfg, env):
    super().__init__(cfg, env)
    self.target_is_exit = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)
    self.target_is_exit[env_ids] = True
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
    layout = getattr(self._env, "fixed_maze_layout", None)
    if layout is None:
      layout = getattr(self._env, "cylinder_layout", None)
    if not isinstance(layout, FixedMazeLayout):
      raise RuntimeError(
        "MazeExitCommand requires FixedMazeLayout before command reset."
      )

    target_is_exit = self.target_is_exit[env_ids]
    goal_xy = layout.endpoint_xy_w(env_ids, target_is_exit)
    if self.cfg.goal_jitter_radius > 0.0:
      goal_xy = self._jitter_goal_xy(layout, env_ids, goal_xy)
    self.pos_command_w[env_ids, :2] = goal_xy
    self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]
    self.heading_command_w[env_ids] = torch.where(
      target_is_exit,
      layout.exit_yaw[env_ids],
      wrap_to_pi(layout.entrance_yaw[env_ids] + math.pi),
    )
    self._update_command_for_envs(env_ids)

  def _jitter_goal_xy(
    self, layout: FixedMazeLayout, env_ids: torch.Tensor, goal_xy: torch.Tensor
  ) -> torch.Tensor:
    jittered = goal_xy.clone()
    pending = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
    for _ in range(self.cfg.max_obstacle_resample_tries):
      angle = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 2.0 * math.pi)
      radius = (
        torch.sqrt(torch.rand(len(env_ids), device=self.device))
        * self.cfg.goal_jitter_radius
      )
      candidate = goal_xy + torch.stack(
        (radius * torch.cos(angle), radius * torch.sin(angle)), dim=1
      )
      free = layout.is_disk_free(candidate, self.cfg.success_radius, env_ids)
      accepted = pending & free
      jittered[accepted] = candidate[accepted]
      pending &= ~free
      if not torch.any(pending):
        break
    return jittered

  def _update_command(self):
    target = self.pos_command_w - self.robot.data.root_link_pos_w
    self.pos_command_b[:] = quat_apply_inverse(
      yaw_quat(self.robot.data.root_link_quat_w), target
    )
    self.heading_command_b[:] = wrap_to_pi(
      self.heading_command_w - self.robot.data.heading_w
    )
    if self.cfg.update_goal_on_success:
      reached = torch.norm(self.pos_command_b[:, :2], dim=1) < self.cfg.success_radius
      self.goal_reached_this_step[:] = reached
      reached_ids = reached.nonzero(as_tuple=False).flatten()
      if len(reached_ids) > 0:
        self.metrics["goals_reached"][reached_ids] += 1.0
        self.target_is_exit[reached_ids] = ~self.target_is_exit[reached_ids]
        self._resample(reached_ids)
    else:
      self.goal_reached_this_step[:] = False


@dataclass(kw_only=True)
class MazeExitCommandCfg(RingPose2dCommandCfg):
  """Configuration for fixed maze endpoint commands."""

  ranges: RingPose2dCommandCfg.Ranges = field(
    default_factory=lambda: RingPose2dCommandCfg.Ranges(distance=(0.0, 0.0))
  )
  goal_jitter_radius: float = 0.25

  def build(self, env) -> MazeExitCommand:
    return MazeExitCommand(self, env)
