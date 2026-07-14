from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi, yaw_quat


class RingPose2dCommand(CommandTerm):
  """Sample 2D pose targets in a distance ring around the robot."""

  cfg: RingPose2dCommandCfg

  def __init__(self, cfg: RingPose2dCommandCfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
    self.pos_command_b = torch.zeros_like(self.pos_command_w)
    self.heading_command_b = torch.zeros_like(self.heading_command_w)
    self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goals_reached"] = torch.zeros(self.num_envs, device=self.device)
    self.goal_reached_this_step = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def __str__(self) -> str:
    msg = "RingPose2dCommand:\n"
    msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
    msg += f"\tDistance range: {self.cfg.ranges.distance}\n"
    msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
    return msg

  @property
  def command(self) -> torch.Tensor:
    return torch.cat((self.pos_command_b, self.heading_command_b.unsqueeze(1)), dim=1)

  def _update_metrics(self) -> None:
    self.metrics["error_pos_2d"][:] = torch.norm(
      self.pos_command_w[:, :2] - self.robot.data.root_link_pos_w[:, :2], dim=1
    )
    self.metrics["error_heading"][:] = torch.abs(
      wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
    )

  def _sample_goal_xy(self, env_ids: torch.Tensor) -> torch.Tensor:
    root = self.robot.data.root_link_pos_w[env_ids]
    angle = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 2.0 * math.pi)
    distance = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.ranges.distance
    )
    return torch.stack(
      (
        root[:, 0] + distance * torch.cos(angle),
        root[:, 1] + distance * torch.sin(angle),
      ),
      dim=1,
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    goal_xy = self._sample_goal_xy(env_ids)
    if self.cfg.obstacle_filter:
      from mjlab.tasks.navigation.mdp.obstacles import get_obstacle_layout

      layout = get_obstacle_layout(self._env)
      if layout is not None:
        for _ in range(self.cfg.max_obstacle_resample_tries):
          free = layout.is_pose_free(goal_xy, self.cfg.success_radius, env_ids)
          if bool(torch.all(free)):
            break
          goal_xy[~free] = self._sample_goal_xy(env_ids[~free])
    self.pos_command_w[env_ids, :2] = goal_xy
    self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]
    root = self.robot.data.root_link_pos_w[env_ids]
    if self.cfg.simple_heading:
      target = torch.atan2(goal_xy[:, 1] - root[:, 1], goal_xy[:, 0] - root[:, 0])
      flipped = wrap_to_pi(target + torch.pi)
      heading = self.robot.data.heading_w[env_ids]
      self.heading_command_w[env_ids] = torch.where(
        wrap_to_pi(target - heading).abs() < wrap_to_pi(flipped - heading).abs(),
        target,
        flipped,
      )
    else:
      self.heading_command_w[env_ids] = torch.empty(
        len(env_ids), device=self.device
      ).uniform_(*self.cfg.ranges.heading)
    self._update_command_for_envs(env_ids)

  def _update_command_for_envs(self, env_ids: torch.Tensor) -> None:
    delta = self.pos_command_w[env_ids] - self.robot.data.root_link_pos_w[env_ids]
    self.pos_command_b[env_ids] = quat_apply_inverse(
      yaw_quat(self.robot.data.root_link_quat_w[env_ids]), delta
    )
    self.heading_command_b[env_ids] = wrap_to_pi(
      self.heading_command_w[env_ids] - self.robot.data.heading_w[env_ids]
    )

  def _update_command(self) -> None:
    env_ids = torch.arange(self.num_envs, device=self.device)
    self._update_command_for_envs(env_ids)
    if self.cfg.update_goal_on_success:
      reached = torch.norm(self.pos_command_b[:, :2], dim=1) < self.cfg.success_radius
      self.goal_reached_this_step[:] = reached
      reached_ids = reached.nonzero(as_tuple=False).flatten()
      self.metrics["goals_reached"][reached_ids] += 1.0
      self._resample(reached_ids)
    else:
      self.goal_reached_this_step[:] = False

  def _debug_vis_impl(self, visualizer) -> None:
    """Draw the source goal region, heading and distance line natively."""
    goal = self.pos_command_w.detach().cpu().numpy()
    root = self.robot.data.root_link_pos_w.detach().cpu().numpy()
    heading = self.heading_command_w.detach().cpu().numpy()
    for env_id in visualizer.get_env_indices(self.num_envs):
      center = goal[env_id].copy()
      center[2] = 0.02
      visualizer.add_cylinder(
        center,
        center + np.array([0.0, 0.0, 0.03]),
        self.cfg.success_radius,
        (0.2, 0.8, 0.2, 0.25),
        label="navigation goal region",
      )
      post_start = goal[env_id].copy()
      post_start[2] += 0.02
      visualizer.add_cylinder(
        post_start,
        post_start + np.array([0.0, 0.0, 0.6]),
        0.08,
        (1.0, 0.55, 0.1, 1.0),
        label="navigation goal post",
      )
      head_start = goal[env_id].copy()
      head_start[2] += 0.12
      head_end = head_start + 0.5 * np.array(
        [math.cos(heading[env_id]), math.sin(heading[env_id]), 0.0]
      )
      visualizer.add_arrow(
        head_start, head_end, (0.2, 0.8, 0.2, 0.9), label="goal heading"
      )
      line_start = root[env_id].copy()
      line_start[2] += 0.48
      line_end = goal[env_id].copy()
      line_end[2] += 0.05
      visualizer.add_arrow(
        line_start, line_end, (0.9, 0.8, 0.1, 0.7), width=0.008, label="goal distance"
      )
    from ..obstacles import get_obstacle_layout

    layout = get_obstacle_layout(self._env)
    if layout is not None and hasattr(layout, "draw_debug_vis"):
      layout.draw_debug_vis(visualizer)

    # mjlab currently delegates visualizers through command/event/reward managers,
    # so draw the source action arrows from this navigation command callback.
    action_name = "pre_trained_policy_action"
    if action_name in self._env.action_manager.active_terms:
      action_term = cast(Any, self._env.action_manager.get_term(action_name))
      if action_term.cfg.debug_vis:
        action_term.draw_debug_vis(visualizer)


@dataclass(kw_only=True)
class RingPose2dCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    distance: tuple[float, float]
    heading: tuple[float, float] = (-math.pi, math.pi)

  entity_name: str
  simple_heading: bool = False
  ranges: Ranges = field(
    default_factory=lambda: RingPose2dCommandCfg.Ranges((2.0, 5.0))
  )
  success_radius: float = 1.0
  update_goal_on_success: bool = False
  obstacle_filter: bool = False
  max_obstacle_resample_tries: int = 128

  def build(self, env) -> RingPose2dCommand:
    return RingPose2dCommand(self, env)
