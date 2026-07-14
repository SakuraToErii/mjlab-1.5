from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch

from .commands.velocity_command import AdaptiveLevelVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _curriculum_fraction(level: int, num_levels: int) -> float:
  if num_levels <= 1:
    return 0.0
  return float(level - 1) / float(num_levels - 1)


def _lerp_scalar(
  a: float, b: float, level: int, num_levels: int, decimals: int = 3
) -> float:
  value = a + _curriculum_fraction(level, num_levels) * (b - a)
  return round(float(value), decimals)


def _lerp_range(
  initial_range: tuple[float, float],
  limit_range: tuple[float, float],
  level: int,
  num_levels: int,
  decimals: int = 3,
) -> tuple[float, float]:
  return (
    _lerp_scalar(initial_range[0], limit_range[0], level, num_levels, decimals),
    _lerp_scalar(initial_range[1], limit_range[1], level, num_levels, decimals),
  )


def _apply_command_ranges_from_level(
  env: ManagerBasedRlEnv,
  level: int,
  num_levels: int,
  command_name: str = "base_velocity",
) -> None:
  command_term = cast(
    AdaptiveLevelVelocityCommand,
    env.command_manager.get_term(command_name),
  )
  ranges = command_term.cfg.ranges
  limit_ranges = command_term.cfg.limit_ranges
  state = cast(Any, env)

  if not hasattr(state, "_curriculum_cmd_initial_ranges"):
    state._curriculum_cmd_initial_ranges = {
      "lin_vel_x": tuple(ranges.lin_vel_x),
      "lin_vel_y": tuple(ranges.lin_vel_y),
      "ang_vel_z": tuple(ranges.ang_vel_z),
    }

  initial_ranges: dict[str, tuple[float, float]] = state._curriculum_cmd_initial_ranges
  ranges.lin_vel_x = _lerp_range(
    initial_ranges["lin_vel_x"], tuple(limit_ranges.lin_vel_x), level, num_levels
  )
  ranges.lin_vel_y = _lerp_range(
    initial_ranges["lin_vel_y"], tuple(limit_ranges.lin_vel_y), level, num_levels
  )
  ranges.ang_vel_z = _lerp_range(
    initial_ranges["ang_vel_z"], tuple(limit_ranges.ang_vel_z), level, num_levels
  )


def _apply_tracking_std_from_level(
  env: ManagerBasedRlEnv,
  level: int,
  num_levels: int,
  lin_var_initial: float,
  lin_var_final: float,
  ang_var_initial: float,
  ang_var_final: float,
  lin_term_name: str = "track_lin_vel_xy",
  ang_term_name: str = "track_ang_vel_z",
  decimals: int = 3,
) -> None:
  lin_var = _lerp_scalar(lin_var_initial, lin_var_final, level, num_levels, decimals)
  ang_var = _lerp_scalar(ang_var_initial, ang_var_final, level, num_levels, decimals)

  # mjlab returns the live RewardTermCfg stored by RewardManager, so mutating
  # ``params`` updates the callable's next evaluation without a setter round-trip.
  lin_cfg = env.reward_manager.get_term_cfg(lin_term_name)
  lin_cfg.params["std"] = round(math.sqrt(max(lin_var, 1e-9)), decimals)

  ang_cfg = env.reward_manager.get_term_cfg(ang_term_name)
  ang_cfg.params["std"] = round(math.sqrt(max(ang_var, 1e-9)), decimals)


def global_low_level_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "base_velocity",
  num_levels: int = 5,
  promotion_ratio: float = 0.8,
  lin_var_initial: float = 0.25,
  lin_var_final: float = 0.18,
  ang_var_initial: float = 0.25,
  ang_var_final: float = 0.15,
  lin_term_name: str = "track_lin_vel_xy",
  ang_term_name: str = "track_ang_vel_z",
  forced_level: int | None = None,
  decimals: int = 3,
) -> dict[str, float]:
  state = cast(Any, env)
  if not hasattr(state, "global_curriculum_level"):
    state.global_curriculum_level = 1

  if forced_level is not None:
    state.global_curriculum_level = int(min(max(forced_level, 1), num_levels))
  elif (
    env.common_step_counter % env.max_episode_length == 0
    and state.global_curriculum_level < num_levels
  ):
    reward_manager = env.reward_manager
    reward_lin_cfg = reward_manager.get_term_cfg(lin_term_name)
    reward_ang_cfg = reward_manager.get_term_cfg(ang_term_name)
    reward_lin = (
      torch.mean(reward_manager._episode_sums[lin_term_name][env_ids])
      / env.max_episode_length_s
    )
    reward_ang = (
      torch.mean(reward_manager._episode_sums[ang_term_name][env_ids])
      / env.max_episode_length_s
    )

    if (
      reward_lin > reward_lin_cfg.weight * promotion_ratio
      and reward_ang > reward_ang_cfg.weight * promotion_ratio
    ):
      state.global_curriculum_level += 1

  level = int(min(max(state.global_curriculum_level, 1), num_levels))
  state.global_curriculum_level = level

  _apply_command_ranges_from_level(env, level, num_levels, command_name=command_name)
  _apply_tracking_std_from_level(
    env,
    level,
    num_levels,
    lin_var_initial=lin_var_initial,
    lin_var_final=lin_var_final,
    ang_var_initial=ang_var_initial,
    ang_var_final=ang_var_final,
    lin_term_name=lin_term_name,
    ang_term_name=ang_term_name,
    decimals=decimals,
  )

  lin_cfg = env.reward_manager.get_term_cfg(lin_term_name)
  ang_cfg = env.reward_manager.get_term_cfg(ang_term_name)
  command_term = cast(
    AdaptiveLevelVelocityCommand,
    env.command_manager.get_term(command_name),
  )
  ranges = command_term.cfg.ranges

  return {
    "level": float(level),
    "cmd_lin_vel_x_min": float(ranges.lin_vel_x[0]),
    "cmd_lin_vel_x_max": float(ranges.lin_vel_x[1]),
    "cmd_lin_vel_y_min": float(ranges.lin_vel_y[0]),
    "cmd_lin_vel_y_max": float(ranges.lin_vel_y[1]),
    "cmd_ang_vel_z_min": float(ranges.ang_vel_z[0]),
    "cmd_ang_vel_z_max": float(ranges.ang_vel_z[1]),
    "track_lin_std": float(lin_cfg.params["std"]),
    "track_ang_std": float(ang_cfg.params["std"]),
  }


def sequential_low_level_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "base_velocity",
  num_levels: int = 5,
  lin_promotion_threshold: float = 0.75,
  ang_promotion_threshold: float = 0.5,
  lin_term_name: str = "track_lin_vel_xy",
  ang_term_name: str = "track_ang_vel_z",
  forced_level: int | None = None,
) -> dict[str, float]:
  """Two-phase sequential curriculum for the low-level locomotion task.

  Phase 1 widens the command ranges level by level. A level is promoted only when both the mean
  linear and angular tracking rewards per second exceed their (absolute) thresholds. Tracking std
  is left untouched (kept fixed via the reward config) and adaptive sampling stays disabled.

  Phase 2 begins once the maximum level is reached: command ranges stay at ``limit_ranges``, no
  further promotion happens, and adaptive command sampling is enabled on the command term.
  """
  state = cast(Any, env)
  if not hasattr(state, "global_curriculum_level"):
    state.global_curriculum_level = 1

  command_term = cast(
    AdaptiveLevelVelocityCommand,
    env.command_manager.get_term(command_name),
  )

  if forced_level is not None:
    state.global_curriculum_level = int(min(max(forced_level, 1), num_levels))
  elif (
    env.common_step_counter % env.max_episode_length == 0
    and state.global_curriculum_level < num_levels
  ):
    reward_manager = env.reward_manager
    reward_lin = (
      torch.mean(reward_manager._episode_sums[lin_term_name][env_ids])
      / env.max_episode_length_s
    )
    reward_ang = (
      torch.mean(reward_manager._episode_sums[ang_term_name][env_ids])
      / env.max_episode_length_s
    )

    if reward_lin > lin_promotion_threshold and reward_ang > ang_promotion_threshold:
      state.global_curriculum_level += 1

  level = int(min(max(state.global_curriculum_level, 1), num_levels))
  state.global_curriculum_level = level

  _apply_command_ranges_from_level(env, level, num_levels, command_name=command_name)

  # Phase 2: enable adaptive sampling once the command curriculum is maxed out.
  phase = 2 if level >= num_levels else 1
  command_term.cfg.adaptive_sampling = phase == 2
  state.sequential_curriculum_phase = phase

  ranges = command_term.cfg.ranges
  return {
    "level": float(level),
    "phase": float(phase),
    "adaptive_sampling_active": float(command_term.cfg.adaptive_sampling),
    "cmd_lin_vel_x_min": float(ranges.lin_vel_x[0]),
    "cmd_lin_vel_x_max": float(ranges.lin_vel_x[1]),
    "cmd_lin_vel_y_min": float(ranges.lin_vel_y[0]),
    "cmd_lin_vel_y_max": float(ranges.lin_vel_y[1]),
    "cmd_ang_vel_z_min": float(ranges.ang_vel_z[0]),
    "cmd_ang_vel_z_max": float(ranges.ang_vel_z[1]),
  }


def lin_vel_cmd_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
  command_term = cast(
    AdaptiveLevelVelocityCommand,
    env.command_manager.get_term("base_velocity"),
  )
  ranges = command_term.cfg.ranges
  limit_ranges = command_term.cfg.limit_ranges

  reward_term = env.reward_manager.get_term_cfg(reward_term_name)
  reward = (
    torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
    / env.max_episode_length_s
  )

  if env.common_step_counter % env.max_episode_length == 0:
    if reward > reward_term.weight * 0.8:
      delta_command = torch.tensor([-0.1, 0.1], device=env.device)
      lin_vel_x = torch.clamp(
        torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
        limit_ranges.lin_vel_x[0],
        limit_ranges.lin_vel_x[1],
      )
      lin_vel_y = torch.clamp(
        torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
        limit_ranges.lin_vel_y[0],
        limit_ranges.lin_vel_y[1],
      )
      ranges.lin_vel_x = (float(lin_vel_x[0]), float(lin_vel_x[1]))
      ranges.lin_vel_y = (float(lin_vel_y[0]), float(lin_vel_y[1]))

  return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
  command_term = cast(
    AdaptiveLevelVelocityCommand,
    env.command_manager.get_term("base_velocity"),
  )
  ranges = command_term.cfg.ranges
  limit_ranges = command_term.cfg.limit_ranges

  reward_term = env.reward_manager.get_term_cfg(reward_term_name)
  reward = (
    torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
    / env.max_episode_length_s
  )

  if env.common_step_counter % env.max_episode_length == 0:
    if reward > reward_term.weight * 0.8:
      delta_command = torch.tensor([-0.1, 0.1], device=env.device)
      ang_vel_z = torch.clamp(
        torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
        limit_ranges.ang_vel_z[0],
        limit_ranges.ang_vel_z[1],
      )
      ranges.ang_vel_z = (float(ang_vel_z[0]), float(ang_vel_z[1]))

  return torch.tensor(ranges.ang_vel_z[1], device=env.device)
