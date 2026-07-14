from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def obstacle_count_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "pose_command",
  level_counts: tuple[int, ...] = (0, 2, 4, 6, 8),
  promote_goals_reached: float = 0.8,
  demote_on_fall: bool = True,
) -> dict[str, torch.Tensor]:
  """Adjust per-environment active obstacle count from recent episode outcomes."""
  state = cast(Any, env)
  if not hasattr(state, "obstacle_curriculum_level"):
    state.obstacle_curriculum_level = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
  if not hasattr(state, "obstacle_num_active"):
    state.obstacle_num_active = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  levels: torch.Tensor = state.obstacle_curriculum_level
  command_term = cast(Any, env.command_manager.get_term(command_name))
  goals_reached = command_term.metrics["goals_reached"][env_ids]

  fall = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
  for term_name in ("bad_orientation", "base_height"):
    if term_name in env.termination_manager.active_terms:
      fall |= env.termination_manager.get_term(term_name)[env_ids].bool()

  promote = (goals_reached >= promote_goals_reached) & ~fall
  levels[env_ids[promote]] = torch.clamp(
    levels[env_ids[promote]] + 1, max=len(level_counts) - 1
  )

  if demote_on_fall:
    levels[env_ids[fall]] = torch.clamp(levels[env_ids[fall]] - 1, min=0)

  counts = torch.tensor(level_counts, dtype=torch.long, device=env.device)
  obstacle_num_active: torch.Tensor = state.obstacle_num_active
  obstacle_num_active[:] = counts[levels]

  return {
    "level": torch.mean(levels.float()),
    "num_active": torch.mean(obstacle_num_active.float()),
  }
