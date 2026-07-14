from __future__ import annotations

from typing import Any, cast

import torch

from mjlab.managers.manager_base import ManagerTermBase


class pose_command_progress(ManagerTermBase):
  def __init__(self, cfg, env):
    super().__init__(env)
    self.cfg = cfg
    self.prev_distance = torch.zeros(env.num_envs, device=env.device)
    self.last_command_counter = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  def reset(self, env_ids):
    params = cast(dict[str, Any], self.cfg.params)
    name = cast(str, params["command_name"])
    command = cast(torch.Tensor, self._env.command_manager.get_command(name))
    self.prev_distance[env_ids] = torch.norm(command[env_ids, :2], dim=1)
    term = cast(Any, self._env.command_manager.get_term(name))
    self.last_command_counter[env_ids] = term.command_counter[env_ids]

  def __call__(self, env, command_name: str):
    command = env.command_manager.get_command(command_name)
    term = cast(Any, env.command_manager.get_term(command_name))
    distance = torch.norm(command[:, :2], dim=1)
    progress = self.prev_distance - distance
    progress[term.command_counter != self.last_command_counter] = 0.0
    self.prev_distance[:] = distance
    self.last_command_counter[:] = term.command_counter
    return progress


def position_command_error_tanh(env, std: float, command_name: str):
  return 1.0 - torch.tanh(
    torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) / std
  )


def goal_reached_bonus(env, command_name: str):
  command = env.command_manager.get_command(command_name)
  term = env.command_manager.get_term(command_name)
  if term.cfg.update_goal_on_success:
    return term.goal_reached_this_step.float()
  return (torch.norm(command[:, :2], dim=1) < term.cfg.success_radius).float()


def obstacle_soft_zone_penalty(env):
  from mjlab.tasks.navigation.mdp.obstacles import get_obstacle_layout

  layout = get_obstacle_layout(env)
  if layout is None:
    return torch.zeros(env.num_envs, device=env.device)
  return layout.soft_proximity_penalty(env.scene["robot"].data.root_link_pos_w[:, :2])


def action_l2(env):
  return env.action_manager.action.square().sum(dim=-1)


def is_terminated_term(env, term_keys: list[str]):
  result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for key in term_keys:
    result |= env.termination_manager.get_term(key)
  return result.float()
