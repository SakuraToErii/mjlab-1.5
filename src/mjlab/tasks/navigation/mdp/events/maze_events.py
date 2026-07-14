from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from ..obstacles import FixedMazeLayout, FixedMazeLayoutCfg

_ROBOT_CFG = SceneEntityCfg("robot")


def _env_ids_tensor(env, env_ids) -> torch.Tensor:
  if env_ids is None or isinstance(env_ids, slice):
    return torch.arange(env.num_envs, device=env.device)
  return env_ids


def ensure_fixed_maze_layout(env, layout_cfg: FixedMazeLayoutCfg) -> FixedMazeLayout:
  """Create the shared fixed maze layout on first use."""
  layout = getattr(env, "cylinder_layout", None)
  if not isinstance(layout, FixedMazeLayout):
    layout = FixedMazeLayout(env, layout_cfg)
    env.cylinder_layout = layout
    env.fixed_maze_layout = layout
  return layout


def load_fixed_maze_layout(
  env,
  env_ids: torch.Tensor,
  layout_cfg: FixedMazeLayoutCfg,
  default_template_level: int = 0,
):
  """Load fixed maze wall templates and write cylinders to the scene."""
  env_ids = _env_ids_tensor(env, env_ids)
  layout = ensure_fixed_maze_layout(env, layout_cfg)
  template_level = getattr(env, "maze_template_level", None)
  active_levels = (
    default_template_level if template_level is None else template_level[env_ids]
  )
  layout.load_template(env_ids, active_levels)


def reset_robot_at_maze_entrance(
  env,
  env_ids: torch.Tensor,
  pose_noise: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
):
  """Reset the robot inside the current maze template entrance region."""
  env_ids = _env_ids_tensor(env, env_ids)
  layout = getattr(env, "fixed_maze_layout", None)
  if layout is None:
    layout = getattr(env, "cylinder_layout", None)
  if not isinstance(layout, FixedMazeLayout):
    raise RuntimeError("FixedMazeLayout must be loaded before the robot reset.")

  asset = env.scene[asset_cfg.name]
  root = asset.data.default_root_state[env_ids].clone()
  keys = ("x", "y", "z", "roll", "pitch", "yaw")
  pose_ranges = torch.tensor(
    [pose_noise.get(key, (0.0, 0.0)) for key in keys], device=env.device
  )
  samples = sample_uniform(
    pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), env.device
  )
  positions = root[:, :3] + env.scene.env_origins[env_ids]
  positions[:, :2] += layout.entrance_xy[env_ids] + samples[:, :2]
  positions[:, 2] += samples[:, 2]
  yaw = layout.entrance_yaw[env_ids] + samples[:, 5]
  rotations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(samples[:, 3], samples[:, 4], yaw)
  )

  velocity_ranges = torch.tensor(
    [velocity_range.get(key, (0.0, 0.0)) for key in keys], device=env.device
  )
  velocities = root[:, 7:13] + sample_uniform(
    velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), env.device
  )
  asset.write_root_link_pose_to_sim(torch.cat((positions, rotations), dim=-1), env_ids)
  asset.write_root_link_velocity_to_sim(velocities, env_ids)
