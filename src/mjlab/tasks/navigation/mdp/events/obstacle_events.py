from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from ..obstacles import (
  CylinderObstacleLayout,
  CylinderObstacleLayoutCfg,
  MixedObstacleLayout,
  MixedObstacleLayoutCfg,
  get_obstacle_layout,
)

_ROBOT_CFG = SceneEntityCfg("robot")


def _env_ids_tensor(env, env_ids):
  return (
    torch.arange(env.num_envs, device=env.device)
    if env_ids is None or isinstance(env_ids, slice)
    else env_ids
  )


def ensure_cylinder_layout(env, layout_cfg):
  layout = getattr(env, "cylinder_layout", None)
  if layout is None:
    layout = CylinderObstacleLayout(env, layout_cfg)
    env.cylinder_layout = layout
  return layout


def ensure_mixed_obstacle_layout(env, layout_cfg):
  layout = getattr(env, "obstacle_layout", None)
  if layout is None:
    layout = MixedObstacleLayout(env, layout_cfg)
    env.obstacle_layout = layout
  return layout


def randomize_cylinder_layout(
  env, env_ids, layout_cfg: CylinderObstacleLayoutCfg, default_num_active: int = 0
):
  env_ids = _env_ids_tensor(env, env_ids)
  layout = ensure_cylinder_layout(env, layout_cfg)
  counts = getattr(env, "obstacle_num_active", None)
  layout.sample_layout(
    env_ids, default_num_active if counts is None else counts[env_ids]
  )
  layout.write_to_sim(env_ids)


def randomize_mixed_obstacle_layout(
  env, env_ids, layout_cfg: MixedObstacleLayoutCfg, default_num_active: int = 0
):
  env_ids = _env_ids_tensor(env, env_ids)
  layout = ensure_mixed_obstacle_layout(env, layout_cfg)
  counts = getattr(env, "obstacle_num_active", None)
  layout.sample_layout(
    env_ids, default_num_active if counts is None else counts[env_ids]
  )
  layout.write_to_sim(env_ids)


def reset_root_state_obstacle_aware(
  env,
  env_ids,
  pose_range,
  velocity_range,
  robot_radius: float = 0.5,
  max_resample_tries: int = 64,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
):
  env_ids = _env_ids_tensor(env, env_ids)
  asset = env.scene[asset_cfg.name]
  root = asset.data.default_root_state[env_ids].clone()
  keys = ("x", "y", "z", "roll", "pitch", "yaw")
  ranges = torch.tensor(
    [pose_range.get(key, (0.0, 0.0)) for key in keys], device=env.device
  )
  samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), env.device)
  layout = get_obstacle_layout(env)
  if layout is not None:
    for _ in range(max_resample_tries):
      positions = root[:, :3] + env.scene.env_origins[env_ids] + samples[:, :3]
      free = layout.is_pose_free(positions[:, :2], robot_radius, env_ids)
      if bool(torch.all(free)):
        break
      samples[~free] = sample_uniform(
        ranges[:, 0], ranges[:, 1], (int((~free).sum()), 6), env.device
      )
  positions = root[:, :3] + env.scene.env_origins[env_ids] + samples[:, :3]
  rotations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
  )
  vel_ranges = torch.tensor(
    [velocity_range.get(key, (0.0, 0.0)) for key in keys], device=env.device
  )
  velocities = root[:, 7:13] + sample_uniform(
    vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), env.device
  )
  asset.write_root_link_pose_to_sim(torch.cat((positions, rotations), dim=-1), env_ids)
  asset.write_root_link_velocity_to_sim(velocities, env_ids)
