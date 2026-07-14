from __future__ import annotations

from typing import cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

_ROBOT_CFG = SceneEntityCfg("robot")


"""Joint penalties."""


def energy(env, asset_cfg: SceneEntityCfg = _ROBOT_CFG):
  """Penalize the energy used by the robot's joints."""
  asset = env.scene[asset_cfg.name]
  qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  qfrc = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  return (qvel.abs() * qfrc.abs()).sum(dim=-1)


def stand_still(
  env, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = _ROBOT_CFG
):
  asset = env.scene[asset_cfg.name]
  deviation = (asset.data.joint_pos - asset.data.default_joint_pos).abs().sum(dim=1)
  return deviation * (
    torch.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
  )


def joint_deviation_l1(env, asset_cfg: SceneEntityCfg):
  asset = env.scene[asset_cfg.name]
  return (
    (
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    .abs()
    .sum(dim=-1)
  )


"""Robot rewards."""


def orientation_l2(
  env, desired_gravity: list[float], asset_cfg: SceneEntityCfg = _ROBOT_CFG
):
  # extract the used quantities (to enable type-hinting)
  desired = torch.tensor(desired_gravity, device=env.device)
  projected = env.scene[asset_cfg.name].data.projected_gravity_b
  cos_dist = torch.sum(projected * desired, dim=-1)  # cosine distance
  normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
  return torch.square(normalized)


def upward(env, asset_cfg: SceneEntityCfg = _ROBOT_CFG):
  # extract the used quantities (to enable type-hinting)
  projected = env.scene[asset_cfg.name].data.projected_gravity_b
  return torch.square(1.0 - projected[:, 2])


def joint_position_penalty(
  env,
  asset_cfg: SceneEntityCfg,
  stand_still_scale: float,
  velocity_threshold: float,
):
  # extract the used quantities (to enable type-hinting)
  asset = env.scene[asset_cfg.name]
  cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
  body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
  penalty = torch.linalg.norm(
    asset.data.joint_pos - asset.data.default_joint_pos, dim=1
  )
  return torch.where(
    torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
    penalty,
    stand_still_scale * penalty,
  )


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = _ROBOT_CFG):
  return env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 2].square()


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = _ROBOT_CFG):
  return env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, :2].square().sum(dim=-1)


def base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg = _ROBOT_CFG):
  return (env.scene[asset_cfg.name].data.root_link_pos_w[:, 2] - target_height).square()


"""Feet rewards."""


def feet_stumble(env, sensor_name: str):
  # extract the used quantities (to enable type-hinting)
  sensor = env.scene.sensors[sensor_name]
  force = cast(torch.Tensor, sensor.data.force)
  forces_z = force[..., 2].abs()
  forces_xy = torch.linalg.norm(force[..., :2], dim=-1)
  # Penalize feet hitting vertical surfaces
  return torch.any(forces_xy > 4.0 * forces_z, dim=1).float()


def feet_height_body(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  target_height: float,
  tanh_mult: float,
):
  asset = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  foot_pos = asset.data.body_link_pos_w[
    :, body_ids
  ] - asset.data.root_link_pos_w.unsqueeze(1)
  foot_vel = asset.data.body_link_lin_vel_w[
    :, body_ids
  ] - asset.data.root_link_lin_vel_w.unsqueeze(1)
  quat = asset.data.root_link_quat_w
  foot_pos_b = torch.stack(
    [quat_apply_inverse(quat, foot_pos[:, i]) for i in range(foot_pos.shape[1])], dim=1
  )
  foot_vel_b = torch.stack(
    [quat_apply_inverse(quat, foot_vel[:, i]) for i in range(foot_vel.shape[1])], dim=1
  )
  height_error = (foot_pos_b[..., 2] - target_height).square()
  velocity = torch.tanh(tanh_mult * foot_vel_b[..., :2].norm(dim=-1))
  reward = (height_error * velocity).sum(dim=1)
  reward *= (
    torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
  )
  reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
  return reward


def foot_clearance_reward(
  env,
  asset_cfg: SceneEntityCfg,
  target_height: float,
  std: float,
  tanh_mult: float,
):
  asset = env.scene[asset_cfg.name]
  error = (
    asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - target_height
  ).square()
  velocity = torch.tanh(
    tanh_mult * asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
  )
  return torch.exp(-(error * velocity).sum(dim=-1) / std)


def feet_too_near(env, threshold: float = 0.2, asset_cfg: SceneEntityCfg = _ROBOT_CFG):
  feet_pos = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids]
  return (threshold - torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)).clamp(min=0)


def feet_contact_without_cmd(
  env, sensor_name: str, command_name: str = "base_velocity"
):
  sensor = env.scene.sensors[sensor_name]
  current_contact_time = cast(torch.Tensor, sensor.data.current_contact_time)
  contact_count = (current_contact_time > 0).float().sum(dim=-1)
  return contact_count * (
    torch.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
  )


def air_time_variance_penalty(env, sensor_name: str):
  # extract the used quantities (to enable type-hinting)
  sensor = env.scene.sensors[sensor_name]
  if not sensor.cfg.track_air_time:
    raise RuntimeError("Activate ContactSensor's track_air_time!")
  last_air_time = cast(torch.Tensor, sensor.data.last_air_time)
  last_contact_time = cast(torch.Tensor, sensor.data.last_contact_time)
  # compute the reward
  return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
    torch.clip(last_contact_time, max=0.5), dim=1
  )


def feet_gait(
  env,
  period: float,
  offset: list[float],
  sensor_name: str,
  threshold: float = 0.5,
  command_name: str | None = None,
):
  sensor = env.scene.sensors[sensor_name]
  current_contact_time = cast(torch.Tensor, sensor.data.current_contact_time)
  is_contact = current_contact_time > 0
  phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
  leg_phase = torch.cat([(phase + value) % 1.0 for value in offset], dim=-1)
  reward = ((leg_phase < threshold) == is_contact).float().sum(dim=-1)
  if command_name is not None:
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
  return reward


def feet_slide(env, sensor_name: str, asset_cfg: SceneEntityCfg):
  sensor = env.scene.sensors[sensor_name]
  found = cast(torch.Tensor, sensor.data.found)
  asset = env.scene[asset_cfg.name]
  velocity = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
  return (velocity * (found > 0)).sum(dim=-1)


"""Other rewards."""


def joint_mirror(env, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]):
  # extract the used quantities (to enable type-hinting)
  asset = env.scene[asset_cfg.name]
  if (
    not hasattr(env, "joint_mirror_joints_cache")
    or env.joint_mirror_joints_cache is None
  ):
    # Cache joint positions for all pairs
    env.joint_mirror_joints_cache = [
      [asset.find_joints(name)[0] for name in pair] for pair in mirror_joints
    ]
  reward = torch.zeros(env.num_envs, device=env.device)
  # Iterate over all joint pairs
  for pair in env.joint_mirror_joints_cache:
    # Calculate the difference for each pair and add to the total reward
    reward += torch.sum(
      torch.square(asset.data.joint_pos[:, pair[0]] - asset.data.joint_pos[:, pair[1]]),
      dim=-1,
    )
  if mirror_joints:
    reward *= 1.0 / len(mirror_joints)
  return reward


def undesired_contacts(env, sensor_name: str, threshold: float):
  sensor = env.scene.sensors[sensor_name]
  current_force = cast(torch.Tensor, sensor.data.force)
  force = (
    sensor.data.force_history
    if sensor.data.force_history is not None
    else current_force.unsqueeze(-2)
  )
  return (force.norm(dim=-1).amax(dim=-1) > threshold).any(dim=-1).float()
