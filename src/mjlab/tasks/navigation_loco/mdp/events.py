from __future__ import annotations

from typing import Literal

import torch

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform


def randomize_joint_default_pos(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  pos_distribution_params: tuple[float, float] | None = None,
  operation: Literal["add", "scale", "abs"] = "abs",
  action_name: str = "joint_pos",
):
  """Randomize default joint positions and keep position-action offsets aligned."""
  # extract the used quantities (to enable type-hinting)
  asset = env.scene[asset_cfg.name]

  # save nominal value for export
  if not hasattr(asset.data, "default_joint_pos_nominal"):
    asset.data.default_joint_pos_nominal = asset.data.default_joint_pos[0].clone()

  # resolve environment ids
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if pos_distribution_params is None:
    return

  # resolve joint indices
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    # mjlab resolves explicit indices here instead of retaining the source slice optimization.
    joint_ids_t = torch.arange(asset.num_joints, device=env.device)
  else:
    joint_ids_t = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
  current = asset.data.default_joint_pos[env_ids[:, None], joint_ids_t].clone()
  # sample and apply the requested randomization operation
  samples = sample_uniform(
    pos_distribution_params[0],
    pos_distribution_params[1],
    current.shape,
    env.device,
  )
  if operation == "add":
    randomized = current + samples
  elif operation == "scale":
    randomized = current * samples
  else:
    randomized = samples

  asset.data.default_joint_pos[env_ids[:, None], joint_ids_t] = randomized
  action = env.action_manager.get_term(action_name)
  # update action offsets because they are not auto-updated
  # Changing only MuJoCo qpos0 would not keep the per-environment calibration bias aligned.
  action._offset[env_ids[:, None], joint_ids_t] = randomized


@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
def randomize_rigid_body_com(
  env,
  env_ids: torch.Tensor | None,
  com_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg,
):
  """Randomize rigid-body CoM by adding sampled xyz offsets."""
  # extract the used quantities (to enable type-hinting)
  # resolve environment ids
  # resolve body indices
  # sample random CoM values
  # get the current CoM values (num_assets, num_bodies)
  # randomize CoM in range
  # set the new CoM values
  # mjlab's native event performs the steps above on expanded MuJoCo model fields.
  dr.body_com_offset(
    env,
    env_ids,
    ranges={axis: com_range[key] for axis, key in enumerate(("x", "y", "z"))},
    asset_cfg=asset_cfg,
    operation="add",
  )
