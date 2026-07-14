from __future__ import annotations

from typing import Literal, cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers import ManagerTermBase
from mjlab.managers.event_manager import (
  EventTermCfg,
  RecomputeLevel,
  requires_model_fields,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

# Positive-format MuJoCo solref uses (time constant, damping ratio). These damping
# ratios were calibrated at the task's 0.005 s timestep with MuJoCo-Warp, Newton/10,
# default solimp, and a normal sphere-plane drop. Linear interpolation preserves the
# source's uniformly sampled restitution buckets much better than the ideal continuous
# oscillator formula at this finite timestep.
_RESTITUTION_ANCHORS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3)
_DAMPING_RATIO_ANCHORS = (
  1.0,
  0.856775,
  0.852829,
  0.837897,
  0.813124,
  0.780635,
  0.717285,
  0.640278,
  0.581178,
  0.506413,
)
_CONTACT_TIME_CONSTANT = 0.02
_CONTACT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


def restitution_to_damping_ratio(restitution: torch.Tensor) -> torch.Tensor:
  """Map target restitution to calibrated MuJoCo contact damping ratios."""
  x = torch.as_tensor(_RESTITUTION_ANCHORS, device=restitution.device)
  y = torch.as_tensor(_DAMPING_RATIO_ANCHORS, device=restitution.device)
  restitution = restitution.clamp(min=x[0], max=x[-1])
  upper = torch.searchsorted(x, restitution).clamp(min=1, max=x.numel() - 1)
  lower = upper - 1
  alpha = (restitution - x[lower]) / (x[upper] - x[lower])
  damping_ratio = y[lower] + alpha * (y[upper] - y[lower])
  return torch.where(restitution == 0.0, torch.ones_like(damping_ratio), damping_ratio)


class RandomizeContactMaterial(ManagerTermBase):
  """Pre-sample and assign MuJoCo contact-material buckets per world and geom."""

  # Class-based event terms declare expanded fields directly; the decorator carries a
  # function-only type signature even though EventManager consumes the same metadata.
  model_fields = ("geom_friction", "geom_solref", "geom_solimp")
  recompute = RecomputeLevel.none
  contact_time_constant = _CONTACT_TIME_CONSTANT
  contact_solimp = _CONTACT_SOLIMP

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    friction_range = cast(tuple[float, float], cfg.params["friction_range"])
    restitution_range = cast(tuple[float, float], cfg.params["restitution_range"])
    num_buckets = cast(int, cfg.params["num_buckets"])
    # Sample material properties once during initialization. Afterwards each world/geom
    # receives a random bucket ID, matching the source material randomizer's lifecycle.
    self.friction_buckets = sample_uniform(
      friction_range[0], friction_range[1], (num_buckets,), env.device
    )
    self.restitution_buckets = sample_uniform(
      restitution_range[0], restitution_range[1], (num_buckets,), env.device
    )
    self.damping_ratio_buckets = restitution_to_damping_ratio(self.restitution_buckets)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
    asset_cfg: SceneEntityCfg,
  ) -> None:
    del friction_range, restitution_range
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
      env_ids = env_ids.to(device=env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
    env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    bucket_ids = torch.randint(
      num_buckets, env_grid.shape, device=env.device, dtype=torch.long
    )

    env.sim.model.geom_friction[env_grid, geom_grid, 0] = self.friction_buckets[
      bucket_ids
    ]
    env.sim.model.geom_solref[env_grid, geom_grid, 0] = _CONTACT_TIME_CONSTANT
    env.sim.model.geom_solref[env_grid, geom_grid, 1] = self.damping_ratio_buckets[
      bucket_ids
    ]
    env.sim.model.geom_solimp[env_grid, geom_grid] = torch.as_tensor(
      _CONTACT_SOLIMP, device=env.device
    )


def randomize_joint_default_pos(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  pos_distribution_params: tuple[float, float] | None = None,
  operation: Literal["add", "scale", "abs"] = "abs",
  action_name: str = "JointPositionAction",
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
