"""Unitree actuator with torque-speed saturation and friction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity

UnitreeCfgT = TypeVar("UnitreeCfgT", bound="UnitreeActuatorCfg")


@dataclass(kw_only=True)
class UnitreeActuatorCfg(IdealPdActuatorCfg):
  """Configuration for a Unitree actuator.

  The actuator applies an asymmetric torque-speed envelope to the PD effort,
  followed by smooth static friction and viscous friction. Command delay uses
  the common mjlab actuator delay fields and samples one lag per environment at
  reset by default.
  """

  effort_limit: float = 1.0e9
  """Simulator-side effort safeguard in N*m."""

  velocity_at_full_effort: float = 1.0e9
  """Knee velocity of the torque-speed curve in rad/s (X1)."""

  velocity_at_zero_effort: float = 1.0e9
  """Zero-effort velocity of the torque-speed curve in rad/s (X2)."""

  drive_effort_limit: float
  """Peak effort when effort and velocity have the same direction in N*m (Y1)."""

  brake_effort_limit: float | None = None
  """Peak effort when effort and velocity have opposite directions in N*m (Y2).

  A value of ``None`` uses ``drive_effort_limit``.
  """

  static_friction: float = 0.0
  """Static friction effort in N*m (Fs)."""

  dynamic_friction: float = 0.0
  """Viscous friction coefficient in N*m*s/rad (Fd)."""

  friction_activation_velocity: float = 0.01
  """Velocity where smooth static friction is substantially active in rad/s (Va)."""

  delay_hold_prob: float = 1.0
  """Probability of holding the reset-sampled lag at each update."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> UnitreeActuator:
    return UnitreeActuator(self, entity, target_ids, target_names)


class UnitreeActuator(IdealPdActuator[UnitreeCfgT], Generic[UnitreeCfgT]):
  """Unitree PD actuator with an asymmetric torque-speed curve.

  The curve uses four hardware parameters::

      Effort limit, N*m
          ^
      Y2--|
          |-----------Y1
          |            |\
          |            | \
      ----+------------|--|----> speed, rad/s
                       X1 X2

  Y1 applies while effort and velocity share a direction. Y2 applies while
  they oppose each other. Full effort is available through X1, then decreases
  linearly to zero at X2.
  """

  def __init__(
    self,
    cfg: UnitreeCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self._joint_vel: torch.Tensor | None = None
    self._drive_effort_limit: torch.Tensor | None = None
    self._brake_effort_limit: torch.Tensor | None = None
    self._velocity_at_full_effort: torch.Tensor | None = None
    self._velocity_at_zero_effort: torch.Tensor | None = None
    self._static_friction: torch.Tensor | None = None
    self._dynamic_friction: torch.Tensor | None = None
    self._friction_activation_velocity: torch.Tensor | None = None

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    shape = (data.nworld, len(self._target_names))
    brake_effort_limit = self.cfg.brake_effort_limit
    if brake_effort_limit is None:
      brake_effort_limit = self.cfg.drive_effort_limit

    self._joint_vel = torch.zeros(shape, device=device)
    self._drive_effort_limit = torch.full(
      shape, self.cfg.drive_effort_limit, dtype=torch.float, device=device
    )
    self._brake_effort_limit = torch.full(
      shape, brake_effort_limit, dtype=torch.float, device=device
    )
    self._velocity_at_full_effort = torch.full(
      shape, self.cfg.velocity_at_full_effort, dtype=torch.float, device=device
    )
    self._velocity_at_zero_effort = torch.full(
      shape, self.cfg.velocity_at_zero_effort, dtype=torch.float, device=device
    )
    self._static_friction = torch.full(
      shape, self.cfg.static_friction, dtype=torch.float, device=device
    )
    self._dynamic_friction = torch.full(
      shape, self.cfg.dynamic_friction, dtype=torch.float, device=device
    )
    self._friction_activation_velocity = torch.full(
      shape,
      self.cfg.friction_activation_velocity,
      dtype=torch.float,
      device=device,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if self._delay_buffer is None:
      return

    lag_state = self._delay_buffer.current_lags
    index = slice(None) if env_ids is None else env_ids
    time_lags = torch.randint(
      low=self.cfg.delay_min_lag,
      high=self.cfg.delay_max_lag + 1,
      size=(lag_state[index].numel(),),
      dtype=lag_state.dtype,
      device=lag_state.device,
    )
    self.set_lags(time_lags, env_ids)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._joint_vel is not None
    assert self._static_friction is not None
    assert self._dynamic_friction is not None
    assert self._friction_activation_velocity is not None

    # Save current joint velocity for torque-speed clipping.
    self._joint_vel[:] = cmd.vel
    effort = super().compute(cmd)

    # Apply the friction model after torque-speed clipping.
    friction = self._static_friction * torch.tanh(
      cmd.vel / self._friction_activation_velocity
    )
    friction += self._dynamic_friction * cmd.vel
    return effort - friction

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self._joint_vel is not None
    assert self._drive_effort_limit is not None
    assert self._brake_effort_limit is not None
    assert self._velocity_at_full_effort is not None

    # Select the peak effort from the effort/velocity direction.
    same_direction = (self._joint_vel * effort) > 0
    max_effort = torch.where(
      same_direction, self._drive_effort_limit, self._brake_effort_limit
    )

    # Apply the linear section of the torque-speed curve above its knee.
    max_effort = torch.where(
      self._joint_vel.abs() < self._velocity_at_full_effort,
      max_effort,
      self._compute_effort_limit(max_effort),
    )
    return torch.clamp(effort, min=-max_effort, max=max_effort)

  def _compute_effort_limit(self, max_effort: torch.Tensor) -> torch.Tensor:
    assert self._joint_vel is not None
    assert self._velocity_at_full_effort is not None
    assert self._velocity_at_zero_effort is not None

    slope = -max_effort / (
      self._velocity_at_zero_effort - self._velocity_at_full_effort
    )
    limit = slope * (self._joint_vel.abs() - self._velocity_at_full_effort) + max_effort
    return limit.clamp_min(0.0)
