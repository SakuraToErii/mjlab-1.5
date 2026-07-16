"""Unitree actuator model presets."""

from dataclasses import dataclass

from mjlab.actuator import UnitreeActuatorCfg


@dataclass(kw_only=True)
class UnitreeM107_15ActuatorCfg(UnitreeActuatorCfg):
  """Unitree M107 actuator with a 15:1 gearbox."""

  velocity_at_full_effort: float = 14.0
  velocity_at_zero_effort: float = 25.6
  drive_effort_limit: float = 150.0
  brake_effort_limit: float | None = 182.8
  armature: float | None = 0.063259741


@dataclass(kw_only=True)
class UnitreeM107_24ActuatorCfg(UnitreeActuatorCfg):
  """Unitree M107 actuator with a 24:1 gearbox."""

  velocity_at_full_effort: float = 8.8
  velocity_at_zero_effort: float = 16.0
  drive_effort_limit: float = 240.0
  brake_effort_limit: float | None = 292.5
  armature: float | None = 0.160478022


@dataclass(kw_only=True)
class UnitreeGo2HvActuatorCfg(UnitreeActuatorCfg):
  """Unitree Go2 high-voltage actuator."""

  velocity_at_full_effort: float = 13.5
  velocity_at_zero_effort: float = 30.0
  drive_effort_limit: float = 20.2
  brake_effort_limit: float | None = 23.4


@dataclass(kw_only=True)
class UnitreeN7520_14p3ActuatorCfg(UnitreeActuatorCfg):
  """Unitree N7520 actuator with a 14.3:1 gearbox."""

  # `p` represents the decimal point in the Python identifier.
  velocity_at_full_effort: float = 22.63
  velocity_at_zero_effort: float = 35.52
  drive_effort_limit: float = 71.0
  brake_effort_limit: float | None = 83.3
  static_friction: float = 1.6
  dynamic_friction: float = 0.16

  # | rotor  | 0.489e-4 kg*m^2
  # | gear_1 | 0.098e-4 kg*m^2 | ratio | 4.5
  # | gear_2 | 0.533e-4 kg*m^2 | ratio | 48/22+1
  armature: float | None = 0.01017752


@dataclass(kw_only=True)
class UnitreeN7520_22p5ActuatorCfg(UnitreeActuatorCfg):
  """Unitree N7520 actuator with a 22.5:1 gearbox."""

  # `p` represents the decimal point in the Python identifier.
  velocity_at_full_effort: float = 14.5
  velocity_at_zero_effort: float = 22.7
  drive_effort_limit: float = 111.0
  brake_effort_limit: float | None = 131.0
  static_friction: float = 2.4
  dynamic_friction: float = 0.24

  # | rotor  | 0.489e-4 kg*m^2
  # | gear_1 | 0.109e-4 kg*m^2 | ratio | 4.5
  # | gear_2 | 0.738e-4 kg*m^2 | ratio | 5.0
  armature: float | None = 0.025101925


@dataclass(kw_only=True)
class UnitreeN5010_16ActuatorCfg(UnitreeActuatorCfg):
  """Unitree N5010 actuator with a 16:1 gearbox."""

  velocity_at_full_effort: float = 27.0
  velocity_at_zero_effort: float = 41.5
  drive_effort_limit: float = 9.5
  brake_effort_limit: float | None = 17.0

  # | rotor  | 0.084e-4 kg*m^2
  # | gear_1 | 0.015e-4 kg*m^2 | ratio | 4
  # | gear_2 | 0.068e-4 kg*m^2 | ratio | 4
  armature: float | None = 0.0021812


@dataclass(kw_only=True)
class UnitreeN5020_16ActuatorCfg(UnitreeActuatorCfg):
  """Unitree N5020 actuator with a 16:1 gearbox."""

  velocity_at_full_effort: float = 30.86
  velocity_at_zero_effort: float = 40.13
  drive_effort_limit: float = 24.8
  brake_effort_limit: float | None = 31.9
  static_friction: float = 0.6
  dynamic_friction: float = 0.06

  # | rotor  | 0.139e-4 kg*m^2
  # | gear_1 | 0.017e-4 kg*m^2 | ratio | 46/18+1
  # | gear_2 | 0.169e-4 kg*m^2 | ratio | 56/16+1
  armature: float | None = 0.003609725


@dataclass(kw_only=True)
class UnitreeW4010_25ActuatorCfg(UnitreeActuatorCfg):
  """Unitree W4010 actuator with a 25:1 gearbox."""

  velocity_at_full_effort: float = 15.3
  velocity_at_zero_effort: float = 24.76
  drive_effort_limit: float = 4.8
  brake_effort_limit: float | None = 8.6
  static_friction: float = 0.6
  dynamic_friction: float = 0.06

  # | rotor  | 0.068e-4 kg*m^2
  # | gear_1 |                | ratio | 5
  # | gear_2 |                | ratio | 5
  armature: float | None = 0.00425
