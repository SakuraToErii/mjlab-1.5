"""Tests for the Unitree torque-speed actuator."""

import pytest
import torch
from conftest import (
  create_entity_with_actuator,
  get_test_device,
  initialize_entity,
  load_fixture_xml,
)

from mjlab.actuator import UnitreeActuator, UnitreeActuatorCfg
from mjlab.asset_zoo.robots.unitree_actuators import (
  UnitreeGo2HvActuatorCfg,
  UnitreeM107_15ActuatorCfg,
  UnitreeM107_24ActuatorCfg,
  UnitreeN5010_16ActuatorCfg,
  UnitreeN5020_16ActuatorCfg,
  UnitreeN7520_14p3ActuatorCfg,
  UnitreeN7520_22p5ActuatorCfg,
  UnitreeW4010_25ActuatorCfg,
)


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def robot_xml():
  return load_fixture_xml("floating_base_articulated")


def make_actuator_cfg(**kwargs) -> UnitreeActuatorCfg:
  return UnitreeActuatorCfg(
    target_names_expr=("joint1",),
    stiffness=0.0,
    damping=0.0,
    velocity_at_full_effort=10.0,
    velocity_at_zero_effort=20.0,
    drive_effort_limit=100.0,
    brake_effort_limit=120.0,
    **kwargs,
  )


@pytest.mark.parametrize(
  ("velocity", "effort", "expected"),
  [
    (5.0, 200.0, 100.0),
    (15.0, 200.0, 50.0),
    (15.0, -200.0, -60.0),
    (20.0, 200.0, 0.0),
    (-15.0, -200.0, -50.0),
    (-15.0, 200.0, 60.0),
  ],
)
def test_unitree_torque_speed_curve(device, robot_xml, velocity, effort, expected):
  entity = create_entity_with_actuator(robot_xml, make_actuator_cfg())
  entity, sim = initialize_entity(entity, device)

  entity.write_joint_state_to_sim(
    position=torch.tensor([[0.0]], device=device),
    velocity=torch.tensor([[velocity]], device=device),
    joint_ids=slice(0, 1),
  )
  entity.set_joint_effort_target(
    torch.tensor([[effort]], device=device), joint_ids=slice(0, 1)
  )
  entity.write_data_to_sim()

  assert torch.allclose(
    sim.data.ctrl[0], torch.tensor([expected], device=device), atol=1e-5
  )


def test_unitree_applies_friction_after_curve_clipping(device, robot_xml):
  cfg = make_actuator_cfg(
    static_friction=2.0,
    dynamic_friction=0.5,
    friction_activation_velocity=1.0,
  )
  entity = create_entity_with_actuator(robot_xml, cfg)
  entity, sim = initialize_entity(entity, device)

  entity.write_joint_state_to_sim(
    position=torch.tensor([[0.0]], device=device),
    velocity=torch.tensor([[1.0]], device=device),
    joint_ids=slice(0, 1),
  )
  entity.set_joint_effort_target(
    torch.tensor([[200.0]], device=device), joint_ids=slice(0, 1)
  )
  entity.write_data_to_sim()

  expected = 100.0 - 2.0 * torch.tanh(torch.tensor(1.0)) - 0.5
  assert torch.allclose(sim.data.ctrl[0], expected.to(device).unsqueeze(0))


def test_unitree_delay_is_sampled_on_reset_and_held(device, robot_xml):
  cfg = make_actuator_cfg(delay_min_lag=1, delay_max_lag=1)
  entity = create_entity_with_actuator(robot_xml, cfg)
  entity, sim = initialize_entity(entity, device)
  entity.reset()

  actuator = entity.actuators[0]
  assert isinstance(actuator, UnitreeActuator)
  assert actuator._delay_buffer is not None
  assert actuator._delay_buffer.current_lags.item() == 1

  for effort in (2.0, 4.0):
    entity.set_joint_effort_target(
      torch.tensor([[effort]], device=device), joint_ids=slice(0, 1)
    )
    entity.write_data_to_sim()

  assert actuator._delay_buffer.current_lags.item() == 1
  assert torch.allclose(sim.data.ctrl[0], torch.tensor([2.0], device=device))


@pytest.mark.parametrize(
  ("cfg_type", "x1", "x2", "y1", "y2", "armature", "fs", "fd"),
  [
    (UnitreeM107_15ActuatorCfg, 14.0, 25.6, 150.0, 182.8, 0.063259741, 0.0, 0.0),
    (UnitreeM107_24ActuatorCfg, 8.8, 16.0, 240.0, 292.5, 0.160478022, 0.0, 0.0),
    (UnitreeGo2HvActuatorCfg, 13.5, 30.0, 20.2, 23.4, None, 0.0, 0.0),
    (UnitreeN7520_14p3ActuatorCfg, 22.63, 35.52, 71.0, 83.3, 0.01017752, 1.6, 0.16),
    (UnitreeN7520_22p5ActuatorCfg, 14.5, 22.7, 111.0, 131.0, 0.025101925, 2.4, 0.24),
    (UnitreeN5010_16ActuatorCfg, 27.0, 41.5, 9.5, 17.0, 0.0021812, 0.0, 0.0),
    (UnitreeN5020_16ActuatorCfg, 30.86, 40.13, 24.8, 31.9, 0.003609725, 0.6, 0.06),
    (UnitreeW4010_25ActuatorCfg, 15.3, 24.76, 4.8, 8.6, 0.00425, 0.6, 0.06),
  ],
)
def test_unitree_motor_presets(cfg_type, x1, x2, y1, y2, armature, fs, fd):
  cfg = cfg_type(target_names_expr=("joint1",), stiffness=0.0, damping=0.0)

  assert cfg.velocity_at_full_effort == x1
  assert cfg.velocity_at_zero_effort == x2
  assert cfg.drive_effort_limit == y1
  assert cfg.brake_effort_limit == y2
  assert cfg.armature == armature
  assert cfg.static_friction == fs
  assert cfg.dynamic_friction == fd
