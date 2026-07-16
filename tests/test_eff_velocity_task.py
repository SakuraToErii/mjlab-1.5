"""Contracts for the G1 residual-effort velocity task."""

import re
from dataclasses import asdict
from unittest.mock import Mock

import torch
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from mjlab.actuator import UnitreeActuatorCfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.tasks.eff_velocity import mdp
from mjlab.tasks.eff_velocity.config.g1.action_cfg import (
  EFFORT_ACTION_CLIP,
  EFFORT_ACTION_LIMIT,
  NOMINAL_TORQUE,
  RESIDUAL_ACTION_SCALE,
  g1_residual_effort_action_cfg,
)
from mjlab.tasks.eff_velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_mha_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_rough_mha_env_cfg,
)
from mjlab.tasks.eff_velocity.config.g1.rl_cfg import (
  unitree_g1_ppo_mha_runner_cfg,
  unitree_g1_ppo_runner_cfg,
)
from mjlab.tasks.eff_velocity.config.g1.robot_cfg import (
  EFFORT_STANDING_JOINT_POSITION,
  EFFORT_STANDING_ROOT_HEIGHT,
  get_g1_effort_robot_cfg,
)
from mjlab.tasks.eff_velocity.rl.models import ResidualMhaModel, ResidualMlpModel
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg as unitree_g1_position_flat_env_cfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_rough_env_cfg as unitree_g1_position_rough_env_cfg,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

EFFORT_TASK_IDS = (
  "Mjlab-Effort-Rough-Unitree-G1",
  "Mjlab-Effort-Flat-Unitree-G1",
  "Mjlab-Effort-Rough-Unitree-G1-MHA",
  "Mjlab-Effort-Flat-Unitree-G1-MHA",
)


def _manager_term_contract(terms: dict) -> dict:
  contract = {}
  for name, term in terms.items():
    entry = {
      "func": term.func.__name__,
      "params": term.params,
    }
    if hasattr(term, "weight"):
      entry["weight"] = term.weight
    contract[name] = entry
  return contract


def _last_linear(module: torch.nn.Module) -> torch.nn.Linear:
  layers = [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)]
  assert layers
  return layers[-1]


def test_effort_tasks_are_registered_with_g1_only() -> None:
  task_ids = list_tasks()
  for task_id in EFFORT_TASK_IDS:
    assert task_id in task_ids
  assert all(
    "Go1" not in task_id for task_id in task_ids if task_id.startswith("Mjlab-Effort-")
  )


def test_g1_effort_robot_has_zero_gain_unitree_actuators() -> None:
  cfg = get_g1_effort_robot_cfg()
  assert cfg.init_state.pos == (0.0, 0.0, EFFORT_STANDING_ROOT_HEIGHT)
  assert cfg.init_state.joint_pos == EFFORT_STANDING_JOINT_POSITION
  assert cfg.articulation is not None
  assert len(cfg.articulation.actuators) == 5

  for actuator_cfg in cfg.articulation.actuators:
    assert isinstance(actuator_cfg, UnitreeActuatorCfg)
    assert actuator_cfg.stiffness == 0.0
    assert actuator_cfg.damping == 0.0

  robot = Entity(cfg)
  controlled_joints = [
    joint_name for actuator in robot.actuators for joint_name in actuator.target_names
  ]
  assert len(controlled_joints) == 29
  assert len(set(controlled_joints)) == 29
  assert set(controlled_joints) == set(robot.joint_names)
  assert [len(actuator.target_names) for actuator in robot.actuators] == [
    5,
    4,
    10,
    6,
    4,
  ]


def test_residual_effort_action_matches_motor_limits_and_nominal_torque() -> None:
  robot = Entity(get_g1_effort_robot_cfg())
  joint_names = list(robot.joint_names)
  action_cfg = g1_residual_effort_action_cfg()
  assert isinstance(action_cfg, JointEffortActionCfg)
  assert action_cfg.scale == RESIDUAL_ACTION_SCALE
  assert action_cfg.offset == NOMINAL_TORQUE
  assert action_cfg.clip == EFFORT_ACTION_CLIP
  assert set(NOMINAL_TORQUE) == set(joint_names)

  indices, _, limits = resolve_matching_names_values(EFFORT_ACTION_LIMIT, joint_names)
  assert sorted(indices) == list(range(29))
  assert len(limits) == 29
  assert all(
    RESIDUAL_ACTION_SCALE[expr] == 0.4 * limit
    for expr, limit in EFFORT_ACTION_LIMIT.items()
  )

  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = 2
  env.device = "cpu"
  env.scene = {"robot": robot}
  action = action_cfg.build(env)
  assert action.target_names == joint_names

  zero_action = torch.zeros(2, 29)
  action.process_actions(zero_action)
  expected_nominal = torch.tensor([NOMINAL_TORQUE[name] for name in joint_names])
  torch.testing.assert_close(action._processed_actions[0], expected_nominal)

  action.process_actions(torch.full((2, 29), 100.0))
  expected_upper = torch.tensor(
    [
      next(
        limit
        for joint_expr, limit in EFFORT_ACTION_LIMIT.items()
        if re.fullmatch(joint_expr, name)
      )
      for name in joint_names
    ]
  )
  torch.testing.assert_close(action._processed_actions[0], expected_upper)


def test_rewards_match_mjlab_velocity_and_rough_curriculum_is_preserved() -> None:
  pairs = (
    (unitree_g1_rough_env_cfg(), unitree_g1_position_rough_env_cfg()),
    (unitree_g1_flat_env_cfg(), unitree_g1_position_flat_env_cfg()),
  )
  for effort_cfg, position_cfg in pairs:
    assert _manager_term_contract(effort_cfg.rewards) == _manager_term_contract(
      position_cfg.rewards
    )

  assert _manager_term_contract(
    unitree_g1_rough_env_cfg().curriculum
  ) == _manager_term_contract(unitree_g1_position_rough_env_cfg().curriculum)


def test_flat_effort_curriculum_is_performance_gated() -> None:
  cfg = unitree_g1_flat_env_cfg()
  assert list(cfg.curriculum) == ["flat_effort_stages"]

  term_cfg = cfg.curriculum["flat_effort_stages"]
  assert term_cfg.func is mdp.FlatEffortCurriculum
  assert term_cfg.params["min_episodes"] == 4096
  assert all("step" not in stage for stage in term_cfg.params["stages"])

  stages = term_cfg.params["stages"]
  assert [stage["push_velocity"] for stage in stages] == [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
  ]
  assert stages[0]["rel_standing_envs"] == 1.0
  assert stages[0]["lin_vel_x"] == (0.0, 0.0)
  assert stages[-1]["lin_vel_x"] == (-0.4, 0.6)
  assert cfg.events["push_robot"].interval_range_s == (5.0, 5.0)


def test_mha_variants_only_add_observation_history() -> None:
  pairs = (
    (unitree_g1_rough_env_cfg(), unitree_g1_rough_mha_env_cfg()),
    (unitree_g1_flat_env_cfg(), unitree_g1_flat_mha_env_cfg()),
  )
  for base_cfg, mha_cfg in pairs:
    assert base_cfg.rewards == mha_cfg.rewards
    assert base_cfg.curriculum == mha_cfg.curriculum
    assert base_cfg.observations["actor"].terms == mha_cfg.observations["actor"].terms
    assert base_cfg.observations["critic"].terms == mha_cfg.observations["critic"].terms
    assert mha_cfg.observations["actor"].history_length == 5
    assert mha_cfg.observations["actor"].flatten_history_dim is False
    assert mha_cfg.observations["critic"].history_length == 5
    assert mha_cfg.observations["critic"].flatten_history_dim is True


def test_all_effort_variants_use_joint_effort_actions() -> None:
  for task_id in EFFORT_TASK_IDS:
    cfg = load_env_cfg(task_id)
    assert list(cfg.actions) == ["joint_effort"]
    assert isinstance(cfg.actions["joint_effort"], JointEffortActionCfg)


def test_ppo_configs_use_effort_initialization_and_mha_model() -> None:
  ppo_cfg = unitree_g1_ppo_runner_cfg()
  mha_cfg = unitree_g1_ppo_mha_runner_cfg()

  assert ppo_cfg.actor.class_name.endswith(":ResidualMlpModel")
  assert mha_cfg.actor.class_name.endswith(":ResidualMhaModel")
  assert resolve_callable(ppo_cfg.actor.class_name) is ResidualMlpModel
  assert resolve_callable(mha_cfg.actor.class_name) is ResidualMhaModel
  assert ppo_cfg.actor.distribution_cfg == {
    "class_name": "GaussianDistribution",
    "init_std": 0.1,
    "std_type": "log",
  }
  assert mha_cfg.actor.distribution_cfg == ppo_cfg.actor.distribution_cfg
  assert ppo_cfg.algorithm.entropy_coef == 0.001
  assert mha_cfg.algorithm.entropy_coef == 0.001
  assert ppo_cfg.max_iterations == 50_000
  assert mha_cfg.max_iterations == 10_000

  serialized_mha_cfg = asdict(mha_cfg)
  assert serialized_mha_cfg["actor"]["history_length"] == 5
  assert serialized_mha_cfg["actor"]["num_heads"] == 4


def test_residual_models_center_actor_output_and_export_mha() -> None:
  batch_size = 8
  output_dim = 29
  distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.1,
    "std_type": "log",
  }

  mlp_obs = TensorDict({"actor": torch.zeros(batch_size, 64)}, batch_size=[batch_size])
  mlp = ResidualMlpModel(
    mlp_obs,
    {"actor": ["actor"]},
    "actor",
    output_dim,
    hidden_dims=(64, 32),
    distribution_cfg=distribution_cfg.copy(),
  )
  mlp_output_layer = _last_linear(mlp.mlp)
  torch.testing.assert_close(
    mlp_output_layer.weight.norm(dim=1),
    torch.full((output_dim,), 0.01),
    atol=1.0e-6,
    rtol=1.0e-5,
  )
  torch.testing.assert_close(
    mlp_output_layer.bias, torch.zeros_like(mlp_output_layer.bias)
  )

  mha_obs = TensorDict(
    {"actor": torch.zeros(batch_size, 5, 64)}, batch_size=[batch_size]
  )
  mha = ResidualMhaModel(
    mha_obs,
    {"actor": ["actor"]},
    "actor",
    output_dim,
    hidden_dims=(64, 32),
    distribution_cfg=distribution_cfg.copy(),
    encoder_hidden_dim=32,
    num_heads=4,
  )
  mha_output_layer = _last_linear(mha.mlp)
  torch.testing.assert_close(
    mha_output_layer.weight.norm(dim=1),
    torch.full((output_dim,), 0.01),
    atol=1.0e-6,
    rtol=1.0e-5,
  )
  torch.testing.assert_close(
    mha_output_layer.bias, torch.zeros_like(mha_output_layer.bias)
  )

  expected = mha(mha_obs)
  exported = mha.as_onnx()(mha_obs["actor"])
  assert expected.shape == (batch_size, output_dim)
  torch.testing.assert_close(exported, expected)
