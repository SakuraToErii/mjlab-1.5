import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.actuator import BuiltinPositionActuator, BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg as get_stock_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.navigation.config.g1.rl_cfg import navigation_ppo_runner_cfg
from mjlab.tasks.navigation.mdp import (
  ArenaPose2dCommandCfg,
  MazeExitCommandCfg,
  assign_fixed_mixed_arena_layout,
  ensure_fixed_maze_layout,
  load_fixed_maze_layout,
  reset_robot_at_maze_entrance,
)
from mjlab.tasks.navigation.mdp.observations import (
  _pool_height_scan,
  height_scan,
  height_scan_pooled,
)
from mjlab.tasks.navigation.mdp.obstacles import (
  V5_MAX_MIXED_OBSTACLES,
  make_mixed_obstacle_collection,
)
from mjlab.tasks.navigation.mdp.obstacles.mixed_arena_templates import (
  bake_mixed_arena_template,
)
from mjlab.tasks.navigation.mdp.obstacles.mixed_obstacle_layout import (
  MixedObstacleLayoutCfg,
  _build_slot_metadata,
)
from mjlab.tasks.navigation.mdp.pre_trained_policy_action import PreTrainedPolicyAction
from mjlab.tasks.navigation.navigation_env_cfg import (
  make_navigation_env_cfg,
  make_navigation_v2_env_cfg,
  make_navigation_v3_maze_env_cfg,
  make_navigation_v4_fixed_maze_env_cfg,
  make_navigation_v5_compact_env_cfg,
  make_navigation_v5_compact_single_goal_env_cfg,
  make_navigation_v5_mixed_obstacle_env_cfg,
)
from mjlab.tasks.navigation_loco.config.g1.rl_cfg import low_level_ppo_runner_cfg
from mjlab.tasks.navigation_loco.mdp import OffsetGridPatternCfg
from mjlab.tasks.navigation_loco.navigation_loco_env_cfg import (
  get_navigation_g1_robot_cfg,
  make_low_level_actions,
  make_low_level_env_cfg,
  make_low_level_inference_observations,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

_TASK_IDS = {
  "Unitree-G1-29dof-LowLevel",
  "Unitree-G1-29dof-Navigation-HRL-Extension",
  "Unitree-G1-29dof-Navigation-HRL-Baseline",
}


def test_navigation_task_ids_and_play_contract():
  assert {
    task for task in list_tasks() if task.startswith("Unitree-G1-29dof-")
  } == _TASK_IDS
  for task_id in _TASK_IDS:
    train = load_env_cfg(task_id)
    play = load_env_cfg(task_id, play=True)
    assert train.observations["actor"].enable_corruption
    assert not play.observations["actor"].enable_corruption
    assert play.episode_length_s >= 1e9
    assert train.observations.keys() == play.observations.keys()
    assert train.actions.keys() == play.actions.keys()
  extension_play = load_env_cfg("Unitree-G1-29dof-Navigation-HRL-Extension", play=True)
  assert (
    extension_play.events["randomize_obstacles"].params["default_num_active"] == 120
  )


def test_low_level_policy_abi_and_timing():
  group = make_low_level_inference_observations()
  assert list(group.terms) == [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
  ]
  assert group.history_length == 5
  assert not group.enable_corruption
  policy_path = (
    Path(__file__).parents[1] / "src/mjlab/tasks/navigation_loco/models/policy.pt"
  )
  params = dict(torch.jit.load(str(policy_path), map_location="cpu").named_parameters())
  assert tuple(params["mlp.0.weight"].shape) == (512, 480)
  assert tuple(params["mlp.6.weight"].shape) == (29, 128)
  low = make_low_level_env_cfg()
  high = make_navigation_v5_mixed_obstacle_env_cfg()
  assert low.sim.mujoco.timestep == 0.005
  assert low.decimation == 4
  assert high.decimation == 40


def test_pretrained_action_processed_buffer_and_counter_reset():
  term = object.__new__(PreTrainedPolicyAction)
  term.cfg = SimpleNamespace(velocity_clip=((-0.5, 1.0), (-0.5, 0.5), (-0.5, 0.5)))
  term._raw_actions = torch.zeros(2, 3)
  term._processed_actions = torch.zeros(2, 3)
  term.low_level_actions = torch.ones(2, 29)
  term._counter = 17
  reset_calls = []
  term._low_level_obs_manager = SimpleNamespace(
    reset=lambda ids: reset_calls.append(ids)
  )
  term._low_level_action_term = SimpleNamespace(
    reset=lambda ids: reset_calls.append(ids)
  )
  actions = torch.tensor([[2.0, -2.0, 0.25], [-1.0, 1.0, -1.0]])
  term.process_actions(actions)
  assert torch.equal(term.raw_action, actions)
  assert torch.equal(
    term.processed_action,
    torch.tensor([[1.0, -0.5, 0.25], [-0.5, 0.5, -0.5]]),
  )
  term.reset(torch.tensor([0]))
  assert term._counter == 0
  assert torch.count_nonzero(term.low_level_actions[0]) == 0
  assert len(reset_calls) == 2


def test_height_pool_uses_ny_nx_content_order():
  scan = torch.arange(8, dtype=torch.float).view(1, 8)
  pooled = _pool_height_scan(scan, ny=2, nx=4, pool_size=2)
  assert torch.equal(pooled, torch.tensor([[5.0, 7.0]]))


def test_height_scan_pattern_preserves_source_ray_origin_offset():
  assert inspect.signature(height_scan).parameters["offset"].default == 0.5
  assert inspect.signature(height_scan_pooled).parameters["offset"].default == 0.5
  pattern = OffsetGridPatternCfg(
    size=(1.6, 1.0), resolution=0.1, origin_offset=(0.0, 0.0, 20.0)
  )
  offsets, directions = pattern.generate_rays(None, "cpu")
  assert torch.all(offsets[:, 2] == 20.0)
  assert torch.all(directions[:, 2] == -1.0)
  for cfg in (
    make_low_level_env_cfg(),
    make_navigation_v5_mixed_obstacle_env_cfg(),
  ):
    scanner = next(
      sensor for sensor in cfg.scene.sensors if sensor.name == "height_scanner"
    )
    assert isinstance(scanner.pattern, OffsetGridPatternCfg)
    assert scanner.pattern.origin_offset == (0.0, 0.0, 20.0)
    assert scanner.max_distance == 100.0


def test_homework_todo_boundaries_are_all_preserved():
  task_root = Path(__file__).parents[1] / "src/mjlab/tasks"
  text = "\n".join(
    path.read_text()
    for task in ("navigation", "navigation_loco")
    for path in (task_root / task).rglob("*.py")
  )
  for todo_id in range(1, 8):
    assert text.count(f"HOMEWORK_TODO_{todo_id}_START") == 1
    assert text.count(f"HOMEWORK_TODO_{todo_id}_END") == 1


def test_g1_sdk_order_and_locomotion_default_builtin_position_controller():
  cfg = get_navigation_g1_robot_cfg()
  stock = get_stock_g1_robot_cfg()
  assert list(make_low_level_actions()) == ["joint_pos"]
  assert not cfg.sort_actuators
  assert cfg.articulation is not None
  assert stock.articulation is not None
  assert all(
    isinstance(group, BuiltinPositionActuatorCfg)
    for group in cfg.articulation.actuators
  )
  groups = [
    cast(BuiltinPositionActuatorCfg, group) for group in cfg.articulation.actuators
  ]
  stock_groups = [
    cast(BuiltinPositionActuatorCfg, group) for group in stock.articulation.actuators
  ]
  assert cfg.articulation.soft_joint_pos_limit_factor == 1.0
  assert len(groups) == len(stock_groups) == 6
  for group, stock_group in zip(groups, stock_groups, strict=True):
    assert group.target_names_expr == stock_group.target_names_expr
    assert group.transmission_type == stock_group.transmission_type
    assert group.stiffness == stock_group.stiffness
    assert group.damping == stock_group.damping
    assert group.effort_limit == stock_group.effort_limit
    assert group.armature == stock_group.armature
    assert group.frictionloss == stock_group.frictionloss
    assert group.viscous_damping == stock_group.viscous_damping

  entity = cfg.build()
  cursor = 0
  for actuator in entity.actuators:
    assert isinstance(actuator, BuiltinPositionActuator)
    num_targets = len(actuator.target_names)
    control_names = [
      control.name.split("/")[-1]
      for control in entity.spec.actuators[cursor : cursor + num_targets]
    ]
    assert control_names == list(actuator.target_names)
    cursor += num_targets
  assert cursor == 29
  joints = [
    joint.name for joint in entity.spec.joints if joint.name != "floating_base_joint"
  ]
  assert joints[:6] == [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
  ]
  assert len(joints) == 29


@pytest.mark.filterwarnings("ignore:dr.body_mass only randomizes mass")
def test_low_level_runtime_native_foot_friction_and_position_routing():
  cfg = make_low_level_env_cfg(sequential=True)
  assert list(cfg.actions) == ["joint_pos"]
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    action = torch.linspace(-1.0, 1.0, 29).repeat(2, 1)
    observations, reward, _, _, _ = env.step(action)
    actor_observations = cast(torch.Tensor, observations["actor"])
    assert torch.isfinite(actor_observations).all()
    assert torch.isfinite(reward).all()

    robot = env.scene["robot"]
    terrain = env.scene["terrain"]
    foot_geom_ids = robot.indexing.geom_ids[
      [i for i, name in enumerate(robot.geom_names) if "_foot" in name]
    ]
    friction = env.sim.model.geom_friction[:, foot_geom_ids, 0]
    assert env.sim.model.geom_friction.shape[0] == 2
    assert bool(((0.3 <= friction) & (friction <= 1.2)).all())
    assert all(torch.unique(friction[env_id]).numel() == 1 for env_id in range(2))

    # Native G1 feet already outrank terrain, while solver parameters remain nominal.
    assert torch.all(env.sim.model.geom_priority[foot_geom_ids] == 1)
    assert torch.all(env.sim.model.geom_priority[terrain.indexing.geom_ids] == 0)
    foot_solref = env.sim.model.geom_solref[:, foot_geom_ids]
    foot_solimp = env.sim.model.geom_solimp[:, foot_geom_ids]
    assert torch.allclose(
      foot_solref,
      torch.tensor((0.02, 1.0)).expand_as(foot_solref),
    )
    assert torch.allclose(
      foot_solimp,
      torch.tensor((0.9, 0.95, 0.001, 0.5, 2.0)).expand_as(foot_solimp),
    )

    command_term = env.command_manager.get_term("base_velocity")
    assert command_term is not None
    command_cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.common_step_counter = 3000 * 24
    env.curriculum_manager.compute(env_ids)
    assert command_cfg.ranges.lin_vel_x == (-1.5, 2.0)
    assert command_cfg.ranges.ang_vel_z == (-0.7, 0.7)
    env.common_step_counter = 6000 * 24
    env.curriculum_manager.compute(env_ids)
    assert command_cfg.ranges.lin_vel_x == (-2.0, 3.0)

    for actuator in robot.actuators:
      assert isinstance(actuator, BuiltinPositionActuator)
      controls = env.sim.data.ctrl[:, actuator.global_ctrl_ids]
      assert torch.allclose(
        controls,
        robot.data.joint_pos_target[:, actuator.target_ids],
      )
  finally:
    env.close()


def test_low_level_material_command_curriculum_and_play_match_native_flat_g1():
  native = unitree_g1_flat_env_cfg()
  native_play = unitree_g1_flat_env_cfg(play=True)
  base = make_low_level_env_cfg(sequential=False)
  sequential = make_low_level_env_cfg(sequential=True)
  play = make_low_level_env_cfg(sequential=True, play=True)

  native_command = cast(UniformVelocityCommandCfg, native.commands["twist"])
  for cfg in (base, sequential):
    command = cast(UniformVelocityCommandCfg, cfg.commands["base_velocity"])
    assert type(command) is UniformVelocityCommandCfg
    assert command.resampling_time_range == native_command.resampling_time_range
    assert command.rel_standing_envs == native_command.rel_standing_envs
    assert command.rel_heading_envs == native_command.rel_heading_envs
    assert command.rel_forward_envs == native_command.rel_forward_envs
    assert command.heading_command == native_command.heading_command
    assert command.heading_control_stiffness == native_command.heading_control_stiffness
    assert command.ranges == native_command.ranges

    assert set(cfg.curriculum) == set(native.curriculum) == {"command_vel"}
    assert cfg.curriculum["command_vel"].func is native.curriculum["command_vel"].func
    velocity_stages = cfg.curriculum["command_vel"].params["velocity_stages"]
    native_stages = native.curriculum["command_vel"].params["velocity_stages"]
    assert [stage["step"] for stage in velocity_stages] == [0, 3000 * 24, 6000 * 24]
    assert [
      {key: value for key, value in stage.items() if key != "step"}
      for stage in velocity_stages
    ] == [
      {key: value for key, value in stage.items() if key != "step"}
      for stage in native_stages
    ]

    material = cfg.events["foot_friction"]
    native_material = native.events["foot_friction"]
    assert material.func is native_material.func
    assert material.mode == native_material.mode == "startup"
    assert material.params["operation"] == native_material.params["operation"]
    assert material.params["ranges"] == native_material.params["ranges"]
    assert material.params["shared_random"] is True
    assert (
      material.params["asset_cfg"].geom_names
      == native_material.params["asset_cfg"].geom_names
    )
    assert "physics_material" not in cfg.events

    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.terrain_generator is not None
    assert not cfg.scene.terrain.terrain_generator.curriculum

  assert base.events["add_base_mass"].params["ranges"] == (-1.0, 3.0)
  assert sequential.events["add_base_mass"].params["ranges"] == (-0.5, 1.0)
  assert base.events["push_robot"].params["velocity_range"]["x"] == (-0.3, 0.3)

  play_command = cast(UniformVelocityCommandCfg, play.commands["base_velocity"])
  native_play_command = cast(UniformVelocityCommandCfg, native_play.commands["twist"])
  assert play_command.ranges == native_play_command.ranges
  assert play.curriculum == native_play.curriculum == {}
  assert play.scene.terrain is not None
  assert play.scene.terrain.terrain_generator is not None
  assert play.scene.terrain.terrain_generator.num_rows == 2
  assert play.scene.terrain.terrain_generator.num_cols == 10
  assert not play.scene.terrain.terrain_generator.curriculum


def test_fixed_map_reset_event_precedes_robot_reset():
  cfg = make_navigation_v5_compact_single_goal_env_cfg()
  keys = list(cfg.events)
  assert cfg.events["assign_arena_maps"].mode == "reset"
  assert keys.index("assign_arena_maps") < keys.index("reset_base")
  assert cfg.events["assign_arena_maps"].func is assign_fixed_mixed_arena_layout


def test_rl_hyperparameters_and_source_wiring():
  low = low_level_ppo_runner_cfg()
  assert not low.actor.obs_normalization and not low.critic.obs_normalization
  assert low.max_iterations == 50000
  assert low.save_interval == 100
  assert low.experiment_name == "Unitree-G1-29dof-LowLevel"
  baseline = navigation_ppo_runner_cfg(baseline=True)
  assert baseline.clip_actions == 1.0
  for task_id in (
    "Unitree-G1-29dof-Navigation-HRL-Extension",
    "Unitree-G1-29dof-Navigation-HRL-Baseline",
  ):
    assert load_rl_cfg(task_id).experiment_name == baseline.experiment_name


def test_navigation_episode_and_height_clip_variants():
  configs = [
    make_navigation_env_cfg(),
    make_navigation_v2_env_cfg(),
    make_navigation_v3_maze_env_cfg(),
    make_navigation_v4_fixed_maze_env_cfg(),
    make_navigation_v5_mixed_obstacle_env_cfg(),
    make_navigation_v5_compact_env_cfg(),
    make_navigation_v5_compact_single_goal_env_cfg(),
  ]
  assert [cfg.episode_length_s for cfg in configs] == [30.0] * len(configs)
  for cfg in configs[:4]:
    term = cfg.observations["actor"].terms["height_scan"]
    assert term.clip == (-1.0, 1.0)
  assert configs[4].observations["actor"].terms["height_scan"].clip == (-1.5, 1.5)
  compact_play = make_navigation_v5_compact_single_goal_env_cfg(play=True)
  assert compact_play.commands["pose_command"].debug_vis
  assert compact_play.viewer.origin_type.name == "ASSET_ROOT"
  assert compact_play.viewer.elevation == -90.0


def test_maze_command_and_event_helpers_are_native_and_constructible():
  arena_cfg = ArenaPose2dCommandCfg(
    entity_name="robot", resampling_time_range=(30.0, 30.0)
  )
  assert arena_cfg.arena_half_extent == 8.0
  cfg = MazeExitCommandCfg(
    entity_name="robot",
    resampling_time_range=(1e9, 1e9),
    update_goal_on_success=True,
  )
  assert cfg.goal_jitter_radius == 0.25
  assert cfg.ranges.distance == (0.0, 0.0)
  assert callable(ensure_fixed_maze_layout)
  assert callable(load_fixed_maze_layout)
  assert callable(reset_robot_at_maze_entrance)


def test_fixed_template_baking_preserves_global_rng():
  layout_cfg = MixedObstacleLayoutCfg(max_obstacles=8)
  torch.manual_seed(1234)
  expected = torch.rand(4)
  torch.manual_seed(1234)
  first = bake_mixed_arena_template(1001, layout_cfg=layout_cfg)
  actual = torch.rand(4)
  second = bake_mixed_arena_template(1001, layout_cfg=layout_cfg)
  assert torch.equal(actual, expected)
  assert torch.equal(first.centers_xy, second.centers_xy)
  assert torch.equal(first.active_slot_ids, second.active_slot_ids)


def test_mixed_obstacle_topology_and_metadata():
  entities = make_mixed_obstacle_collection()
  assert len(entities) == V5_MAX_MIXED_OBSTACLES == 120
  slot_type, radius, half_extents, heights = _build_slot_metadata("cpu")
  assert slot_type.shape == radius.shape == heights.shape == (120,)
  assert half_extents.shape == (120, 2)
  assert torch.allclose(radius[:30], torch.full((30,), 0.25))
  assert torch.allclose(radius[30:60], torch.full((30,), 0.4))
  assert torch.allclose(radius[60:80], torch.full((20,), 0.55))
  assert torch.allclose(heights[80:100], torch.full((20,), 0.6))
  assert torch.allclose(heights[100:], torch.full((20,), 2.0))
