import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import pytest
import torch

from mjlab.actuator import (
  BuiltinPdActuator,
  BuiltinPdActuatorCfg,
  BuiltinPositionActuatorCfg,
)
from mjlab.asset_zoo.robots import get_g1_robot_cfg as get_stock_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
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
from mjlab.tasks.navigation_loco.mdp import (
  OffsetGridPatternCfg,
  RandomizeContactMaterial,
  UniformLevelVelocityCommandCfg,
  restitution_to_damping_ratio,
)
from mjlab.tasks.navigation_loco.mdp.curriculums import (
  _apply_tracking_std_from_level,
)
from mjlab.tasks.navigation_loco.navigation_loco_env_cfg import (
  get_navigation_g1_robot_cfg,
  make_low_level_env_cfg,
  make_low_level_inference_observations,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

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


def test_global_curriculum_mutates_live_reward_cfg_without_setter():
  lin_cfg = SimpleNamespace(params={"std": 1.0})
  ang_cfg = SimpleNamespace(params={"std": 1.0})
  reward_manager = SimpleNamespace(
    get_term_cfg=lambda name: lin_cfg if name == "lin" else ang_cfg
  )
  env = SimpleNamespace(reward_manager=reward_manager)
  _apply_tracking_std_from_level(
    cast(Any, env),
    level=5,
    num_levels=5,
    lin_var_initial=0.25,
    lin_var_final=0.18,
    ang_var_initial=0.35,
    ang_var_final=0.15,
    lin_term_name="lin",
    ang_term_name="ang",
  )
  assert lin_cfg.params["std"] == round(0.18**0.5, 3)
  assert ang_cfg.params["std"] == round(0.15**0.5, 3)


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


def test_g1_sdk_order_and_locomotion_default_builtin_pd_controller():
  cfg = get_navigation_g1_robot_cfg()
  stock = get_stock_g1_robot_cfg()
  assert not cfg.sort_actuators
  assert cfg.articulation is not None
  assert stock.articulation is not None
  groups = [cast(BuiltinPdActuatorCfg, group) for group in cfg.articulation.actuators]
  stock_groups = [
    cast(BuiltinPositionActuatorCfg, group) for group in stock.articulation.actuators
  ]
  assert cfg.articulation.soft_joint_pos_limit_factor == 1.0
  assert len(groups) == len(stock_groups) == 6
  for group, stock_group in zip(groups, stock_groups, strict=True):
    assert group.target_names_expr == stock_group.target_names_expr
    assert group.stiffness == stock_group.stiffness
    assert group.damping == stock_group.damping
    assert group.effort_limit == stock_group.effort_limit
    assert group.armature == stock_group.armature

  entity = cfg.build()
  cursor = 0
  for actuator in entity.actuators:
    assert isinstance(actuator, BuiltinPdActuator)
    num_targets = len(actuator.target_names)
    control_names = [
      control.name.split("/")[-1]
      for control in entity.spec.actuators[cursor : cursor + 2 * num_targets]
    ]
    assert control_names == [
      *(f"{name}_pd_pos" for name in actuator.target_names),
      *(f"{name}_pd_vel" for name in actuator.target_names),
    ]
    cursor += 2 * num_targets
  assert cursor == 58
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


def test_restitution_mapping_matches_mujoco_warp_normal_drop():
  targets = torch.tensor((0.0, 0.05, 0.1, 0.2, 0.3))
  num_worlds = len(targets)
  damping_ratios = restitution_to_damping_ratio(targets)
  spec_xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <body name="ball" pos="0 0 0.5">
          <freejoint/>
          <geom name="ball_geom" type="sphere" size="0.05" mass="1"
                priority="1" friction="0.6 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>
  """
  scene = Scene(
    SceneCfg(
      num_envs=num_worlds,
      env_spacing=3.0,
      entities={"ball": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(spec_xml))},
    ),
    device="cpu",
  )
  simulation = Simulation(
    num_envs=num_worlds,
    cfg=SimulationCfg(
      nconmax=100,
      njmax=100,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    model=scene.compile(),
    device="cpu",
  )
  scene.initialize(simulation.mj_model, simulation.model, simulation.data)
  simulation.expand_model_fields(("geom_solref", "geom_solimp"))
  ball_geom_id = simulation.mj_model.geom("ball/ball_geom").id
  simulation.model.geom_solref[:, ball_geom_id, 0] = (
    RandomizeContactMaterial.contact_time_constant
  )
  simulation.model.geom_solref[:, ball_geom_id, 1] = damping_ratios
  simulation.model.geom_solimp[:, ball_geom_id] = torch.tensor(
    RandomizeContactMaterial.contact_solimp
  )

  rising = torch.zeros(num_worlds, dtype=torch.bool)
  reached_apex = torch.zeros_like(rising)
  apex = torch.full((num_worlds,), 0.05)
  for _ in range(300):
    simulation.step()
    height = simulation.data.qpos[:, 2]
    vertical_velocity = simulation.data.qvel[:, 2]
    active = ~reached_apex
    rising |= active & (vertical_velocity > 0.0)
    apex = torch.where(rising & active, torch.maximum(apex, height), apex)
    reached_apex |= rising & (vertical_velocity <= 0.0)

  measured = torch.sqrt(torch.clamp((apex - 0.05) / (0.5 - 0.05), min=0.0))
  assert torch.isfinite(simulation.data.qpos).all()
  assert torch.allclose(measured, targets, atol=0.02, rtol=0.0)


@pytest.mark.filterwarnings("ignore:dr.body_mass only randomizes mass")
def test_low_level_runtime_material_expansion_and_pd_routing():
  cfg = make_low_level_env_cfg(sequential=True)
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
    geom_ids = robot.indexing.geom_ids
    friction = env.sim.model.geom_friction[:, geom_ids, 0]
    solref = env.sim.model.geom_solref[:, geom_ids]
    solimp = env.sim.model.geom_solimp[:, geom_ids]
    assert env.sim.model.geom_friction.shape[0] == 2
    assert env.sim.model.geom_solref.shape[0] == 2
    assert env.sim.model.geom_solimp.shape[0] == 2
    assert bool(((0.5 <= friction) & (friction <= 1.0)).all())
    assert torch.all(solref[..., 0] == 0.02)
    assert bool(((0.780635 <= solref[..., 1]) & (solref[..., 1] <= 1.0)).all())
    expected_solimp = torch.tensor(RandomizeContactMaterial.contact_solimp)
    assert torch.allclose(solimp, expected_solimp.expand_as(solimp))

    for actuator in robot.actuators:
      assert isinstance(actuator, BuiltinPdActuator)
      num_targets = len(actuator.target_names)
      controls = env.sim.data.ctrl[:, actuator.global_ctrl_ids]
      assert torch.allclose(
        controls[:, :num_targets],
        robot.data.joint_pos_target[:, actuator.target_ids],
      )
      assert torch.allclose(
        controls[:, num_targets:],
        robot.data.joint_vel_target[:, actuator.target_ids],
      )
  finally:
    env.close()


def test_low_level_base_sequential_and_play_variants():
  base = make_low_level_env_cfg(sequential=False)
  sequential = make_low_level_env_cfg(sequential=True)
  play = make_low_level_env_cfg(sequential=True, play=True)
  base_command = cast(UniformLevelVelocityCommandCfg, base.commands["base_velocity"])
  sequential_command = cast(
    UniformLevelVelocityCommandCfg, sequential.commands["base_velocity"]
  )
  play_command = cast(UniformLevelVelocityCommandCfg, play.commands["base_velocity"])
  assert base_command.adaptive_sampling
  assert not sequential_command.adaptive_sampling
  assert set(base.curriculum) == {"terrain_levels", "global_low_level_curriculum"}
  assert set(sequential.curriculum) == {"sequential_low_level_curriculum"}
  assert base.scene.terrain is not None
  assert base.scene.terrain.terrain_generator is not None
  assert sequential.scene.terrain is not None
  assert sequential.scene.terrain.terrain_generator is not None
  assert base.scene.terrain.terrain_generator.curriculum
  assert not sequential.scene.terrain.terrain_generator.curriculum
  assert base.scene.entities["robot"].collisions[0].priority == 1
  assert sequential.scene.entities["robot"].collisions[0].priority == 1
  base_material = base.events["physics_material"]
  sequential_material = sequential.events["physics_material"]
  assert base_material.func is RandomizeContactMaterial
  assert sequential_material.func is RandomizeContactMaterial
  assert base_material.params["friction_range"] == (0.3, 1.2)
  assert base_material.params["restitution_range"] == (0.0, 0.3)
  assert sequential_material.params["friction_range"] == (0.5, 1.0)
  assert sequential_material.params["restitution_range"] == (0.0, 0.1)
  assert base_material.params["num_buckets"] == 64
  assert base.events["add_base_mass"].params["ranges"] == (-1.0, 3.0)
  assert sequential.events["add_base_mass"].params["ranges"] == (-0.5, 1.0)
  assert base.events["push_robot"].params["velocity_range"]["x"] == (-0.3, 0.3)
  # Source sequential play starts with adaptive sampling disabled; the forced
  # max curriculum level enables phase 2 during the first reset.
  assert not play_command.adaptive_sampling
  assert play.scene.terrain is not None
  assert play.scene.terrain.terrain_generator is not None
  assert play.scene.terrain.terrain_generator.num_rows == 2
  assert play.scene.terrain.terrain_generator.num_cols == 10
  assert set(play.curriculum) == {"sequential_low_level_curriculum"}
  assert play.curriculum["sequential_low_level_curriculum"].params["forced_level"] == 5


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
