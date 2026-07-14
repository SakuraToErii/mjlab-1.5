"""Low-level G1 locomotion task used by hierarchical navigation."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import cast

from mjlab.actuator import BuiltinPdActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import flat, random_rough
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from . import mdp

NUM_LEVELS = 5


def get_navigation_g1_robot_cfg() -> EntityCfg:
  """G1 with the source policy pose/ABI and mjlab locomotion PD parameters."""
  cfg = get_g1_robot_cfg()
  cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.8),
    joint_pos={
      ".*_hip_pitch_joint": -0.1,
      ".*_knee_joint": 0.3,
      ".*_ankle_pitch_joint": -0.2,
      ".*_shoulder_pitch_joint": 0.3,
      "left_shoulder_roll_joint": 0.25,
      "right_shoulder_roll_joint": -0.25,
      ".*_elbow_joint": 0.97,
      "left_wrist_roll_joint": 0.15,
      "right_wrist_roll_joint": -0.15,
    },
    joint_vel={".*": 0.0},
  )
  # Use the same motor-derived gains, effort caps, reflected inertias and joint
  # grouping as mjlab's G1 locomotion task, but express the controller as native
  # paired position/velocity PD elements. The source velocity_limit_sim has no
  # BuiltinPd equivalent and is intentionally omitted for low-level retraining.
  native_articulation = cast(EntityArticulationInfoCfg, cfg.articulation)
  native_actuators = cast(
    tuple[BuiltinPositionActuatorCfg, ...], native_articulation.actuators
  )
  cfg.articulation = EntityArticulationInfoCfg(
    soft_joint_pos_limit_factor=1.0,
    actuators=tuple(
      BuiltinPdActuatorCfg(
        target_names_expr=actuator.target_names_expr,
        transmission_type=actuator.transmission_type,
        stiffness=actuator.stiffness,
        damping=actuator.damping,
        effort_limit=actuator.effort_limit,
        armature=actuator.armature,
        frictionloss=actuator.frictionloss,
        viscous_damping=actuator.viscous_damping,
      )
      for actuator in native_actuators
    ),
  )
  # BuiltinPd emits [position targets..., velocity targets...] per actuator group.
  # Keep group declaration order so each paired control block remains contiguous;
  # policy actions are still resolved by the robot's 29-DoF joint-definition order.
  cfg.sort_actuators = False
  return cfg


def make_low_level_observations() -> dict[str, ObservationGroupCfg]:
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      scale=0.2,
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
    ),
    "velocity_commands": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "base_velocity"}
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel_rel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      scale=0.05,
      noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }
  critic_terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel, scale=0.2),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "velocity_commands": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "base_velocity"}
    ),
    "joint_pos_rel": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel_rel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, scale=0.05),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }
  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      history_length=5,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      history_length=5,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def make_low_level_inference_observations() -> ObservationGroupCfg:
  """Exact deterministic 480-wide input group used by the packaged policy."""
  # 提示：必须 deepcopy 低层训练时的 actor observation group，不能重新猜测输入顺序。
  # 推理时关闭 corruption 和四个本体感知项的 noise，并保留训练时的 5 帧历史。
  # >>> HOMEWORK_TODO_6_START
  # deepcopy 继承低层训练 actor group 的 term 顺序/维度/func/params，逐字段对齐 TorchScript checkpoint 输入。
  group = deepcopy(make_low_level_observations()["actor"])
  # 推理确定性：关闭 corruption，清空训练时仅有的四项本体感知 noise。
  group.enable_corruption = False
  for term_name in (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
  ):
    group.terms[term_name].noise = None
  # 显式锁定 group 结构：term 拼接 + 5 帧历史，匹配 checkpoint 输入宽度与格式。
  group.history_length = 5
  group.concatenate_terms = True
  return group
  # <<< HOMEWORK_TODO_6_END


def make_low_level_actions() -> dict[str, ActionTermCfg]:
  return {
    "JointPositionAction": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=0.25, use_default_offset=True
    )
  }


def make_low_level_commands(
  *, adaptive_sampling: bool = False
) -> dict[str, CommandTermCfg]:
  return {
    "base_velocity": mdp.UniformLevelVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      rel_standing_envs=0.08,
      rel_heading_envs=1.0,
      heading_command=False,
      debug_vis=True,
      ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
      ),
      limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5)
      ),
      adaptive_sampling=adaptive_sampling,
      num_bins={"lin_vel_x": 15, "lin_vel_y": 10, "ang_vel_z": 10},
      ema_alpha=0.1,
      min_bin_probability=0.02,
      sampling_temperature=1.0,
      warmup_resamples=2,
    )
  }


def make_low_level_events(*, sequential: bool = True) -> dict[str, EventTermCfg]:
  friction_range = (0.5, 1.0) if sequential else (0.3, 1.2)
  restitution_range = (0.0, 0.1) if sequential else (0.0, 0.3)
  mass_range = (-0.5, 1.0) if sequential else (-1.0, 3.0)
  push = 0.15 if sequential else 0.3
  return {
    # The source samples 64 material buckets once and assigns one bucket to each
    # environment/shape. MuJoCo has one sliding-friction coefficient, so use the
    # source dynamic-friction range and map restitution to calibrated contact damping.
    "physics_material": EventTermCfg(
      func=mdp.RandomizeContactMaterial,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
        "friction_range": friction_range,
        "restitution_range": restitution_range,
        "num_buckets": 64,
      },
    ),
    "add_base_mass": EventTermCfg(
      func=dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "ranges": mass_range,
        "operation": "add",
      },
    ),
    "add_joint_default_pos": EventTermCfg(
      func=mdp.randomize_joint_default_pos,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "pos_distribution_params": (-0.01, 0.01),
        "operation": "add",
      },
    ),
    "base_com": EventTermCfg(
      func=mdp.randomize_rigid_body_com,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.025, 0.025)},
      },
    ),
    "base_external_force_torque": EventTermCfg(
      func=envs_mdp.apply_external_force_torque,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "force_range": (0.0, 0.0),
        "torque_range": (0.0, 0.0),
      },
    ),
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (-1.0, 1.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 7.0),
      params={"velocity_range": {"x": (-push, push), "y": (-push, push)}},
    ),
  }


def make_low_level_rewards(*, sequential: bool = True) -> dict[str, RewardTermCfg]:
  feet = SceneEntityCfg(
    "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
  )
  return {
    "track_lin_vel_xy": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={
        "command_name": "base_velocity",
        "std": math.sqrt(0.20 if sequential else 0.25),
      },
    ),
    "track_ang_vel_z": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    ),
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.10),
    "base_linear_velocity": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
    "base_angular_velocity": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
    "joint_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-0.001),
    "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-5.0),
    "energy": RewardTermCfg(func=mdp.energy, weight=-2e-5),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still, weight=-0.1, params={"command_name": "base_velocity"}
    ),
    "joint_deviation_arms": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.1,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*")
        )
      },
    ),
    "joint_deviation_waists": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist.*",))},
    ),
    "joint_deviation_legs": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-1.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(".*_hip_roll_joint", ".*_hip_yaw_joint")
        )
      },
    ),
    "flat_orientation_l2": RewardTermCfg(
      func=envs_mdp.flat_orientation_l2, weight=-5.0
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78}
    ),
    "gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.8,
        "offset": [0.0, 0.5],
        "threshold": 0.55,
        "command_name": "base_velocity",
        "sensor_name": "feet_ground_contact",
      },
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide,
      weight=-0.2,
      params={"asset_cfg": feet, "sensor_name": "feet_ground_contact"},
    ),
    "feet_clearance": RewardTermCfg(
      func=mdp.foot_clearance_reward,
      weight=1.0,
      params={"std": 0.05, "tanh_mult": 2.0, "target_height": 0.1, "asset_cfg": feet},
    ),
    "feet_contact_without_cmd": RewardTermCfg(
      func=mdp.feet_contact_without_cmd,
      weight=0.05,
      params={"sensor_name": "feet_ground_contact", "command_name": "base_velocity"},
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-1.0,
      params={"threshold": 1.0, "sensor_name": "undesired_contacts"},
    ),
  }


def make_low_level_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum, params={"minimum_height": 0.2}
    ),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": 0.8}
    ),
  }


def make_low_level_curriculum(
  *, sequential: bool = True
) -> dict[str, CurriculumTermCfg]:
  if sequential:
    return {
      "sequential_low_level_curriculum": CurriculumTermCfg(
        func=mdp.sequential_low_level_curriculum,
        params={
          "command_name": "base_velocity",
          "num_levels": NUM_LEVELS,
          "lin_promotion_threshold": 0.75,
          "ang_promotion_threshold": 0.5,
        },
      )
    }
  return {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel, params={"command_name": "base_velocity"}
    ),
    "global_low_level_curriculum": CurriculumTermCfg(
      func=mdp.global_low_level_curriculum,
      params={
        "command_name": "base_velocity",
        "num_levels": NUM_LEVELS,
        "promotion_ratio": 0.70,
        "lin_var_initial": 0.25,
        "lin_var_final": 0.18,
        "ang_var_initial": 0.35,
        "ang_var_final": 0.15,
      },
    ),
  }


def make_low_level_env_cfg(
  *, sequential: bool = True, play: bool = False
) -> ManagerBasedRlEnvCfg:
  terrain = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    curriculum=not sequential,
    sub_terrains={
      "flat": flat(proportion=0.6),
      "random_rough": random_rough(
        proportion=0.4, noise_range=(0.01, 0.03), noise_step=0.01
      ),
    },
  )
  scan = RayCastSensorCfg(
    name="height_scanner",
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    ray_alignment="yaw",
    pattern=mdp.OffsetGridPatternCfg(
      size=(1.6, 1.0), resolution=0.1, origin_offset=(0.0, 0.0, 20.0)
    ),
    max_distance=100.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )
  feet_sensor = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  undesired_sensor = ContactSensorCfg(
    name="undesired_contacts",
    primary=ContactMatch(mode="body", pattern=r"^(?!.*ankle.*).*$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  robot_cfg = get_navigation_g1_robot_cfg()
  # The source terrain uses friction=1 with multiply combination, so the sampled
  # robot coefficient is the effective contact coefficient. MuJoCo combines
  # equal-priority friction with max(); giving robot collision geoms higher
  # priority makes the task's randomized sliding coefficient win instead.
  collision_cfg = deepcopy(robot_cfg.collisions[0])
  collision_cfg.priority = 1
  robot_cfg.collisions = (collision_cfg,)
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator", terrain_generator=terrain, max_init_terrain_level=3
      ),
      entities={"robot": robot_cfg},
      sensors=(scan, feet_sensor, undesired_sensor),
      num_envs=4096,
      env_spacing=2.5,
    ),
    observations=make_low_level_observations(),
    actions=make_low_level_actions(),
    commands=make_low_level_commands(adaptive_sampling=not sequential),
    events=make_low_level_events(sequential=sequential),
    rewards=make_low_level_rewards(sequential=sequential),
    terminations=make_low_level_terminations(),
    curriculum=make_low_level_curriculum(sequential=sequential),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
    ),
    sim=SimulationCfg(
      nconmax=70,
      njmax=1500,
      contact_sensor_maxmatch=500,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
  if play:
    # Keep mjlab's registered-play safety convention (infinite deterministic rollout,
    # no external pushes) while preserving the source command/curriculum state.
    cfg.scene.num_envs = 32
    cfg.episode_length_s = 1e9
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    terrain.num_rows = 2
    terrain.num_cols = 10
    terrain.curriculum = not sequential
    base_velocity_command = cast(
      mdp.UniformLevelVelocityCommandCfg, cfg.commands["base_velocity"]
    )
    base_velocity_command.ranges = deepcopy(base_velocity_command.limit_ranges)
    curriculum_name = (
      "sequential_low_level_curriculum" if sequential else "global_low_level_curriculum"
    )
    cfg.curriculum[curriculum_name].params["forced_level"] = NUM_LEVELS
  return cfg
