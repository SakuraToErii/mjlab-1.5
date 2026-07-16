"""Low-level G1 locomotion task used by hierarchical navigation."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import cast

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
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
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import flat, random_rough
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from . import mdp

_G1_FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def get_navigation_g1_robot_cfg() -> EntityCfg:
  """G1 with the native flat-velocity default pose and position actuators."""
  cfg = get_g1_robot_cfg()
  # Keep the stock G1 knees-bent initial pose and BuiltinPosition actuator groups,
  # including their motor-derived gains, effort caps, reflected inertias and order.
  # Only the migrated task's soft joint-position limit factor remains task-specific.
  native_articulation = cast(EntityArticulationInfoCfg, cfg.articulation)
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=native_articulation.actuators,
    soft_joint_pos_limit_factor=1.0,
  )
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
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }


def make_low_level_commands() -> dict[str, CommandTermCfg]:
  return {
    "base_velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      rel_forward_envs=0.2,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }


def make_low_level_events(*, sequential: bool = True) -> dict[str, EventTermCfg]:
  mass_range = (-0.5, 1.0) if sequential else (-1.0, 3.0)
  return {
    "foot_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=_G1_FOOT_GEOM_NAMES),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
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
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        }
      },
    ),
  }


def make_low_level_rewards() -> dict[str, RewardTermCfg]:
  """Match the native G1 flat velocity rewards, without height-command tracking."""
  site_names = ("left_foot", "right_foot")
  return {
    "track_linear_velocity": RewardTermCfg(
      func=velocity_mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=velocity_mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=velocity_mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    ),
    "pose": RewardTermCfg(
      func=velocity_mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "base_velocity",
        "std_standing": {".*": 0.05},
        "std_walking": {
          r".*hip_pitch.*": 0.3,
          r".*hip_roll.*": 0.15,
          r".*hip_yaw.*": 0.15,
          r".*knee.*": 0.35,
          r".*ankle_pitch.*": 0.25,
          r".*ankle_roll.*": 0.1,
          r".*waist_yaw.*": 0.2,
          r".*waist_roll.*": 0.08,
          r".*waist_pitch.*": 0.1,
          r".*shoulder_pitch.*": 0.15,
          r".*shoulder_roll.*": 0.15,
          r".*shoulder_yaw.*": 0.1,
          r".*elbow.*": 0.15,
          r".*wrist.*": 0.3,
        },
        "std_running": {
          r".*hip_pitch.*": 0.5,
          r".*hip_roll.*": 0.2,
          r".*hip_yaw.*": 0.2,
          r".*knee.*": 0.6,
          r".*ankle_pitch.*": 0.35,
          r".*ankle_roll.*": 0.15,
          r".*waist_yaw.*": 0.3,
          r".*waist_roll.*": 0.08,
          r".*waist_pitch.*": 0.2,
          r".*shoulder_pitch.*": 0.5,
          r".*shoulder_roll.*": 0.2,
          r".*shoulder_yaw.*": 0.15,
          r".*elbow.*": 0.35,
          r".*wrist.*": 0.3,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=velocity_mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    ),
    "angular_momentum": RewardTermCfg(
      func=velocity_mdp.angular_momentum_penalty,
      weight=-0.02,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=velocity_mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=velocity_mdp.action_rate_l2, weight=-0.1),
    "air_time": RewardTermCfg(
      func=velocity_mdp.feet_air_time,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "base_velocity",
        "command_threshold": 0.5,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=velocity_mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": "base_velocity",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=velocity_mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "base_velocity",
        "command_threshold": 0.05,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=velocity_mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "base_velocity",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=velocity_mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "base_velocity",
        "command_threshold": 0.05,
      },
    ),
    "self_collisions": RewardTermCfg(
      func=velocity_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }


def make_low_level_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }


def make_low_level_curriculum() -> dict[str, CurriculumTermCfg]:
  return {
    "command_vel": CurriculumTermCfg(
      func=velocity_mdp.commands_vel,
      params={
        "command_name": "base_velocity",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.3, 0.3)},
          {"step": 3000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 6000 * 24, "lin_vel_x": (-2.0, 3.0), "ang_vel_z": (-0.7, 0.7)},
        ],
      },
    )
  }


def make_low_level_env_cfg(
  *, sequential: bool = True, play: bool = False
) -> ManagerBasedRlEnvCfg:
  terrain = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    curriculum=False,
    sub_terrains={
      "flat": flat(proportion=0.6),
      "random_rough": random_rough(
        proportion=0.4, noise_range=(0.01, 0.03), noise_step=0.01
      ),
    },
  )
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(
      ObjRef(type="site", name=site_name, entity="robot")
      for site_name in ("left_foot", "right_foot")
    ),
    ray_alignment="yaw",
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )
  feet_sensor = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_sensor = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  robot_cfg = get_navigation_g1_robot_cfg()
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator", terrain_generator=terrain, max_init_terrain_level=3
      ),
      entities={"robot": robot_cfg},
      sensors=(foot_height_scan, feet_sensor, self_collision_sensor),
      num_envs=4096,
      env_spacing=2.5,
    ),
    observations=make_low_level_observations(),
    actions=make_low_level_actions(),
    commands=make_low_level_commands(),
    events=make_low_level_events(sequential=sequential),
    rewards=make_low_level_rewards(),
    terminations=make_low_level_terminations(),
    curriculum=make_low_level_curriculum(),
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
    # Apply the repository's play contract and the native flat-G1 command range.
    cfg.scene.num_envs = 32
    cfg.episode_length_s = 1e9
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    terrain.num_rows = 2
    terrain.num_cols = 10
    terrain.curriculum = False
    cfg.curriculum = {}
    base_velocity_command = cast(
      UniformVelocityCommandCfg, cfg.commands["base_velocity"]
    )
    base_velocity_command.ranges.lin_vel_x = (-2.0, 3.0)
    base_velocity_command.ranges.ang_vel_z = (-0.7, 0.7)
  return cfg
