"""Hierarchical G1 navigation environments."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from typing import cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.navigation_loco.mdp.raycast_patterns import OffsetGridPatternCfg
from mjlab.tasks.navigation_loco.navigation_loco_env_cfg import (
  get_navigation_g1_robot_cfg,
  make_low_level_actions,
  make_low_level_inference_observations,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from . import mdp

V3_MAX_CYLINDER_OBSTACLES = 8
V4_MAX_MAZE_OBSTACLES = 81
V5_MAX_MIXED_OBSTACLES = 120
V4_OBSTACLE_COUNT_LEVELS = (0, 45, 65, 81, 120)
V5_OBSTACLE_COUNT_LEVELS = (0, 50, 80, 100, 120)
# Resolve the packaged pretrained low-level policy relative to the mjlab task package.
# __file__ = .../src/mjlab/tasks/navigation/navigation_env_cfg.py
_DEFAULT_POLICY_PATH = (
  Path(__file__).parents[1] / "navigation_loco" / "models" / "policy.pt"
)
_POLICY_PATH = Path(
  os.environ.get("UNITREE_G1_LOW_LEVEL_POLICY_PATH", _DEFAULT_POLICY_PATH)
)


def make_navigation_scene_cfg(variant: int = 1) -> SceneCfg:
  entities = {"robot": get_navigation_g1_robot_cfg()}
  if variant == 3:
    entities.update(mdp.make_cylinder_obstacle_entities(8, 0.4, 2.0))
    scan_size, resolution = (1.6, 1.0), 0.1
  elif variant == 4:
    entities.update(mdp.make_cylinder_obstacle_entities(81, 0.35, 2.0))
    scan_size, resolution = (2.4, 1.6), 0.1
  elif variant >= 5:
    entities.update(mdp.make_mixed_obstacle_collection())
    # mjlab's inclusive endpoint convention needs 4.92 m here to retain the
    # source 42x26=1092 ray ABI (and 21x13=273 after 2x2 pooling).
    scan_size, resolution = (4.92, 3.0), 0.12
  else:
    scan_size, resolution = (1.6, 1.0), 0.1
  scanner = RayCastSensorCfg(
    name="height_scanner",
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    ray_alignment="yaw",
    pattern=OffsetGridPatternCfg(
      size=scan_size, resolution=resolution, origin_offset=(0.0, 0.0, 20.0)
    ),
    max_distance=100.0,
    exclude_parent_body=True,
    include_geom_groups=(0, 1),
    debug_vis=False,
  )
  return SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities=entities,
    sensors=(scanner,),
    num_envs=4096,
    env_spacing=8.0 if variant == 1 else 60.0,
  )


def make_navigation_actions() -> dict[str, ActionTermCfg]:
  return {
    "pre_trained_policy_action": mdp.PreTrainedPolicyActionCfg(
      entity_name="robot",
      policy_path=str(_POLICY_PATH),
      low_level_decimation=4,
      low_level_actions=deepcopy(make_low_level_actions()["JointPositionAction"]),
      low_level_observations=make_low_level_inference_observations(),
      velocity_clip=((-0.5, 1.0), (-0.5, 0.5), (-0.5, 0.5)),
      debug_vis=True,
    )
  }


def make_navigation_commands(
  variant: int = 1, *, compact_single_goal: bool = False
) -> dict[str, CommandTermCfg]:
  if compact_single_goal:
    command = mdp.RingPose2dCommandCfg(
      entity_name="robot",
      simple_heading=True,
      resampling_time_range=(30.0, 30.0),
      debug_vis=False,
      success_radius=0.5,
      update_goal_on_success=False,
      obstacle_filter=True,
      max_obstacle_resample_tries=128,
      ranges=mdp.RingPose2dCommandCfg.Ranges(
        distance=(5.0, 10.0), heading=(-math.pi, math.pi)
      ),
    )
  elif variant >= 4:
    command = mdp.ArenaPose2dCommandCfg(
      entity_name="robot",
      simple_heading=False,
      resampling_time_range=(1e9, 1e9),
      debug_vis=True,
      success_radius=0.5,
      update_goal_on_success=True,
      obstacle_filter=True,
      max_obstacle_resample_tries=128,
      arena_half_extent=25.0,
      arena_margin=0.5,
      ranges=mdp.ArenaPose2dCommandCfg.Ranges(
        distance=(0.0, 0.0), heading=(-math.pi, math.pi)
      ),
    )
  else:
    command = mdp.RingPose2dCommandCfg(
      entity_name="robot",
      simple_heading=False,
      resampling_time_range=(30.0, 30.0) if variant == 1 else (1e9, 1e9),
      debug_vis=True,
      success_radius=1.0 if variant == 1 else 0.5,
      update_goal_on_success=variant >= 2,
      obstacle_filter=variant >= 3,
      max_obstacle_resample_tries=128,
      ranges=mdp.RingPose2dCommandCfg.Ranges(
        distance=(2.0, 5.0) if variant == 1 else (5.0, 25.0),
        heading=(-math.pi, math.pi),
      ),
    )
  return {"pose_command": command}


def make_navigation_observations(
  *, pooled: bool = False, height_clip: tuple[float, float] = (-1.5, 1.5)
) -> dict[str, ObservationGroupCfg]:
  # 提示：高层策略需要“目标 + 局部感知 + 本体状态 + 控制历史”。
  # 参考基线包含：机体线/角速度、投影重力、目标命令、上一高层命令、高度扫描、
  # 关节位置/速度，以及冻结低层策略输出的上一关节动作。
  # >>> HOMEWORK_TODO_7_START
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel, scale=0.2),
    "pose_command": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "pose_command"}
    ),
    "last_high_level_command": ObservationTermCfg(func=mdp.last_high_level_command),
    ("height_scan_pooled" if pooled else "height_scan"): ObservationTermCfg(
      func=mdp.height_scan_pooled if pooled else mdp.height_scan,
      params={"sensor_name": "height_scanner", **({"pool_size": 2} if pooled else {})},
      clip=height_clip,
    ),
    "joint_pos_rel": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel_rel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, scale=0.05),
    "low_level_last_action": ObservationTermCfg(func=mdp.low_level_last_action),
  }
  # <<< HOMEWORK_TODO_7_END
  critic_terms = {
    **actor_terms,
    "base_height": ObservationTermCfg(func=mdp.base_pos_z),
    "command_distance": ObservationTermCfg(
      func=mdp.command_distance, params={"command_name": "pose_command"}
    ),
  }
  return {
    # Target task registry expects train actor corruption enabled. These terms have no
    # noise, so this flag does not change the source observation values.
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }


def make_navigation_events(
  variant: int = 1, *, fixed_arena: bool = False
) -> dict[str, EventTermCfg]:
  events: dict[str, EventTermCfg] = {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default, mode="reset"
    )
  }
  if variant == 3:
    events["randomize_cylinders"] = EventTermCfg(
      func=mdp.randomize_cylinder_layout,
      mode="reset",
      params={
        "layout_cfg": mdp.CylinderObstacleLayoutCfg(
          max_obstacles=8,
          cylinder_radius=0.4,
          cylinder_height=2.0,
          soft_margin=0.6,
          min_center_separation=2.0,
        ),
        "default_num_active": 0,
      },
    )
  elif variant == 4:
    events["randomize_cylinders"] = EventTermCfg(
      func=mdp.randomize_cylinder_layout,
      mode="reset",
      params={
        "layout_cfg": mdp.CylinderObstacleLayoutCfg(
          max_obstacles=81,
          cylinder_radius=0.35,
          cylinder_height=2.0,
          soft_margin=0.4,
          min_center_separation=1.45,
          exclude_origin=False,
          max_resample_tries=256,
        ),
        "default_num_active": 0,
      },
    )
  elif variant >= 5:
    layout = mdp.MixedObstacleLayoutCfg(
      max_obstacles=120,
      soft_margin=0.4,
      min_center_separation=1.1,
      exclude_origin=False,
    )
    if fixed_arena:
      # Target reset_data restores mocap poses, so rewrite the same sticky per-env
      # template after every reset and before the obstacle-aware robot reset.
      events["assign_arena_maps"] = EventTermCfg(
        func=mdp.assign_fixed_mixed_arena_layout,
        mode="reset",
        params={"layout_cfg": layout},
      )
    else:
      events["randomize_obstacles"] = EventTermCfg(
        func=mdp.randomize_mixed_obstacle_layout,
        mode="reset",
        params={"layout_cfg": layout, "default_num_active": 0},
      )
  pose = (
    {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)}
    if variant < 4
    else {"x": (-25.0, 25.0), "y": (-25.0, 25.0), "yaw": (-math.pi, math.pi)}
  )
  reset_func = (
    mdp.reset_root_state_obstacle_aware
    if variant >= 3
    else envs_mdp.reset_root_state_uniform
  )
  events["reset_base"] = EventTermCfg(
    func=reset_func,
    mode="reset",
    params={
      "pose_range": pose,
      "velocity_range": {},
      **({"robot_radius": 0.5} if variant >= 3 else {}),
    },
  )
  events["reset_robot_joints"] = EventTermCfg(
    func=envs_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (-1.0, 1.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  return events


def make_navigation_rewards(variant: int = 1) -> dict[str, RewardTermCfg]:
  rewards = {
    "termination_penalty": RewardTermCfg(
      func=mdp.is_terminated_term,
      weight=-400.0,
      params={"term_keys": ["base_height", "bad_orientation"]},
    ),
    "position_progress": RewardTermCfg(
      func=mdp.pose_command_progress,
      weight=1.0 if variant == 1 else 2.0,
      params={"command_name": "pose_command"},
    ),
    "position_tracking_fine_grained": RewardTermCfg(
      func=mdp.position_command_error_tanh,
      weight=0.5,
      params={"std": 0.2 if variant == 1 else 0.1, "command_name": "pose_command"},
    ),
    "success_bonus": RewardTermCfg(
      func=mdp.goal_reached_bonus,
      weight=100.0 if variant == 1 else 50.0,
      params={"command_name": "pose_command"},
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    "action_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-0.01),
  }
  if variant >= 3:
    rewards["obstacle_soft_zone"] = RewardTermCfg(
      func=mdp.obstacle_soft_zone_penalty, weight=-6.0 if variant >= 5 else -2.0
    )
  return rewards


def make_navigation_terminations(
  variant: int = 1, *, single_goal: bool = False
) -> dict[str, TerminationTermCfg]:
  terms = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum, params={"minimum_height": 0.2}
    ),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": 0.8}
    ),
  }
  if variant == 1 or single_goal:
    terms["goal_reached"] = TerminationTermCfg(
      func=mdp.goal_reached,
      params={"command_name": "pose_command", "threshold": 0.5 if single_goal else 1.0},
    )
  return terms


def make_navigation_curriculum(variant: int) -> dict[str, CurriculumTermCfg]:
  if variant == 3:
    return {"obstacle_count": CurriculumTermCfg(func=mdp.obstacle_count_levels)}
  if variant == 4:
    return {
      "obstacle_count": CurriculumTermCfg(
        func=mdp.obstacle_count_levels,
        params={"level_counts": V4_OBSTACLE_COUNT_LEVELS},
      )
    }
  if variant >= 5:
    return {
      "obstacle_count": CurriculumTermCfg(
        func=mdp.obstacle_count_levels,
        params={"level_counts": V5_OBSTACLE_COUNT_LEVELS},
      )
    }
  return {}


def _make_navigation_cfg(
  variant: int, *, compact: bool = False, single_goal: bool = False, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = ManagerBasedRlEnvCfg(
    scene=make_navigation_scene_cfg(variant),
    actions=make_navigation_actions(),
    observations=make_navigation_observations(
      pooled=compact,
      height_clip=(-1.5, 1.5) if variant >= 5 else (-1.0, 1.0),
    ),
    events=make_navigation_events(variant, fixed_arena=compact),
    commands=make_navigation_commands(variant, compact_single_goal=single_goal),
    rewards=make_navigation_rewards(variant),
    terminations=make_navigation_terminations(variant, single_goal=single_goal),
    curriculum={} if compact else make_navigation_curriculum(variant),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=10.0 if compact else 3.0,
    ),
    sim=SimulationCfg(
      nconmax=700,
      njmax=2500,
      contact_sensor_maxmatch=500,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=40,
    # V2 and all derived V3-V5 training variants explicitly keep 30 s episodes
    # even though their commands only resample on success.
    episode_length_s=30.0,
  )
  if compact:
    hierarchical_action = cast(
      mdp.PreTrainedPolicyActionCfg,
      cfg.actions["pre_trained_policy_action"],
    )
    hierarchical_action.debug_vis = False
    cfg.commands["pose_command"].debug_vis = False
  if play:
    # Target play convention: deterministic actor, infinite horizon and identical I/O structure.
    cfg.scene.num_envs = 16
    cfg.episode_length_s = 1e9
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events.pop("push_robot", None)
    if "randomize_obstacles" in cfg.events:
      cfg.events["randomize_obstacles"].params["default_num_active"] = (
        V5_MAX_MIXED_OBSTACLES
      )
    cfg.viewer.height = 1080
    cfg.viewer.width = 1920
    if compact:
      # Top-down camera: follows robot root; mjlab expresses the +Z view with elevation.
      cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
      cfg.viewer.body_name = None
      cfg.viewer.distance = 10.0
      cfg.viewer.elevation = -90.0
      cfg.commands["pose_command"].debug_vis = True
      for sensor in cfg.scene.sensors:
        if sensor.name == "height_scanner":
          cast(RayCastSensorCfg, sensor).debug_vis = True
  return cfg


def make_navigation_env_cfg(play: bool = False):
  return _make_navigation_cfg(1, play=play)


def make_navigation_v2_env_cfg(play: bool = False):
  return _make_navigation_cfg(2, play=play)


def make_navigation_v3_maze_env_cfg(play: bool = False):
  return _make_navigation_cfg(3, play=play)


def make_navigation_v4_fixed_maze_env_cfg(play: bool = False):
  return _make_navigation_cfg(4, play=play)


def make_navigation_v5_mixed_obstacle_env_cfg(play: bool = False):
  return _make_navigation_cfg(5, play=play)


def make_navigation_v5_compact_env_cfg(play: bool = False):
  return _make_navigation_cfg(5, compact=True, play=play)


def make_navigation_v5_compact_single_goal_env_cfg(play: bool = False):
  return _make_navigation_cfg(5, compact=True, single_goal=True, play=play)
