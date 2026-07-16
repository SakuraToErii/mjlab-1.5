"""Unitree G1 residual-effort velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.eff_velocity import mdp
from mjlab.tasks.eff_velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.eff_velocity.velocity_env_cfg import make_eff_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from .action_cfg import g1_residual_effort_action_cfg
from .robot_cfg import get_g1_effort_robot_cfg

_FULL_ACTOR_NOISE = {
  "base_lin_vel": (-0.5, 0.5),
  "base_ang_vel": (-0.2, 0.2),
  "projected_gravity": (-0.05, 0.05),
  "joint_pos": (-0.01, 0.01),
  "joint_vel": (-1.5, 1.5),
}


def _scaled_actor_noise(scale: float) -> dict[str, tuple[float, float]]:
  return {
    term_name: (n_min * scale, n_max * scale)
    for term_name, (n_min, n_max) in _FULL_ACTOR_NOISE.items()
  }


def _flat_effort_curriculum_stages() -> list[mdp.FlatEffortStage]:
  def robust_stage(name: str, push_velocity: float) -> mdp.FlatEffortStage:
    return {
      "name": name,
      "lin_vel_x": (-0.4, 0.6),
      "lin_vel_y": (-0.25, 0.25),
      "ang_vel_z": (-0.1, 0.1),
      "rel_standing_envs": 0.2,
      "push_velocity": push_velocity,
      "observation_noise": _scaled_actor_noise(1.0),
      "foot_friction": (0.3, 1.2),
      "encoder_bias": (-0.015, 0.015),
      "base_com": {
        0: (-0.025, 0.025),
        1: (-0.025, 0.025),
        2: (-0.03, 0.03),
      },
      "timeout_threshold": 0.8,
      "max_mean_lin_vel_error": 0.35,
      "max_mean_yaw_vel_error": 0.25,
    }

  return [
    {
      "name": "static_balance",
      "lin_vel_x": (0.0, 0.0),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (0.0, 0.0),
      "rel_standing_envs": 1.0,
      "push_velocity": 0.0,
      "observation_noise": _scaled_actor_noise(0.2),
      "foot_friction": (0.7, 1.0),
      "encoder_bias": (-0.003, 0.003),
      "base_com": {
        0: (-0.005, 0.005),
        1: (-0.005, 0.005),
        2: (-0.006, 0.006),
      },
      "timeout_threshold": 0.9,
      "max_mean_lin_vel_error": 0.15,
      "max_mean_yaw_vel_error": 0.15,
    },
    {
      "name": "slow_locomotion",
      "lin_vel_x": (-0.2, 0.3),
      "lin_vel_y": (-0.1, 0.1),
      "ang_vel_z": (-0.1, 0.1),
      "rel_standing_envs": 0.5,
      "push_velocity": 0.1,
      "observation_noise": _scaled_actor_noise(0.5),
      "foot_friction": (0.5, 1.1),
      "encoder_bias": (-0.0075, 0.0075),
      "base_com": {
        0: (-0.0125, 0.0125),
        1: (-0.0125, 0.0125),
        2: (-0.015, 0.015),
      },
      "timeout_threshold": 0.85,
      "max_mean_lin_vel_error": 0.25,
      "max_mean_yaw_vel_error": 0.2,
    },
    robust_stage("flat_robust_push_0_2", 0.2),
    robust_stage("flat_robust_push_0_3", 0.3),
    robust_stage("flat_robust_push_0_4", 0.4),
    robust_stage("flat_robust_push_0_5", 0.5),
  ]


def _configure_flat_effort_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  stages = _flat_effort_curriculum_stages()
  initial_stage = stages[0]

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = initial_stage["lin_vel_x"]
  twist_cmd.ranges.lin_vel_y = initial_stage["lin_vel_y"]
  twist_cmd.ranges.ang_vel_z = initial_stage["ang_vel_z"]
  twist_cmd.rel_standing_envs = initial_stage["rel_standing_envs"]

  for term_name, (n_min, n_max) in initial_stage["observation_noise"].items():
    cfg.observations["actor"].terms[term_name].noise = UniformNoiseCfg(
      n_min=n_min,
      n_max=n_max,
    )

  push_cfg = cfg.events["push_robot"]
  push_cfg.interval_range_s = (5.0, 5.0)
  push_cfg.params["velocity_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
  }
  cfg.events["foot_friction"].params["ranges"] = initial_stage["foot_friction"]
  cfg.events["encoder_bias"].params["bias_range"] = initial_stage["encoder_bias"]
  cfg.events["base_com"].params["ranges"] = dict(initial_stage["base_com"])

  cfg.curriculum = {
    "flat_effort_stages": CurriculumTermCfg(
      func=mdp.FlatEffortCurriculum,
      params={
        "command_name": "twist",
        "push_event_name": "push_robot",
        "stages": stages,
        "min_episodes": 4096,
      },
    )
  }


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the G1 rough-terrain residual-effort configuration."""
  cfg = make_eff_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_g1_effort_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
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
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  cfg.actions = {"joint_effort": g1_residual_effort_action_cfg()}

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the G1 flat-terrain residual-effort configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)
  else:
    _configure_flat_effort_curriculum(cfg)

  return cfg


def _enable_mha_history(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Retain mjlab's time axis for the actor and flatten critic history."""
  actor_obs = cfg.observations["actor"]
  actor_obs.history_length = 5
  actor_obs.flatten_history_dim = False

  critic_obs = cfg.observations["critic"]
  critic_obs.history_length = 5
  critic_obs.flatten_history_dim = True
  return cfg


def unitree_g1_rough_mha_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the rough-terrain effort task with five-frame MHA history."""
  return _enable_mha_history(unitree_g1_rough_env_cfg(play=play))


def unitree_g1_flat_mha_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat-terrain effort task with five-frame MHA history."""
  return _enable_mha_history(unitree_g1_flat_env_cfg(play=play))
