"""Unitree G1 configuration for feed-forward torque control."""

from dataclasses import replace

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_actuators import (
  UnitreeN5020_16ActuatorCfg,
  UnitreeN7520_14p3ActuatorCfg,
  UnitreeN7520_22p5ActuatorCfg,
  UnitreeW4010_25ActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

EFFORT_STANDING_JOINT_POSITION: dict[str, float] = {
  "left_hip_pitch_joint": -0.059750,
  "right_hip_pitch_joint": -0.028836,
  "waist_yaw_joint": -0.002169,
  "left_hip_roll_joint": 0.001778,
  "right_hip_roll_joint": -0.003393,
  "waist_roll_joint": -0.001681,
  "left_hip_yaw_joint": 0.011778,
  "right_hip_yaw_joint": -0.003191,
  "waist_pitch_joint": -0.000188,
  "left_knee_joint": 0.175060,
  "right_knee_joint": 0.122524,
  "left_shoulder_pitch_joint": 0.312136,
  "right_shoulder_pitch_joint": 0.324009,
  "left_ankle_pitch_joint": -0.120162,
  "right_ankle_pitch_joint": -0.098493,
  "left_shoulder_roll_joint": 0.257946,
  "right_shoulder_roll_joint": -0.246549,
  "left_ankle_roll_joint": 0.005809,
  "right_ankle_roll_joint": 0.007654,
  "left_shoulder_yaw_joint": 0.003801,
  "right_shoulder_yaw_joint": -0.019234,
  "left_elbow_joint": 0.929968,
  "right_elbow_joint": 0.969214,
  "left_wrist_roll_joint": 0.152960,
  "right_wrist_roll_joint": -0.152853,
  "left_wrist_pitch_joint": -0.025037,
  "right_wrist_pitch_joint": -0.013271,
  "left_wrist_yaw_joint": 0.013372,
  "right_wrist_yaw_joint": 0.007627,
}

EFFORT_STANDING_ROOT_HEIGHT = 0.789733


def get_g1_effort_robot_cfg() -> EntityCfg:
  """Return a fresh G1 config with zero-gain Unitree actuator groups."""
  cfg = get_g1_robot_cfg()
  assert cfg.articulation is not None

  actuators = (
    UnitreeN7520_14p3ActuatorCfg(
      target_names_expr=(
        r".*_hip_pitch_joint",
        r".*_hip_yaw_joint",
        "waist_yaw_joint",
      ),
      stiffness=0.0,
      damping=0.0,
    ),
    UnitreeN7520_22p5ActuatorCfg(
      target_names_expr=(r".*_hip_roll_joint", r".*_knee_joint"),
      stiffness=0.0,
      damping=0.0,
    ),
    UnitreeN5020_16ActuatorCfg(
      target_names_expr=(
        r".*_shoulder_.*",
        r".*_elbow_joint",
        r".*_wrist_roll_joint",
      ),
      stiffness=0.0,
      damping=0.0,
    ),
    UnitreeN5020_16ActuatorCfg(
      target_names_expr=(
        r".*_ankle_.*",
        "waist_roll_joint",
        "waist_pitch_joint",
      ),
      stiffness=0.0,
      damping=0.0,
    ),
    UnitreeW4010_25ActuatorCfg(
      target_names_expr=(r".*_wrist_pitch_joint", r".*_wrist_yaw_joint"),
      stiffness=0.0,
      damping=0.0,
    ),
  )

  return replace(
    cfg,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, EFFORT_STANDING_ROOT_HEIGHT),
      joint_pos=EFFORT_STANDING_JOINT_POSITION.copy(),
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=actuators,
      soft_joint_pos_limit_factor=cfg.articulation.soft_joint_pos_limit_factor,
    ),
  )
