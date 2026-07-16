"""Residual effort action constants for the Unitree G1."""

from mjlab.envs.mdp.actions import JointEffortActionCfg

# Same-direction peak torque (Y1) for each Unitree motor family.
EFFORT_ACTION_LIMIT: dict[str, float] = {
  r".*_hip_pitch_joint": 71.0,
  r".*_hip_yaw_joint": 71.0,
  "waist_yaw_joint": 71.0,
  r".*_hip_roll_joint": 111.0,
  r".*_knee_joint": 111.0,
  r".*_shoulder_.*": 24.8,
  r".*_elbow_joint": 24.8,
  r".*_wrist_roll.*": 24.8,
  r".*_ankle_.*": 24.8,
  "waist_roll_joint": 24.8,
  "waist_pitch_joint": 24.8,
  r".*_wrist_pitch.*": 4.8,
  r".*_wrist_yaw.*": 4.8,
}

EFFORT_ACTION_CLIP: dict[str, tuple[float, float]] = {
  joint_expr: (-limit, limit) for joint_expr, limit in EFFORT_ACTION_LIMIT.items()
}

# The policy controls a 40% residual band around the standing-torque reference.
RESIDUAL_ACTION_SCALE: dict[str, float] = {
  joint_expr: 0.4 * limit for joint_expr, limit in EFFORT_ACTION_LIMIT.items()
}

# Standing-torque reference collected from the paired IsaacLab effort task.
NOMINAL_TORQUE: dict[str, float] = {
  "left_hip_pitch_joint": -1.171482,
  "right_hip_pitch_joint": -0.392699,
  "waist_yaw_joint": -0.049220,
  "left_hip_roll_joint": 11.628835,
  "right_hip_roll_joint": -9.975743,
  "waist_roll_joint": 0.391920,
  "left_hip_yaw_joint": -3.657360,
  "right_hip_yaw_joint": 3.118077,
  "waist_pitch_joint": 1.178703,
  "left_knee_joint": -1.801823,
  "right_knee_joint": -0.455979,
  "left_shoulder_pitch_joint": 0.630705,
  "right_shoulder_pitch_joint": 0.784696,
  "left_ankle_pitch_joint": 4.369744,
  "right_ankle_pitch_joint": 3.314344,
  "left_shoulder_roll_joint": 1.602750,
  "right_shoulder_roll_joint": -1.674056,
  "left_ankle_roll_joint": 0.662338,
  "right_ankle_roll_joint": 0.517337,
  "left_shoulder_yaw_joint": 0.344647,
  "right_shoulder_yaw_joint": -0.339807,
  "left_elbow_joint": -0.398630,
  "right_elbow_joint": -0.325242,
  "left_wrist_roll_joint": -0.002183,
  "right_wrist_roll_joint": 0.002247,
  "left_wrist_pitch_joint": -0.106291,
  "right_wrist_pitch_joint": -0.086652,
  "left_wrist_yaw_joint": 0.057935,
  "right_wrist_yaw_joint": -0.055295,
}


def g1_residual_effort_action_cfg() -> JointEffortActionCfg:
  """Create ``tau = tau_nominal + action * scale`` for all 29 joints."""
  return JointEffortActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=RESIDUAL_ACTION_SCALE,
    offset=NOMINAL_TORQUE,
    clip=EFFORT_ACTION_CLIP,
  )
