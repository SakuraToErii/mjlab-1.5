"""Shared utilities for ONNX policy export across RL tasks."""

import onnx
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import BaseAction


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
  """Convert list to CSV string with specified decimal precision."""
  fmt = f"{{:.{decimals}f}}"
  return delimiter.join(
    fmt.format(x)
    if isinstance(x, (int, float))
    else str(x)  # numbers → format, strings → as-is
    for x in arr
  )


def get_base_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
  """Get base metadata common to all RL policy exports.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.

  Returns:
    Dictionary of metadata fields that are common across all tasks.
  """
  robot: Entity = env.scene["robot"]
  action_term_names = env.action_manager.active_terms
  if len(action_term_names) != 1:
    raise ValueError(
      "Policy export metadata requires exactly one affine action term, "
      f"received {action_term_names}."
    )
  action = env.action_manager.get_term(action_term_names[0])
  if not isinstance(action, BaseAction):
    raise TypeError(
      "Policy export metadata requires a BaseAction-compatible action term, "
      f"received {type(action).__name__}."
    )
  # Build mapping from joint name to actuator ID for natural joint order.
  # Each spec actuator controls exactly one joint (via its target field).
  joint_name_to_ctrl_id = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id
  # Get actuator IDs in natural joint order (same order as robot.joint_names).
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names  # global joint order
    if jname in joint_name_to_ctrl_id  # skip non-actuated joints
  ]
  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
  return {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": env.observation_manager.active_terms["actor"],
    "action_term": action_term_names[0],
    "action_names": action.target_names,
    "action_scale": action.scale[0].cpu().tolist()
    if isinstance(action.scale, torch.Tensor)
    else action.scale,
    "action_offset": action.offset[0].cpu().tolist()
    if isinstance(action.offset, torch.Tensor)
    else action.offset,
  }


def attach_metadata_to_onnx(
  onnx_path: str, metadata: dict[str, list | str | float]
) -> None:
  """Attach metadata to an ONNX model file.

  Args:
    onnx_path: Path to the ONNX model file.
    metadata: Dictionary of metadata key-value pairs to attach.
  """
  model = onnx.load(onnx_path)

  for k, v in metadata.items():
    entry = onnx.StringStringEntryProto()
    entry.key = k
    entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
