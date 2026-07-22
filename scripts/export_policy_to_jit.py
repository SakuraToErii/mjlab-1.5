"""Export a registered RSL-RL actor checkpoint as a TorchScript policy."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import tyro

import mjlab
import mjlab.tasks  # noqa: F401  # Populate the task registry.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TASK_ID = "Unitree-G1-29dof-LowLevel"
_DEFAULT_CHECKPOINT = (
  _REPO_ROOT
  / "logs/rsl_rl/Unitree-G1-29dof-LowLevel"
  / "2026-07-15_17-10-50_g1-lowlevel-sequential-4096-8k/model_9999.pt"
)
_DEFAULT_OUTPUT = _REPO_ROOT / "src/mjlab/tasks/navigation_loco/models/policy.pt"


def main(
  task_id: str = _DEFAULT_TASK_ID,
  checkpoint: Path = _DEFAULT_CHECKPOINT,
  output: Path = _DEFAULT_OUTPUT,
  device: str = "cpu",
) -> None:
  """Load a training checkpoint and export its deterministic actor."""
  checkpoint = checkpoint.expanduser().resolve()
  output = output.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(task_id)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  try:
    observations = env.get_observations()
    actor_input = torch.cat(
      [observations[name] for name in agent_cfg.obs_groups["actor"]], dim=-1
    )[:1]

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    runner.export_policy_to_jit(str(output.parent), output.name)

    exported = torch.jit.load(str(output), map_location="cpu").eval()
    with torch.inference_mode():
      actions = exported(actor_input.detach().cpu())
    expected_shape = (actor_input.shape[0], env.num_actions)
    if tuple(actions.shape) != expected_shape:
      raise RuntimeError(
        f"Exported policy output shape is {tuple(actions.shape)}, "
        f"expected {expected_shape}."
      )
    if not torch.isfinite(actions).all():
      raise RuntimeError("Exported policy produced non-finite actions.")

    print(
      f"Exported {checkpoint} -> {output} "
      f"(ABI: {actor_input.shape[-1]} -> {actions.shape[-1]})"
    )
  finally:
    env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
