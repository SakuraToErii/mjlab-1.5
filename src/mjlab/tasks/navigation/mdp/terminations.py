import torch


def goal_reached(env, command_name: str, threshold: float) -> torch.Tensor:
  return (
    torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < threshold
  )
