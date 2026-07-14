from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def gait_phase(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  return torch.stack(
    (
      torch.sin(global_phase * torch.pi * 2.0),
      torch.cos(global_phase * torch.pi * 2.0),
    ),
    dim=-1,
  )
