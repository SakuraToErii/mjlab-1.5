from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch

from mjlab.sensor import GridPatternCfg


@dataclass
class OffsetGridPatternCfg(GridPatternCfg):
  """Grid rays whose origins are translated in the aligned sensor frame.

  The source task keeps the configured ray-start offset separate from the
  reported attachment-frame pose. Expressing the offset in the pattern preserves that
  behavior with mjlab's native raycast sensor: ray starts move, while
  ``frame_pos_w`` remains the physical body position used by ``height_scan``.
  """

  origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    local_offsets, local_directions = super().generate_rays(mj_model, device)
    local_offsets += torch.tensor(
      self.origin_offset, device=device, dtype=local_offsets.dtype
    )
    return local_offsets, local_directions
