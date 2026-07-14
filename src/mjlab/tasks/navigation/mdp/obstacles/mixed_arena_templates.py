"""Pre-baked V5 mixed-arena layouts sampled once per template seed."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .mixed_obstacle_collection import V5_MAX_MIXED_OBSTACLES
from .mixed_obstacle_layout import MixedObstacleLayoutCfg, _build_slot_metadata


@dataclass(frozen=True)
class FixedMixedArenaTemplate:
  """Procedurally sampled obstacle layout baked at max density."""

  name: str
  seed: int
  centers_xy: torch.Tensor
  active_slot_ids: torch.Tensor
  num_active: int


# Indexed variants of the same dense-mix sampler; seeds distinguish layouts.
_TEMPLATE_SPECS: tuple[tuple[str, int], ...] = (
  ("dense_mix_0", 1001),
  ("dense_mix_1", 1002),
  ("dense_mix_2", 1003),
)

_BAKED_CACHE: dict[tuple, FixedMixedArenaTemplate] = {}


def _pair_separation_sq(
  footprint_radius: torch.Tensor,
  min_center_separation: float,
  slot_a: int,
  slot_b: int,
) -> float:
  sep = footprint_radius[slot_a] + footprint_radius[slot_b] + min_center_separation
  return float(sep.item() ** 2)


def bake_mixed_arena_template(
  seed: int,
  layout_cfg: MixedObstacleLayoutCfg | None = None,
  device: torch.device | str = "cpu",
) -> FixedMixedArenaTemplate:
  """Sample a single max-density mixed layout with a fixed RNG seed."""
  if layout_cfg is None:
    layout_cfg = MixedObstacleLayoutCfg()

  device = torch.device(device)
  _, footprint_radius, _, _ = _build_slot_metadata(device)

  # Keep template baking deterministic without replacing the environment's global
  # RNG state before the subsequent robot and command reset events.
  generator = torch.Generator(device=device).manual_seed(seed)

  target_count = min(layout_cfg.max_obstacles, V5_MAX_MIXED_OBSTACLES)
  low = -layout_cfg.arena_half_extent + layout_cfg.arena_margin
  high = layout_cfg.arena_half_extent - layout_cfg.arena_margin

  centers_xy = torch.zeros(target_count, 2, device=device)
  active_slot_ids = torch.full((target_count,), -1, dtype=torch.long, device=device)

  def slot_clearance(slot_id: int) -> float:
    return footprint_radius[slot_id].item() + layout_cfg.soft_margin

  origin_clearance_sq = (slot_clearance(0) + 0.8) ** 2

  perm = torch.randperm(V5_MAX_MIXED_OBSTACLES, device=device, generator=generator)[
    :target_count
  ]
  placed = 0
  for slot_id_tensor in perm:
    slot_id = int(slot_id_tensor.item())
    if placed >= target_count:
      break
    slot_clearance_sq = (slot_clearance(slot_id) + 0.8) ** 2
    for _ in range(layout_cfg.max_resample_tries):
      candidate = torch.empty(2, device=device).uniform_(low, high, generator=generator)
      if (
        layout_cfg.exclude_origin
        and torch.sum(candidate.square()) < origin_clearance_sq
      ):
        continue
      if (
        layout_cfg.exclude_origin and torch.sum(candidate.square()) < slot_clearance_sq
      ):
        continue
      if placed > 0:
        delta = centers_xy[:placed] - candidate
        dist_sq = torch.sum(delta.square(), dim=1)
        sep_sq = torch.tensor(
          [
            _pair_separation_sq(
              footprint_radius, layout_cfg.min_center_separation, int(s), slot_id
            )
            for s in active_slot_ids[:placed]
          ],
          device=device,
        )
        if torch.any(dist_sq < sep_sq):
          continue
      centers_xy[placed] = candidate
      active_slot_ids[placed] = slot_id
      placed += 1
      break

  return FixedMixedArenaTemplate(
    name="",
    seed=seed,
    centers_xy=centers_xy[:placed].clone(),
    active_slot_ids=active_slot_ids[:placed].clone(),
    num_active=placed,
  )


def get_fixed_mixed_arena_template(
  level: int, layout_cfg: MixedObstacleLayoutCfg | None = None
) -> FixedMixedArenaTemplate:
  """Return a cached baked template for the given level index (0-2)."""
  if layout_cfg is None:
    layout_cfg = MixedObstacleLayoutCfg()

  cache_key = (
    level,
    layout_cfg.arena_half_extent,
    layout_cfg.arena_margin,
    layout_cfg.min_center_separation,
  )
  if cache_key not in _BAKED_CACHE:
    name, seed = _TEMPLATE_SPECS[level]
    baked = bake_mixed_arena_template(seed, layout_cfg=layout_cfg)
    _BAKED_CACHE[cache_key] = FixedMixedArenaTemplate(
      name=name,
      seed=seed,
      centers_xy=baked.centers_xy,
      active_slot_ids=baked.active_slot_ids,
      num_active=baked.num_active,
    )
  return _BAKED_CACHE[cache_key]


def num_fixed_mixed_arena_templates() -> int:
  return len(_TEMPLATE_SPECS)
