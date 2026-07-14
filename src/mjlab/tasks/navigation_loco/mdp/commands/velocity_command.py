from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class AdaptiveLevelVelocityCommand(UniformVelocityCommand):
  """Velocity command generator with error-driven sampling over fixed bins."""

  cfg: UniformLevelVelocityCommandCfg
  _AXIS_NAMES = ("lin_vel_x", "lin_vel_y", "ang_vel_z")

  def __init__(self, cfg: UniformLevelVelocityCommandCfg, env):
    super().__init__(cfg, env)

    # build fixed absolute bin geometry from limit_ranges (one entry per axis)
    self._num_bins_axis: list[int] = []
    self._bin_edges: list[torch.Tensor] = []
    self._bin_centers: list[torch.Tensor] = []
    self._bin_error_ema: list[torch.Tensor] = []
    self._bin_update_count: list[torch.Tensor] = []
    self._bin_prob: list[torch.Tensor] = []
    for axis_name in self._AXIS_NAMES:
      low, high = getattr(cfg.limit_ranges, axis_name)
      n = self._resolve_num_bins(axis_name)
      edges = torch.linspace(float(low), float(high), n + 1, device=self.device)
      self._num_bins_axis.append(n)
      self._bin_edges.append(edges)
      self._bin_centers.append(0.5 * (edges[:-1] + edges[1:]))
      self._bin_error_ema.append(torch.zeros(n, device=self.device))
      self._bin_update_count.append(
        torch.zeros(n, dtype=torch.long, device=self.device)
      )
      self._bin_prob.append(torch.full((n,), 1.0 / n, device=self.device))
    # bin index currently assigned to each env per axis, shape (num_envs, 3)
    self._env_bin = torch.zeros(self.num_envs, 3, dtype=torch.long, device=self.device)
    # accumulated absolute tracking error within the current command window, shape (num_envs, 3)
    self._cmd_error_accum = torch.zeros(self.num_envs, 3, device=self.device)
    # number of steps accumulated within the current command window, shape (num_envs,)
    self._cmd_step_count = torch.zeros(self.num_envs, device=self.device)

  def _resolve_num_bins(self, axis_name: str) -> int:
    return int(
      self.cfg.num_bins[axis_name]
      if isinstance(self.cfg.num_bins, dict)
      else self.cfg.num_bins
    )

  def _update_metrics(self) -> None:
    super()._update_metrics()
    if not self.cfg.adaptive_sampling:
      return
    # per-axis absolute tracking error in the robot base frame
    self._cmd_error_accum[:, 0] += torch.abs(
      self.vel_command_b[:, 0] - self.robot.data.root_link_lin_vel_b[:, 0]
    )
    self._cmd_error_accum[:, 1] += torch.abs(
      self.vel_command_b[:, 1] - self.robot.data.root_link_lin_vel_b[:, 1]
    )
    self._cmd_error_accum[:, 2] += torch.abs(
      self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    )
    self._cmd_step_count += 1

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if not self.cfg.adaptive_sampling:
      super()._resample_command(env_ids)
      return

    # flush accumulated error from the window that just ended into the bin EMAs
    self._update_bin_ema(env_ids)

    # sample new commands per axis from the error-weighted bin distribution
    for axis_idx, axis_name in enumerate(self._AXIS_NAMES):
      lo, hi = getattr(self.cfg.ranges, axis_name)
      values, bins = self._sample_axis(axis_idx, float(lo), float(hi), len(env_ids))
      self.vel_command_b[env_ids, axis_idx] = values
      self._env_bin[env_ids, axis_idx] = bins

    # reset the per-window accumulators for the resampled envs
    self._cmd_error_accum[env_ids] = 0.0
    self._cmd_step_count[env_ids] = 0.0

    # heading target (only used when heading_command is enabled)
    if self.cfg.heading_command:
      heading_range = cast(tuple[float, float], self.cfg.ranges.heading)
      r = torch.empty(len(env_ids), device=self.device)
      self.heading_target[env_ids] = r.uniform_(*heading_range)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

    # update standing envs
    r = torch.empty(len(env_ids), device=self.device)
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

  def _update_bin_ema(self, env_ids: torch.Tensor) -> None:
    env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
      return
    steps = self._cmd_step_count[env_ids]
    # only consider envs that actually executed at least one step in the window
    # and that were not standing (their command was zeroed, so error is uninformative)
    valid = (steps > 0) & (~self.is_standing_env[env_ids])
    if not torch.any(valid):
      return
    valid_ids = env_ids[valid]
    mean_err = self._cmd_error_accum[valid_ids] / self._cmd_step_count[
      valid_ids
    ].clamp_min(1.0).unsqueeze(-1)
    for axis_idx in range(3):
      bins = self._env_bin[valid_ids, axis_idx]
      sums = torch.zeros(self._num_bins_axis[axis_idx], device=self.device)
      counts = torch.zeros_like(sums)
      sums.scatter_add_(0, bins, mean_err[:, axis_idx])
      counts.scatter_add_(0, bins, torch.ones_like(mean_err[:, axis_idx]))
      updated = counts > 0
      idx = updated.nonzero(as_tuple=False).flatten()
      if idx.numel() == 0:
        continue
      observed = sums[idx] / counts[idx]
      old = self._bin_error_ema[axis_idx][idx]
      first = self._bin_update_count[axis_idx][idx] == 0
      new = old * (1.0 - self.cfg.ema_alpha) + observed * self.cfg.ema_alpha
      # for bins seen for the first time, initialize the EMA directly to the observed mean
      self._bin_error_ema[axis_idx][idx] = torch.where(first, observed, new)
      self._bin_update_count[axis_idx][idx] += 1

  def _eligible(self, axis_idx: int, low: float, high: float) -> torch.Tensor:
    edges = self._bin_edges[axis_idx]
    mask = (edges[1:] > low) & (edges[:-1] < high)
    if not torch.any(mask):
      # degenerate active range (e.g. zero-width): fall back to the bin containing the midpoint
      idx = (
        int(
          torch.searchsorted(
            edges, torch.tensor((low + high) * 0.5, device=self.device)
          ).clamp(1, len(edges) - 1)
        )
        - 1
      )
      mask[idx] = True
    return mask

  def _axis_prob(self, axis_idx: int, low: float, high: float) -> torch.Tensor:
    mask = self._eligible(axis_idx, low, high)
    prob = torch.zeros(self._num_bins_axis[axis_idx], device=self.device)
    n = int(mask.sum())
    counts = self._bin_update_count[axis_idx][mask]
    if int(counts.min()) < self.cfg.warmup_resamples:
      # warmup: explore every eligible bin uniformly until each has enough data
      prob[mask] = 1.0 / n
    else:
      base = self._bin_error_ema[axis_idx][mask].clamp_min(0.0)
      prob[mask] = (
        1.0 / n
        if float(base.sum()) <= 1e-8
        else base.pow(1.0 / max(self.cfg.sampling_temperature, 1e-6))
      )
      prob /= prob.sum()
    floor = self.cfg.min_bin_probability
    # mix with a uniform floor over eligible bins so exploration stays broad
    if floor > 0.0 and n * floor < 1.0:
      prob[mask] = prob[mask] * (1.0 - n * floor) + floor
    return prob / prob.sum()

  def _sample_axis(self, axis_idx: int, low: float, high: float, count: int):
    prob = self._axis_prob(axis_idx, low, high)
    self._bin_prob[axis_idx] = prob
    bins = torch.multinomial(prob, count, replacement=True)
    edges = self._bin_edges[axis_idx]
    values = edges[bins] + torch.rand(count, device=self.device) * (
      edges[bins + 1] - edges[bins]
    )
    # clip to the active range for bins that only partially overlap it
    return values.clamp(low, high), bins

  def get_sampling_log(self) -> dict[str, float]:
    """Return adaptive-sampling metrics for logging."""
    if not self.cfg.adaptive_sampling:
      return {}
    log: dict[str, float] = {}
    for axis_idx, name in enumerate(("vx", "vy", "wz")):
      prob = self._bin_prob[axis_idx]
      ema = self._bin_error_ema[axis_idx]
      edges = self._bin_edges[axis_idx]
      eligible = prob > 0.0
      eligible_ids = eligible.nonzero(as_tuple=False).flatten()
      top_bin = int(torch.argmax(prob).item())
      p_active = prob[eligible]
      entropy = float((-(p_active * torch.log(p_active.clamp_min(1e-12))).sum()).item())
      # hardest eligible bin by EMA error
      ema_masked = ema.clone()
      ema_masked[~eligible] = float("-inf")
      hard_bin = int(torch.argmax(ema_masked).item())
      prefix = f"AdaptiveSampling/{name}"
      log[f"{prefix}/top1_range_min"] = float(edges[top_bin].item())
      log[f"{prefix}/top1_range_max"] = float(edges[top_bin + 1].item())
      log[f"{prefix}/top1_prob"] = float(prob[top_bin].item())
      log[f"{prefix}/entropy_active"] = entropy
      log[f"{prefix}/num_active_bins"] = float(eligible.sum().item())
      log[f"{prefix}/num_total_bins"] = float(self._num_bins_axis[axis_idx])
      log[f"{prefix}/hard_range_min"] = float(edges[hard_bin].item())
      log[f"{prefix}/hard_range_max"] = float(edges[hard_bin + 1].item())
      log[f"{prefix}/edge_active_prob"] = float(
        (prob[eligible_ids[0]] + prob[eligible_ids[-1]]).item()
      )
    return log


@dataclass(kw_only=True)
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
  limit_ranges: UniformVelocityCommandCfg.Ranges
  adaptive_sampling: bool = False
  num_bins: dict[str, int] | int = 10
  ema_alpha: float = 0.1
  min_bin_probability: float = 0.02
  sampling_temperature: float = 1.0
  warmup_resamples: int = 2

  def build(self, env) -> AdaptiveLevelVelocityCommand:
    return AdaptiveLevelVelocityCommand(self, env)
