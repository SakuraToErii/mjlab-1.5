from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationManager
from mjlab.utils.lab_api.math import quat_apply, yaw_quat


class PreTrainedPolicyAction(ActionTerm):
  """High-level velocity action consumed by a frozen 50 Hz low-level policy."""

  cfg: PreTrainedPolicyActionCfg

  def __init__(self, cfg: PreTrainedPolicyActionCfg, env) -> None:
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]

    policy_path = Path(cfg.policy_path)
    if not policy_path.is_file():
      raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")

    self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)
    self._low_level_action_term = cfg.low_level_actions.build(env)
    self.low_level_actions = torch.zeros(
      self.num_envs, self._low_level_action_term.action_dim, device=self.device
    )
    self._counter = 0

    def last_low_level_action() -> torch.Tensor:
      self.low_level_actions[env.episode_length_buf == 0, :] = 0.0
      return self.low_level_actions

    # 提示：
    # 1. 用 torch.jit.load 加载随仓库提供的低层策略，并切换到 eval 模式。
    # 2. 将低层观测中的 velocity_commands 绑定到裁剪后的高层动作。
    # 3. 将 last_action（或兼容名称 actions）绑定到上一次低层关节动作。
    # 4. 最后创建只含 ll_policy 组的 ObservationManager。
    # >>> HOMEWORK_TODO_3_START
    # 1. 加载本地低层策略。加载只在 ActionTerm.build() 创建运行时 term 时发生，
    #    不会在任务导入或配置构造阶段提前读取 checkpoint。
    self.low_level_policy = torch.jit.load(
      str(policy_path), map_location=self.device
    ).eval()

    observations = ObservationGroupCfg(
      terms={name: term for name, term in cfg.low_level_observations.terms.items()},
      concatenate_terms=True,
      enable_corruption=False,
      history_length=cfg.low_level_observations.history_length,
    )
    # 2. 将低层观测中的 velocity_commands 绑定到裁剪后的高层动作。
    #    清空 params：原 term 带 params={"command_name": "base_velocity"}，lambda 不接收它；
    #    ObservationManager 按 func(env, **params) 调用，残留 kwarg 会触发 TypeError。
    observations.terms["velocity_commands"].func = lambda _env: self._processed_actions
    observations.terms["velocity_commands"].params = {}

    # 3. 将 last_action 绑定到上一次低层关节动作。
    #    last_low_level_action 是零参闭包，包成 lambda env: ... 以匹配 func(env, **params) 调用约定。
    observations.terms["last_action"].func = lambda _env: last_low_level_action()
    observations.terms["last_action"].params = {}
    # 兼容源任务中名称为 actions 的旧 observation 配置。
    if "actions" in observations.terms:
      observations.terms["actions"].func = lambda _env: last_low_level_action()
      observations.terms["actions"].params = {}

    # 4. 创建只含 ll_policy 组的 ObservationManager。
    self._low_level_obs_manager = ObservationManager({"ll_policy": observations}, env)
    # <<< HOMEWORK_TODO_3_END

  @property
  def action_dim(self) -> int:
    return 3

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def processed_action(self) -> torch.Tensor:
    """High-level velocity command after per-axis clipping."""
    return self._processed_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    # 提示：保留原始动作供日志与调试使用；真正传给低层策略的动作须逐维裁剪。
    # >>> HOMEWORK_TODO_4_START
    self._raw_actions[:] = actions
    for axis, limits in enumerate(self.cfg.velocity_clip):
      self._processed_actions[:, axis] = actions[:, axis].clamp(*limits)
    # <<< HOMEWORK_TODO_4_END

  def apply_actions(self) -> None:
    # 提示：低层策略只在 low_level_decimation 到达时更新一次，其输出在中间物理步保持。
    # 推理必须放在 torch.inference_mode() 中；低层 action term 每个物理步都要 apply_actions。
    # >>> HOMEWORK_TODO_5_START
    if self._counter % self.cfg.low_level_decimation == 0:
      obs = cast(
        torch.Tensor,
        self._low_level_obs_manager.compute_group("ll_policy", update_history=True),
      )
      with torch.inference_mode():
        self.low_level_actions[:] = self.low_level_policy(obs)
      self._low_level_action_term.process_actions(self.low_level_actions)
    self._low_level_action_term.apply_actions()
    self._counter += 1
    # <<< HOMEWORK_TODO_5_END

  def draw_debug_vis(self, visualizer) -> None:
    """Draw desired and measured planar base-velocity arrows natively."""
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    desired_b = torch.cat(
      (self.processed_action[:, :2], torch.zeros(self.num_envs, 1, device=self.device)),
      dim=-1,
    )
    desired_w = quat_apply(yaw_quat(root_quat), desired_b)
    measured_w = self.robot.data.root_link_lin_vel_w.clone()
    measured_w[:, 2] = 0.0

    for env_id in visualizer.get_env_indices(self.num_envs):
      start = root_pos[env_id].detach().cpu().numpy().copy()
      start[2] += 0.5
      desired_end = start + 1.5 * desired_w[env_id].detach().cpu().numpy()
      measured_end = start + 1.5 * measured_w[env_id].detach().cpu().numpy()
      visualizer.add_arrow(
        start,
        desired_end,
        (0.1, 0.9, 0.2, 0.9),
        label="desired base velocity",
      )
      visualizer.add_arrow(
        start + np.array([0.0, 0.0, 0.03]),
        measured_end + np.array([0.0, 0.0, 0.03]),
        (0.15, 0.35, 1.0, 0.9),
        label="measured base velocity",
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0
    self.low_level_actions[env_ids] = 0.0
    self._counter = 0
    self._low_level_obs_manager.reset(env_ids)
    self._low_level_action_term.reset(env_ids)


@dataclass(kw_only=True)
class PreTrainedPolicyActionCfg(ActionTermCfg):
  policy_path: str
  low_level_decimation: int
  low_level_actions: ActionTermCfg
  low_level_observations: ObservationGroupCfg
  velocity_clip: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
  debug_vis: bool = False

  def build(self, env) -> PreTrainedPolicyAction:
    return PreTrainedPolicyAction(self, env)
