from __future__ import annotations

import torch
import torch.nn.functional as F

from mjlab.envs.mdp.observations import height_scan as _height_scan


def last_high_level_command(
  env, action_name: str = "pre_trained_policy_action"
) -> torch.Tensor:
  """Previous high-level velocity command after command clipping."""
  # 提示：高层策略需要观察真正传给低层策略的速度指令，而不是裁剪前的原始动作。
  # 可通过 action_manager 按名称取得 action term，并读取其 processed_action。
  # >>> HOMEWORK_TODO_1_START
  return env.action_manager.get_term(action_name).processed_action
  # <<< HOMEWORK_TODO_1_END


def low_level_last_action(
  env, action_name: str = "pre_trained_policy_action"
) -> torch.Tensor:
  """Last 29-DoF joint action emitted by the frozen low-level policy."""
  return env.action_manager.get_term(action_name).low_level_actions


def command_distance(env, command_name: str = "pose_command") -> torch.Tensor:
  return torch.norm(
    env.command_manager.get_command(command_name)[:, :2], dim=-1, keepdim=True
  )


def base_pos_z(env) -> torch.Tensor:
  return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def height_scan(
  env, sensor_name: str = "height_scanner", offset: float = 0.5
) -> torch.Tensor:
  return _height_scan(env, sensor_name=sensor_name, offset=offset)


def _height_scan_grid_shape(pattern) -> tuple[int, int]:
  """Return ``(ny, nx)`` for mjlab's ``meshgrid(indexing='xy')`` ordering."""
  nx = int(
    torch.arange(
      -pattern.size[0] / 2,
      pattern.size[0] / 2 + pattern.resolution * 0.5,
      pattern.resolution,
    ).numel()
  )
  ny = int(
    torch.arange(
      -pattern.size[1] / 2,
      pattern.size[1] / 2 + pattern.resolution * 0.5,
      pattern.resolution,
    ).numel()
  )
  return ny, nx


def _pool_height_scan(
  scan: torch.Tensor, ny: int, nx: int, pool_size: int = 2
) -> torch.Tensor:
  """Restore the ray grid as ``(ny, nx)``, max-pool it, then flatten."""
  # 提示：先取得扁平射线高度，再依据 (ny, nx) 恢复二维网格。
  # 为适配 F.max_pool2d，需要添加通道维；池化后再展平为 (num_envs, -1)。
  # >>> HOMEWORK_TODO_2_START
  # 1. 输入高度扫描扁平射线，形状为 (num_envs, num_rays)
  # 2. 根据网格形状 (ny, nx)，恢复为 (num_envs, ny, nx)
  height_scan_grid = scan.view(-1, ny, nx)
  # 3. 添加通道维度 (num_envs, 1, ny, nx)
  height_scan_grid = height_scan_grid.unsqueeze(1)
  # 4. 最大池化 (num_envs, 1, ny//pool_size, nx//pool_size)
  pooled_height_scan = F.max_pool2d(
    height_scan_grid, kernel_size=pool_size, stride=pool_size
  )
  # 5. 展平为 (num_envs, -1)，便于 MLP 输入
  return pooled_height_scan.flatten(1)
  # <<< HOMEWORK_TODO_2_END


def height_scan_pooled(
  env,
  sensor_name: str = "height_scanner",
  offset: float = 0.5,
  pool_size: int = 2,
) -> torch.Tensor:
  scan = _height_scan(env, sensor_name=sensor_name, offset=offset)
  pattern = env.scene.sensors[sensor_name].cfg.pattern
  ny, nx = _height_scan_grid_shape(pattern)
  return _pool_height_scan(scan, ny, nx, pool_size)
