from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class FlatEffortStage(TypedDict):
  """One performance-gated stage of the flat residual-effort curriculum."""

  name: str
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]
  rel_standing_envs: float
  push_velocity: float
  observation_noise: dict[str, tuple[float, float]]
  foot_friction: tuple[float, float]
  encoder_bias: tuple[float, float]
  base_com: dict[int, tuple[float, float]]


class FlatEffortCurriculum(ManagerTermBase):
  """Advance flat torque-control difficulty from completed-episode performance."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._command_name = cast(str, cfg.params["command_name"])
    self._push_event_name = cast(str, cfg.params["push_event_name"])
    self._stages = tuple(cast(list[FlatEffortStage], cfg.params["stages"]))
    self._min_episodes = cast(int, cfg.params["min_episodes"])
    self._promotion_timeout_threshold = cast(
      float, cfg.params["promotion_timeout_threshold"]
    )
    self._demotion_timeout_threshold = cast(
      float, cfg.params["demotion_timeout_threshold"]
    )

    self._stage = 0
    self._window_episodes = 0
    self._window_timeouts = 0
    self._last_timeout_ratio = 0.0
    self._apply_stage()

  def _apply_stage(self) -> None:
    stage = self._stages[self._stage]

    command_term = cast(
      UniformVelocityCommand,
      self._env.command_manager.get_term(self._command_name),
    )
    command_cfg = command_term.cfg
    command_cfg.ranges.lin_vel_x = stage["lin_vel_x"]
    command_cfg.ranges.lin_vel_y = stage["lin_vel_y"]
    command_cfg.ranges.ang_vel_z = stage["ang_vel_z"]
    command_cfg.rel_standing_envs = stage["rel_standing_envs"]

    push_cfg = self._env.event_manager.get_term_cfg(self._push_event_name)
    push_velocity = stage["push_velocity"]
    push_cfg.params["velocity_range"] = {
      "x": (-push_velocity, push_velocity),
      "y": (-push_velocity, push_velocity),
      "z": (0.0, 0.0),
      "roll": (0.0, 0.0),
      "pitch": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }

    for term_name, (n_min, n_max) in stage["observation_noise"].items():
      term_cfg = self._env.observation_manager.get_term_cfg("actor", term_name)
      noise = term_cfg.noise
      operation = noise.operation if isinstance(noise, UniformNoiseCfg) else "add"
      term_cfg.noise = UniformNoiseCfg(
        n_min=n_min,
        n_max=n_max,
        operation=operation,
      )

    self._env.event_manager.get_term_cfg("foot_friction").params["ranges"] = stage[
      "foot_friction"
    ]
    self._env.event_manager.get_term_cfg("encoder_bias").params["bias_range"] = stage[
      "encoder_bias"
    ]
    self._env.event_manager.get_term_cfg("base_com").params["ranges"] = dict(
      stage["base_com"]
    )

  def _accumulate_completed_episodes(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
  ) -> None:
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

    completed_ids = env_ids[env.episode_length_buf[env_ids] > 0]
    if len(completed_ids) == 0:
      return

    self._window_episodes += len(completed_ids)
    self._window_timeouts += int(
      torch.count_nonzero(env.termination_manager.time_outs[completed_ids]).item()
    )

  def _evaluate_window(self) -> None:
    if self._window_episodes < self._min_episodes:
      return

    self._last_timeout_ratio = self._window_timeouts / self._window_episodes

    if (
      self._last_timeout_ratio > self._promotion_timeout_threshold
      and self._stage < len(self._stages) - 1
    ):
      self._stage += 1
      self._apply_stage()
    elif (
      self._last_timeout_ratio < self._demotion_timeout_threshold and self._stage > 0
    ):
      self._stage -= 1
      self._apply_stage()

    self._window_episodes = 0
    self._window_timeouts = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    command_name: str,
    push_event_name: str,
    stages: list[FlatEffortStage],
    min_episodes: int,
    promotion_timeout_threshold: float,
    demotion_timeout_threshold: float,
  ) -> dict[str, float]:
    del (
      command_name,
      push_event_name,
      stages,
      min_episodes,
      promotion_timeout_threshold,
      demotion_timeout_threshold,
    )
    self._accumulate_completed_episodes(env, env_ids)
    self._evaluate_window()
    return {
      "stage": float(self._stage),
      "timeout_ratio": self._last_timeout_ratio,
      "window_episodes": float(self._window_episodes),
      "push_velocity": self._stages[self._stage]["push_velocity"],
    }


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }
