"""Performance-gated curriculum contracts for the flat residual-effort task."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.eff_velocity.mdp.curriculums import (
  FlatEffortCurriculum,
  FlatEffortStage,
)
from mjlab.utils.noise import UniformNoiseCfg


def _stage(
  name: str,
  *,
  push_velocity: float,
  standing: float,
  noise: float,
) -> FlatEffortStage:
  return {
    "name": name,
    "lin_vel_x": (0.0, 0.0) if push_velocity == 0.0 else (-0.2, 0.3),
    "lin_vel_y": (0.0, 0.0) if push_velocity == 0.0 else (-0.1, 0.1),
    "ang_vel_z": (0.0, 0.0) if push_velocity == 0.0 else (-0.1, 0.1),
    "rel_standing_envs": standing,
    "push_velocity": push_velocity,
    "observation_noise": {"joint_vel": (-noise, noise)},
    "foot_friction": (0.7, 1.0) if push_velocity == 0.0 else (0.5, 1.1),
    "encoder_bias": (-0.003, 0.003),
    "base_com": {0: (-0.005, 0.005)},
  }


def _make_term() -> tuple[FlatEffortCurriculum, Any, dict[str, Any]]:
  num_envs = 20
  command_cfg = SimpleNamespace(
    ranges=SimpleNamespace(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-0.5, 0.5),
    ),
    rel_standing_envs=0.1,
  )
  command_term = SimpleNamespace(cfg=command_cfg)
  command_manager = Mock()
  command_manager.get_term.return_value = command_term

  event_cfgs = {
    "push_robot": SimpleNamespace(params={"velocity_range": {}}),
    "foot_friction": SimpleNamespace(params={"ranges": (0.3, 1.2)}),
    "encoder_bias": SimpleNamespace(params={"bias_range": (-0.015, 0.015)}),
    "base_com": SimpleNamespace(params={"ranges": {0: (-0.025, 0.025)}}),
  }
  event_manager = Mock()
  event_manager.get_term_cfg.side_effect = event_cfgs.__getitem__

  observation_cfg = ObservationTermCfg(
    func=lambda env: torch.zeros(env.num_envs, 1),
    noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
  )
  observation_manager = Mock()
  observation_manager.get_term_cfg.return_value = observation_cfg

  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    step_dt=0.02,
    episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    termination_manager=SimpleNamespace(
      time_outs=torch.zeros(num_envs, dtype=torch.bool)
    ),
    command_manager=command_manager,
    event_manager=event_manager,
    observation_manager=observation_manager,
  )
  params = {
    "command_name": "twist",
    "push_event_name": "push_robot",
    "stages": [
      _stage("balance", push_velocity=0.0, standing=1.0, noise=0.3),
      _stage("slow", push_velocity=0.1, standing=0.5, noise=0.75),
    ],
    "min_episodes": num_envs,
    "promotion_timeout_threshold": 0.95,
    "demotion_timeout_threshold": 0.5,
  }
  cfg = CurriculumTermCfg(func=FlatEffortCurriculum, params=params)
  mock_env = cast(ManagerBasedRlEnv, env)
  return FlatEffortCurriculum(cfg, mock_env), env, params


def _set_episode_window(
  env: Any,
  *,
  timeout_count: int,
) -> None:
  env.episode_length_buf[:] = 1000
  env.termination_manager.time_outs[:] = False
  env.termination_manager.time_outs[:timeout_count] = True


def test_flat_effort_curriculum_advances_from_episode_performance() -> None:
  term, env, params = _make_term()
  _set_episode_window(env, timeout_count=20)

  state = term(env, torch.arange(env.num_envs), **params)

  assert state["stage"] == 1.0
  assert state["timeout_ratio"] == pytest.approx(1.0)
  assert state["push_velocity"] == pytest.approx(0.1)

  command_cfg = env.command_manager.get_term("twist").cfg
  assert command_cfg.ranges.lin_vel_x == (-0.2, 0.3)
  assert command_cfg.rel_standing_envs == 0.5
  assert env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["x"] == (
    -0.1,
    0.1,
  )
  noise_cfg = env.observation_manager.get_term_cfg("actor", "joint_vel").noise
  assert isinstance(noise_cfg, UniformNoiseCfg)
  assert noise_cfg.n_min == pytest.approx(-0.75)
  assert noise_cfg.n_max == pytest.approx(0.75)
  env.event_manager.apply.assert_not_called()


def test_flat_effort_curriculum_requires_more_than_ninety_five_percent_timeouts() -> (
  None
):
  term, env, params = _make_term()
  _set_episode_window(env, timeout_count=19)

  state = term(env, torch.arange(env.num_envs), **params)

  assert state["stage"] == 0.0
  assert state["timeout_ratio"] == pytest.approx(0.95)
  assert state["window_episodes"] == 0.0
  env.event_manager.apply.assert_not_called()


def test_flat_effort_curriculum_demotes_below_fifty_percent_timeouts() -> None:
  term, env, params = _make_term()
  _set_episode_window(env, timeout_count=20)
  assert term(env, torch.arange(env.num_envs), **params)["stage"] == 1.0

  _set_episode_window(env, timeout_count=10)
  state = term(env, torch.arange(env.num_envs), **params)
  assert state["stage"] == 1.0
  assert state["timeout_ratio"] == pytest.approx(0.5)

  _set_episode_window(env, timeout_count=9)
  state = term(env, torch.arange(env.num_envs), **params)
  assert state["stage"] == 0.0
  assert state["timeout_ratio"] == pytest.approx(0.45)
  env.event_manager.apply.assert_not_called()
