"""RSL-RL models for residual effort policies."""

from __future__ import annotations

import copy
import math
from typing import cast

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


def _initialize_output_layer(module: nn.Module, gain: float) -> None:
  """Initialize the final linear layer around the residual-action origin."""
  linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
  if not linear_layers:
    raise ValueError("The policy head must contain an nn.Linear output layer.")
  output_layer = linear_layers[-1]
  nn.init.orthogonal_(output_layer.weight, gain=gain)
  nn.init.zeros_(output_layer.bias)


class ResidualMlpModel(MLPModel):
  """MLP policy with a small-gain orthogonal action output layer."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    output_gain: float = 0.01,
  ) -> None:
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    _initialize_output_layer(self.mlp, output_gain)


def _sinusoidal_position_encoding(length: int, dim: int) -> torch.Tensor:
  encoding = torch.zeros(1, length, dim)
  position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
  div_term = torch.exp(
    torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim)
  )
  encoding[0, :, 0::2] = torch.sin(position * div_term)
  encoding[0, :, 1::2] = torch.cos(position * div_term[: encoding[0, :, 1::2].shape[1]])
  return encoding


class HistoryMhaEncoder(nn.Module):
  """Encode past observations with a pre-norm self-attention residual block."""

  def __init__(
    self,
    input_dim: int,
    history_length: int,
    hidden_dim: int,
    num_heads: int,
    activation: str,
    learnable_pos_embedding: bool,
  ) -> None:
    super().__init__()
    self.input_projection = nn.Linear(input_dim, hidden_dim)
    self.activation = resolve_nn_activation(activation)
    positional_encoding = _sinusoidal_position_encoding(history_length, hidden_dim)
    if learnable_pos_embedding:
      self.positional_embedding = nn.Parameter(positional_encoding)
    else:
      self.register_buffer("positional_embedding", positional_encoding)
    self.pre_norm = nn.LayerNorm(hidden_dim)
    self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
    self.output_norm = nn.LayerNorm(hidden_dim)

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    hidden = self.activation(self.input_projection(observations))
    hidden = hidden + self.positional_embedding[:, : hidden.shape[1]]
    normalized = self.pre_norm(hidden)
    attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
    hidden = self.output_norm(hidden + attended)
    return hidden[:, -1]


class ResidualMhaModel(MLPModel):
  """4+1 MHA policy over native time-major mjlab observation history."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    output_gain: float = 0.01,
    history_length: int = 5,
    num_heads: int = 4,
    encoder_hidden_dim: int = 256,
    learnable_pos_embedding: bool = True,
  ) -> None:
    if history_length < 2:
      raise ValueError("ResidualMhaModel requires at least two observation frames.")
    if encoder_hidden_dim % num_heads != 0:
      raise ValueError("encoder_hidden_dim must be divisible by num_heads.")

    self.history_length = history_length
    self.encoder_hidden_dim = encoder_hidden_dim
    self.single_step_dim = 0
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    self.history_encoder = HistoryMhaEncoder(
      input_dim=self.single_step_dim,
      history_length=history_length - 1,
      hidden_dim=encoder_hidden_dim,
      num_heads=num_heads,
      activation=activation,
      learnable_pos_embedding=learnable_pos_embedding,
    )
    self.current_projection = nn.Linear(self.single_step_dim, encoder_hidden_dim)
    self.current_activation = resolve_nn_activation(activation)
    self.current_norm = nn.LayerNorm(encoder_hidden_dim)
    _initialize_output_layer(self.mlp, output_gain)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    sequence = torch.cat(
      [cast(torch.Tensor, obs[group]) for group in self.obs_groups], dim=-1
    )
    sequence = self.obs_normalizer(sequence)
    history_embedding = self.history_encoder(sequence[:, :-1])
    current_embedding = self.current_projection(sequence[:, -1])
    current_embedding = self.current_activation(current_embedding)
    current_embedding = self.current_norm(current_embedding)
    return torch.cat((current_embedding, history_embedding), dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      sequence = torch.cat(
        [cast(torch.Tensor, obs[group]) for group in self.obs_groups], dim=-1
      )
      flat_sequence = sequence.reshape(-1, self.single_step_dim)
      self.obs_normalizer.update(flat_sequence)  # type: ignore[union-attr]

  def as_jit(self) -> nn.Module:
    return _TorchResidualMhaModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxResidualMhaModel(self, verbose)

  def _get_obs_dim(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], int]:
    active_obs_groups = obs_groups[obs_set]
    single_step_dim = 0
    for obs_group in active_obs_groups:
      shape = obs[obs_group].shape
      if len(shape) != 3:
        raise ValueError(
          "ResidualMhaModel expects [batch, history, features] observations, "
          f"received {shape} for '{obs_group}'."
        )
      if shape[-2] != self.history_length:
        raise ValueError(
          f"Expected {self.history_length} history frames for '{obs_group}', "
          f"received {shape[-2]}."
        )
      single_step_dim += shape[-1]
    self.single_step_dim = single_step_dim
    return active_obs_groups, single_step_dim

  def _get_latent_dim(self) -> int:
    return 2 * self.encoder_hidden_dim


class _ResidualMhaExportCore(nn.Module):
  def __init__(self, model: ResidualMhaModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.history_encoder = copy.deepcopy(model.history_encoder)
    self.current_projection = copy.deepcopy(model.current_projection)
    self.current_activation = copy.deepcopy(model.current_activation)
    self.current_norm = copy.deepcopy(model.current_norm)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is None:
      self.deterministic_output = nn.Identity()
    else:
      self.deterministic_output = model.distribution.as_deterministic_output_module()

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    sequence = self.obs_normalizer(observations)
    history_embedding = self.history_encoder(sequence[:, :-1])
    current_embedding = self.current_projection(sequence[:, -1])
    current_embedding = self.current_activation(current_embedding)
    current_embedding = self.current_norm(current_embedding)
    latent = torch.cat((current_embedding, history_embedding), dim=-1)
    return self.deterministic_output(self.mlp(latent))


class _TorchResidualMhaModel(_ResidualMhaExportCore):
  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxResidualMhaModel(_ResidualMhaExportCore):
  is_recurrent = False

  def __init__(self, model: ResidualMhaModel, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose
    self.history_length = model.history_length
    self.single_step_dim = model.single_step_dim

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.history_length, self.single_step_dim),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
