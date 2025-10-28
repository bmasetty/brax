# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO vision networks."""

from typing import Any, Callable, Mapping, Sequence, Tuple, Literal

from brax.training import distribution
from brax.training import networks
from brax.training import types
import flax
from flax import linen
import jax.numpy as jp
import jax


ModuleDef = Any
ActivationFn = Callable[[jp.ndarray], jp.ndarray]
Initializer = Callable[..., Any]


@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_ppo_networks_vision(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.swish,
    normalise_channels: bool = False,
    policy_obs_key: str = "",
    value_obs_key: str = "",
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
    policy_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    value_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
    value_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
) -> PPONetworks:
  """Make Vision PPO networks with preprocessor."""

  policy_kernel_init_kwargs = policy_network_kernel_init_kwargs or {}
  value_kernel_init_kwargs = value_network_kernel_init_kwargs or {}

  parametric_action_distribution: distribution.ParametricDistribution
  if distribution_type == 'normal':
    parametric_action_distribution = distribution.NormalDistribution(
        event_size=action_size
    )
  elif distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )

  policy_network = networks.make_policy_network_vision(
      observation_size=observation_size,
      output_size=parametric_action_distribution.param_size,
      preprocess_observations_fn=preprocess_observations_fn,
      activation=activation,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      distribution_type=distribution_type,
      noise_std_type=noise_std_type,
      init_noise_std=init_noise_std,
      state_dependent_std=state_dependent_std,
      state_obs_key=policy_obs_key,
      normalise_channels=normalise_channels,
      kernel_init=policy_network_kernel_init_fn(**policy_kernel_init_kwargs),
  )

  value_network = networks.make_value_network_vision(
      observation_size=observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      activation=activation,
      hidden_layer_sizes=value_hidden_layer_sizes,
      distribution_type='tanh_normal',
      state_obs_key=value_obs_key,
      normalise_channels=normalise_channels,
      kernel_init=value_network_kernel_init_fn(**value_kernel_init_kwargs),
  )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )
