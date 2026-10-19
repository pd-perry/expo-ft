"""EXPO-FT learner: Pi0.5 base policy + edit actor + critic (EXPOLearner)."""

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import dataclasses

import logging

import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import struct
from flax.training.train_state import TrainState

import numpy as np

from expo_ft.agents.alg.agent import AgentLearner, initialize_checkpoint_dir
from expo_ft.agents.alg.batch_utils import prepare_critic_batch, prepare_actor_sampling_batch, extract_critic_fields, CRITIC_CAMERA_KEYS
from expo_ft.networks.temperature import Temperature
from expo_ft.data.dataset import DatasetDict
from expo_ft.distributions import TanhNormal
from expo_ft.networks import (
    MLP,
    Ensemble,
    MLPResNetV2,
    StateActionValue,
    subsample_image_ensemble,
    PixelMultiplexer,
    PixelEditMultiplexer,
    BatchEncoder,
)
from expo_ft.networks.encoders import ResNetV2Encoder

from expo_ft.utils.augmentation import make_data_augmentation_fn

import openpi.shared.array_typing as at
import openpi.training.sharding as _sharding
import openpi.training.utils as training_utils


def _split_params(agent: Any) -> tuple[Any, dict[str, at.Params]]:
    batch_encoder_params = agent.batch_encoder.params
    edit_actor_params = agent.edit_actor.params
    temp_params = agent.temp.params
    critic_params = agent.critic.params
    
    with at.disable_typechecking():
        if agent.actor_train_state.ema_params is not None:
            actor_params = agent.actor_train_state.ema_params
            actor_train_state = dataclasses.replace(agent.actor_train_state, ema_params=None)
        else:
            actor_params = agent.actor_train_state.params
            actor_train_state = dataclasses.replace(agent.actor_train_state, params={})

    agent = dataclasses.replace(
        agent, 
        batch_encoder=dataclasses.replace(agent.batch_encoder, params={}),
        edit_actor=dataclasses.replace(agent.edit_actor, params={}), 
        temp=dataclasses.replace(agent.temp, params={}), 
        critic=dataclasses.replace(agent.critic, params={}),
        actor_train_state=actor_train_state
    )

    params = {
        "batch_encoder_params": batch_encoder_params,
        "edit_actor_params": edit_actor_params, 
        "temp_params": temp_params, 
        "critic_params": critic_params,
        "actor_params": actor_params
    }
    return agent, params


def _merge_params(agent: Any, params: dict[str, at.Params]) -> Any:
    batch_encoder = dataclasses.replace(agent.batch_encoder, params=params["batch_encoder_params"])
    edit_actor = dataclasses.replace(agent.edit_actor, params=params["edit_actor_params"])
    temp = dataclasses.replace(agent.temp, params=params["temp_params"])
    critic = dataclasses.replace(agent.critic, params=params["critic_params"])

    with at.disable_typechecking():
        if agent.actor_train_state.params:
            actor_train_state = dataclasses.replace(agent.actor_train_state, ema_params=params["actor_params"])
        else:
            actor_train_state = dataclasses.replace(agent.actor_train_state, params=params["actor_params"])

    agent = dataclasses.replace(
        agent, 
        batch_encoder=batch_encoder,
        edit_actor=edit_actor,
        temp=temp,
        critic=critic,
        actor_train_state=actor_train_state,
    )
    return agent


def restore_checkpoint(checkpoint_manager, agent, step: int | None = None):
    agent, params = _split_params(agent)
    # Explicit restore_args from the live (current-topology) shardings, so a
    # checkpoint saved on a different device count can be resharded at load.
    agent_restore_args = ocp.checkpoint_utils.construct_restore_args(agent)
    params_restore_args = ocp.checkpoint_utils.construct_restore_args(params)
    restored = checkpoint_manager.restore(
        step,
        items={
            "agent": agent,
            "params": params,
        },
        restore_kwargs={
            "agent": {"restore_args": agent_restore_args},
            "params": {"restore_args": params_restore_args},
        },
    )
    return _merge_params(restored["agent"], restored["params"])

def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    agent: Any,
    step: int,
):
    agent, params = _split_params(agent)
    items = {
        "agent": agent,
        "params": params,
    }
    checkpoint_manager.save(step, items)

def load_agent(seed, example_observation, example_action, example_state,
               actor, actor_train_state, target_actor_params, agent_kwargs, metadata,
               mesh, data_sharding, replicated_sharding, resume, replan_steps,
               default_prompt, edit_action_xyzg):
    """Create an EXPOLearner from a pre-built VLA actor and remaining config kwargs."""
    agent_kwargs.update(
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        mesh=mesh,
        resume=resume,
        replan_steps=replan_steps,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt=default_prompt,
        edit_action_xyzg=edit_action_xyzg,
        **metadata,
    )
    return EXPOLearner.create(seed, example_observation, example_action, example_state, **agent_kwargs)

def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))

@partial(jax.jit, static_argnames=('critic_fn', 'num_min_qs'))
def compute_q(critic_fn, critic_params, observations, actions, states, num_min_qs=None):
    q_values = critic_fn({'params': critic_params}, observations, actions, p=states, sample_num=num_min_qs)
    q_values = q_values.min(axis=0)
    return q_values


@partial(jax.jit, static_argnames=('encoder_fn', 'stop_gradient'))
def batch_encode(encoder_fn, encoder_params, observations, stop_gradient=False):
    encoded = encoder_fn({'params': encoder_params}, observations, stop_gradient=stop_gradient)
    return encoded


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: jnp.ndarray, states, actions) -> jnp.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations, actions=actions, p=states)
    return dist.sample(seed=key), rng


class EXPOLearner(AgentLearner, struct.PyTreeNode):
    rng: jax.random.PRNGKey
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    critic: TrainState
    batch_encoder: TrainState
    target_critic: TrainState
    actor: Any = struct.field(pytree_node=False)
    actor_train_state: training_utils.TrainState
    target_actor_params: at.Params
    edit_actor: TrainState
    temp: TrainState
    N: int = struct.field(pytree_node=False)
    n_edit_samples: int = struct.field(pytree_node=False)
    edit_scale: float = struct.field(pytree_node=False)
    edit_action_xyzg: bool = struct.field(pytree_node=False)
    batch_split: int = struct.field(pytree_node=False)
    encode_batch_split: int = struct.field(pytree_node=False)
    actor_tau: float
    tau: float
    discount: float
    target_entropy: float
    entropy_scale: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    action_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    full_action_dim: int = struct.field(pytree_node=False)
    replan_steps: int = struct.field(pytree_node=False)
    action_horizon: int = struct.field(pytree_node=False)
    resize_size: Optional[int] = struct.field(pytree_node=False)
    default_prompt: Optional[str] = struct.field(pytree_node=False)
    data_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    batch_encoder_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    freeze_encoder: Optional[bool] = struct.field(pytree_node=False)
    freeze_critic_encoder: bool = struct.field(pytree_node=False)
    actor_success_only: bool = struct.field(pytree_node=False)
    _infer_cache: Optional[dict] = struct.field(pytree_node=False, default=None)
    critic_camera_keys: Tuple[str, ...] = struct.field(pytree_node=False, default=CRITIC_CAMERA_KEYS)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        states,
        # Pre-built VLA actor (constructed by build_pi05 or similar factory)
        actor: Any = None,
        actor_train_state: Any = None,
        target_actor_params: Any = None,
        # VLA metadata (extracted by factory from backbone config)
        action_horizon: int = 1,
        mesh: Optional[Any] = None,
        freeze_encoder: bool = False,
        # EXPO-specific params
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        adjust_target_entropy: Optional[bool] = False,
        entropy_scale: float = 1.0,
        init_temperature: float = 1.0,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        actor_drop: Optional[float] = None,
        N: int = 32,
        batch_split: int = 1,
        encode_batch_split: int = 1,
        n_edit_samples: int = 0,
        edit_scale: float = 1.0,
        edit_action_xyzg: bool = False,
        actor_tau: float = 0.001,
        include_state: bool = True,
        latent_dim_image: int = 50,
        latent_dim_state: int = 50,
        encoder_stage_sizes: Tuple[int, int, int, int] = (2, 2, 2, 2),
        encoder_num_filters: int = 64,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
        resume: bool = False,
        replan_steps: int = 1,
        freeze_critic_encoder: bool = False,
        data_sharding: Optional[jax.sharding.NamedSharding] = None,
        replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
        default_prompt: Optional[str] = None,
        resize_size: Optional[int] = None,
        actor_success_only: bool = False,
        use_full_augmentation: bool = True,
        critic_camera_keys: Tuple[str, ...] = CRITIC_CAMERA_KEYS,
        **kwargs,
    ):
        action_dim = action_space.shape[-1]
        state_dim = states.shape[-1]
        full_action_dim = replan_steps * action_dim
        observations = observation_space
        actions = jnp.zeros((full_action_dim,))
        print("observation shape: ", observations.shape)
        print("action shape: ", actions.shape, "action horizon: ", action_horizon, "action dim: ", action_dim)
        print("edit actor output size / q function input size: ", full_action_dim, " (replan_steps=", replan_steps, "* action_dim=", action_dim, ")")
        print("states shape: ", states.shape)

        if target_entropy is None:
            if adjust_target_entropy:
                target_entropy = -full_action_dim / 2 + full_action_dim * jnp.log(edit_scale)
            else:
                target_entropy = -full_action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        rng, encoder_key = jax.random.split(rng, 2)

        encoder_cls = partial(
            ResNetV2Encoder,
            stage_sizes=encoder_stage_sizes,
            num_filters=encoder_num_filters,
        )

        batch_encoder_def = BatchEncoder(
            encoder_cls=encoder_cls,
            latent_dim=latent_dim_image,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )

        batch_encoder_params = batch_encoder_def.init(encoder_key, observations)["params"]
        batch_encoder = TrainState.create(
            apply_fn=batch_encoder_def.apply,
            params=batch_encoder_params,
            tx=optax.adam(learning_rate=critic_lr),
        )

        batch_encoder_shape = jax.eval_shape(lambda: batch_encoder)
        batch_encoder_sharding = _sharding.fsdp_sharding(batch_encoder_shape, mesh, log=True)
        batch_encoder = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=batch_encoder_sharding,
        )(batch_encoder)

        critic_observations = jnp.ones((1, latent_dim_image))
        critic_actions = jnp.expand_dims(actions, axis = 0)
        critic_states = jnp.expand_dims(states, axis = 0)
        critic_states_ext = critic_states
        print("critic actions shape: ", critic_actions.shape)
        print("critic states shape: ", critic_states.shape)

        edit_actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm
        )
        edit_actor_cls= TanhNormal(edit_actor_base_cls, full_action_dim)
        edit_actor_def = PixelEditMultiplexer(
            network_cls=edit_actor_cls,
            latent_dim=latent_dim_state,
            include_state=include_state,
        )

        edit_actor_params = edit_actor_def.init(actor_key, jnp.ones((1, latent_dim_image)), actions=jnp.ones((1, full_action_dim)), p=critic_states)["params"]
        edit_actor = TrainState.create(
            apply_fn=edit_actor_def.apply, 
            params=edit_actor_params, 
            tx=optax.adam(learning_rate=actor_lr),
        )

        edit_actor_shape = jax.eval_shape(lambda: edit_actor)
        edit_actor_sharding = _sharding.fsdp_sharding(edit_actor_shape, mesh, log=True)
        edit_actor = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=edit_actor_sharding,
        )(edit_actor)

        if use_critic_resnet:
            critic_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate, 
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )

        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        critic_def = PixelMultiplexer(
            network_cls=critic_cls,
            latent_dim=latent_dim_state,
            include_state=include_state,
        )
        critic_params = critic_def.init(critic_key, critic_observations, critic_actions, p=critic_states_ext)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
            
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )

        critic_shape = jax.eval_shape(lambda: critic)
        critic_sharding = _sharding.fsdp_sharding(critic_shape, mesh, log=True)
        critic = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=critic_sharding,
        )(critic)

        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        target_critic_shape = jax.eval_shape(lambda: target_critic)
        target_critic_sharding = _sharding.fsdp_sharding(target_critic_shape, mesh, log=True)
        target_critic = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=target_critic_sharding,
        )(target_critic)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        temp_shape = jax.eval_shape(lambda: temp)
        temp_sharding = _sharding.fsdp_sharding(temp_shape, mesh, log=True)
        temp = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=temp_sharding,
        )(temp)


        agent = cls(
            rng=rng,
            actor=actor,
            actor_train_state=actor_train_state,
            target_actor_params=target_actor_params,
            edit_actor=edit_actor,
            N=N,
            n_edit_samples=n_edit_samples,
            encode_batch_split=encode_batch_split,
            edit_scale=edit_scale,
            edit_action_xyzg=edit_action_xyzg,
            batch_split=batch_split,
            actor_tau=actor_tau,
            critic=critic,
            target_critic=target_critic,
            batch_encoder=batch_encoder,
            temp=temp,
            target_entropy=target_entropy,
            entropy_scale=entropy_scale,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            data_augmentation_fn=make_data_augmentation_fn(use_full_augmentation),

            action_dim=action_dim,
            state_dim=state_dim,
            full_action_dim=full_action_dim,
            action_horizon=action_horizon,
            replan_steps=replan_steps,
            resize_size=resize_size,
            default_prompt=default_prompt,
            data_sharding=data_sharding,
            batch_encoder_sharding=batch_encoder_sharding,
            freeze_encoder=freeze_encoder,
            freeze_critic_encoder=freeze_critic_encoder,
            actor_success_only=actor_success_only,
            critic_camera_keys=tuple(critic_camera_keys),
        )
        if not resume:
            agent = agent.cache_infer_params()
        return agent

    def _apply_edit_xyzg_mask(self, edit: jnp.ndarray) -> jnp.ndarray:
        """Zero out rotation dims (3,4,5) when edit_action_xyzg is True. Keeps xyz (0,1,2) and gripper (6)."""
        if not self.edit_action_xyzg:
            return edit
        # mask: 1 for xyz (0,1,2) and gripper (last dim), 0 for rotation (3,4,5)
        mask = jnp.ones(self.action_dim).at[3:6].set(0.0)
        full_mask = jnp.tile(mask, self.replan_steps)
        return edit * full_mask

    def cache_infer_params(self):
        """Copy params onto infer_sharding for rollout sampling.

        sample_actions reads _infer_cache to avoid device_put on every env step.
        Call again after update() so rollouts use the latest weights.
        """
        s = self.actor.infer_sharding
        return self.replace(_infer_cache={
            "actor_train_state": jax.device_put(self.actor_train_state, s),
            "batch_encoder_params": jax.device_put(self.batch_encoder.params, s),
            "edit_actor_params": jax.device_put(self.edit_actor.params, s),
            "target_critic_params": jax.device_put(self.target_critic.params, s),
        })

    def _sample_edit(self, key, edit_params, encoded_obs, states, base_actions):
        r_samples, rng = _sample_actions(
            key, self.edit_actor.apply_fn, edit_params, encoded_obs, states, base_actions
        )
        edit_scaled = self._apply_edit_xyzg_mask(r_samples * self.edit_scale)
        combined = edit_scaled + base_actions
        return combined, edit_scaled, rng

    def _encode_observations(self, observations, encoder_params=None, stop_gradient=True):
        params = encoder_params or self.batch_encoder.params
        if self.encode_batch_split > 1:
            one_call = observations.shape[0] // self.encode_batch_split
            chunks = [
                batch_encode(self.batch_encoder.apply_fn, params, observations[i * one_call:(i + 1) * one_call], stop_gradient=stop_gradient)
                for i in range(self.encode_batch_split)
            ]
            return jnp.concatenate(chunks)
        return batch_encode(self.batch_encoder.apply_fn, params, observations, stop_gradient=stop_gradient)

    def _compute_q_split(self, critic_fn, critic_params, obs, actions, states):
        if self.batch_split > 1:
            total = obs.shape[0]
            one_call = total // self.batch_split
            q_list = [
                compute_q(critic_fn, critic_params, obs[i * one_call:(i + 1) * one_call], actions[i * one_call:(i + 1) * one_call], states[i * one_call:(i + 1) * one_call], self.num_min_qs)
                for i in range(self.batch_split)
            ]
            return jnp.concatenate(q_list)
        return compute_q(critic_fn, critic_params, obs, actions, states, self.num_min_qs)

    def sample_actions(self, observations, only_base_actions=False):
        # Keep inference-time randomness on the same single device as inference params
        # to avoid cross-device gather/indexing mismatches.
        infer_sharding = self.actor.infer_sharding
        rng = jax.device_put(self.rng, infer_sharding)
        c = self._infer_cache or {}
        _actor_train_state = c.get("actor_train_state") or self.actor_train_state
        _batch_encoder_params = c.get("batch_encoder_params") or jax.device_put(self.batch_encoder.params, infer_sharding)
        _edit_actor_params = c.get("edit_actor_params") or jax.device_put(self.edit_actor.params, infer_sharding)
        _target_critic_params = c.get("target_critic_params") or jax.device_put(self.target_critic.params, infer_sharding)

        transformed_inputs = self.actor.process_raw_inputs(observations, self.action_dim, self.resize_size)
        transformed_inputs = extract_critic_fields(transformed_inputs, self.actor.model_config.action_dim, self.state_dim, self.critic_camera_keys)

        key, rng = jax.random.split(rng)
        transformed_actions, sample_time = self.actor.sample_actions(
            transformed_inputs,
            train_state=_actor_train_state,
            rng=key,
            train=False,
            num_samples=self.N,
        )
        raw_actions = self.actor.process_transformed_outputs(transformed_actions)
        
        if only_base_actions:
            action = raw_actions[0].reshape(self.action_horizon, self.action_dim)
            sample_info = {"sample_time": sample_time, "selected_action_type": "main"}
            return jnp.array(action), self.replace(rng=rng), sample_info

        transformed_full = transformed_actions  # (N, action_horizon, action_dim)
        transformed_actions = transformed_full[:, :self.replan_steps, :].reshape(self.N, self.full_action_dim)

        critic_transformed_obs = transformed_inputs["critic_obs"]
        transformed_states = transformed_inputs["critic_states"]
        
        if self.freeze_encoder:
            critic_transformed_obs = critic_transformed_obs.repeat(self.N, axis=0)
            transformed_states = transformed_states.repeat(self.N, axis=0)

        critic_encoded_obs = batch_encode(self.batch_encoder.apply_fn, _batch_encoder_params, critic_transformed_obs[:1], stop_gradient=True)
        critic_encoded_obs = critic_encoded_obs.repeat(self.N, axis=0)

        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_image_ensemble(
                key, _target_critic_params, self.num_min_qs, self.num_qs
            )

            if self.n_edit_samples > 0:
                key, rng = jax.random.split(rng, 2)

                critic_encoded_obs = jnp.concatenate([critic_encoded_obs, jnp.expand_dims(critic_encoded_obs[0], axis = 0).repeat(self.n_edit_samples, axis = 0)], axis=0)
                transformed_states = jnp.concatenate([transformed_states, jnp.expand_dims(transformed_states[0], axis = 0).repeat(self.n_edit_samples, axis = 0)], axis=0)

                r_observations = jnp.repeat(jnp.expand_dims(critic_encoded_obs[0], axis = 0), self.n_edit_samples, axis=0)
                r_states = jnp.repeat(jnp.expand_dims(transformed_states[0], axis = 0), self.n_edit_samples, axis=0)
                base_actions = transformed_actions.copy()[:self.n_edit_samples]

                r_samples, _, rng = self._sample_edit(key, _edit_actor_params, r_observations, r_states, base_actions)
                transformed_actions = jnp.concatenate([base_actions, r_samples], axis=0)

                r_modified = r_samples.reshape(self.n_edit_samples, self.replan_steps, self.action_dim)
                full_r_modified = transformed_full[:self.n_edit_samples].at[:, :self.replan_steps, :].set(r_modified)
                raw_r_samples = self.actor.process_transformed_outputs(full_r_modified)
                raw_actions = jnp.concatenate([raw_actions, raw_r_samples], axis=0)

            qs = compute_q(self.target_critic.apply_fn, target_params, critic_encoded_obs, transformed_actions, transformed_states, self.num_min_qs)

            idx = jnp.argmax(qs)

            action = raw_actions[idx]
        else:
            action = raw_actions[0]
            idx = 0

        action = action.reshape(self.action_horizon, self.action_dim)
        rng, _ = jax.random.split(rng, 2)
        sample_info = {"sample_time": sample_time}
        return jnp.array(action), self.replace(rng=rng), sample_info

    def sample_batch_actions(self, batch):
        critic_obs = jnp.squeeze(batch["next_observations"])
        critic_obs = jax.device_put(critic_obs)
        batch_size = critic_obs.shape[0]
        states = jnp.squeeze(batch["next_states"])
        states = jax.device_put(states, self.data_sharding)
        rng = self.rng

        # Prepare VLA inputs
        transformed_inputs = prepare_actor_sampling_batch(batch)
        if not self.freeze_encoder:
            def repeat_value(v, n):
                if v is None:
                    return None
                elif isinstance(v, dict):
                    return {k: repeat_value(vv, n) for k, vv in v.items()}
                else:
                    return jnp.repeat(v, n, axis=0)
            transformed_inputs = {k: repeat_value(v, self.N) for k, v in transformed_inputs.items()}

        # Encode observations
        encoded_obs = self._encode_observations(critic_obs, stop_gradient=True)
        encoded_obs = jax.device_put(encoded_obs, self.data_sharding)

        # Sample base actions from VLA
        key, rng = jax.random.split(rng)
        actor_actions, sample_time = self.actor.sample_training_actions(
            transformed_inputs=transformed_inputs,
            train_state=self.actor_train_state,
            rng=key,
            train=False,
            num_samples=self.N if self.freeze_encoder else 1,
        )
        actor_actions = actor_actions[:, :self.replan_steps, :]
        actions = actor_actions.reshape(batch_size, self.N, self.full_action_dim)

        # Sample edit actions
        total_candidates = self.N + self.n_edit_samples

        if self.n_edit_samples > 0:
            key, rng = jax.random.split(rng, 2)
            r_observations = jax.device_put(jnp.repeat(encoded_obs, self.n_edit_samples, axis=0), self.data_sharding)
            r_states = jax.device_put(jnp.repeat(states, self.n_edit_samples, axis=0), self.data_sharding)
            d_actions = actions.copy()[:, :self.n_edit_samples].reshape(-1, actions.shape[-1])

            r_samples, edit_scaled, rng = self._sample_edit(key, self.edit_actor.params, r_observations, r_states, d_actions)
            mean_d_actions_norm = jnp.mean(jnp.linalg.norm(d_actions, axis=1))
            mean_edit_scaled_norm = jnp.mean(jnp.linalg.norm(edit_scaled, axis=1))

            actions = jnp.concatenate([actions, r_samples.reshape(batch_size, self.n_edit_samples, -1)], axis=1)
        
        else:
            mean_d_actions_norm = jnp.array(0.0, dtype=jnp.float32)
            mean_edit_scaled_norm = jnp.array(0.0, dtype=jnp.float32)

        # Select best actions via Q-values
        if self.N > 1:
            key, rng = jax.random.split(rng)
            target_params = subsample_image_ensemble(
                key, self.target_critic.params, self.num_min_qs, self.num_qs
            )

            obs_flat = jax.device_put(jnp.repeat(encoded_obs, total_candidates, axis=0), self.data_sharding)
            states_flat = jax.device_put(jnp.repeat(states, total_candidates, axis=0), self.data_sharding)
            actions_flat = jax.device_put(actions.reshape(-1, actions.shape[-1]), self.data_sharding)

            qs = self._compute_q_split(self.target_critic.apply_fn, target_params, obs_flat, actions_flat, states_flat)
            qs = qs.reshape(batch_size, total_candidates)

            best_indices = jnp.argmax(qs, axis=1)

            batch_indices = jnp.arange(batch_size)
            best_actions = actions[batch_indices, best_indices]

            without_edit_mask = best_indices < self.N
            with_edit_mask = (best_indices >= self.N) & (best_indices < total_candidates)
            vf_select_ratio_without_edit = jnp.mean(without_edit_mask.astype(jnp.float32))
            vf_select_ratio_with_edit = jnp.mean(with_edit_mask.astype(jnp.float32))
        else:
            best_actions = actions[:, 0]
            vf_select_ratio_without_edit = 1
            vf_select_ratio_with_edit = 0

        sample_info_extra = {
            "select_ratio_without_edit": vf_select_ratio_without_edit,
            "select_ratio_with_edit": vf_select_ratio_with_edit,
            "mean_d_actions_norm": mean_d_actions_norm,
            "mean_edit_scaled_norm": mean_edit_scaled_norm,
        }

        rng, _ = jax.random.split(rng, 2)
        return jnp.array(best_actions.squeeze()), sample_info_extra, rng
         
    def update_edit_actor(self, batch: DatasetDict) -> Tuple[AgentLearner, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        dropout_key, rng = jax.random.split(rng)

        def edit_actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:

            observations = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["observations"], stop_gradient=True)
            
            # Apply sharding constraint to encoded observations
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)
            dist = self.edit_actor.apply_fn({"params": actor_params}, observations, actions=batch["actions"], training=True, p=batch['states'], rngs={"dropout": dropout_key},)
            actions = dist.sample(seed=key)

            log_probs = dist.log_prob(actions)
            edit_scaled = self._apply_edit_xyzg_mask(actions * self.edit_scale)
            # Subtract log of action scale for each action dimension
            log_probs -= actions.shape[-1] * jnp.log(self.edit_scale)

            actions = edit_scaled + batch["actions"]

            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                observations,
                actions,
                True,
                p=batch['critic_states'], 
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            edit_actor_loss = (
                self.entropy_scale * log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return edit_actor_loss, {"edit_q": q.mean(), "edit_actor_loss": edit_actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(edit_actor_loss_fn, has_aux=True)(self.edit_actor.params)
        edit_actor = self.edit_actor.apply_gradients(grads=grads)

        return self.replace(edit_actor=edit_actor, rng=rng), actor_info
    
    def update_actor(self, batch: DatasetDict) -> Tuple[AgentLearner, Dict[str, float]]:
        actor_batch = self.actor.prepare_batch_for_actor(batch)
        
        # Ensure actor_batch has correct sharding
        def ensure_sharding(x):
            if isinstance(x, jnp.ndarray):
                return jax.device_put(x, self.data_sharding)
            return x
        actor_batch = jax.tree_util.tree_map(ensure_sharding, actor_batch)

        rng = self.rng
        key, rng = jax.random.split(rng, 2)
        
        with _sharding.set_mesh(self.actor.mesh):
            new_train_state, info = self.actor.train_step(key, self.actor_train_state, actor_batch)

        new_train_state_params = self.actor.get_params(new_train_state)
        target_score_params = optax.incremental_update(
            new_train_state_params, self.target_actor_params, self.actor_tau
        )

        new_agent = self.replace(actor_train_state=new_train_state, target_actor_params=target_score_params, rng=rng)
        
        return new_agent, info


    def update_temperature(self, entropy: float) -> Tuple[AgentLearner, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info


    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        next_actions, sample_info, rng = self.sample_batch_actions(batch)
        next_actions = jax.device_put(next_actions, self.data_sharding)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_image_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)

        next_observations = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, 
                                        batch["next_observations"], stop_gradient=True)
        next_observations = jax.device_put(next_observations, self.data_sharding)  

        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            next_observations,
            next_actions,
            False,  # training=False: target must be deterministic (no dropout)
            p=batch['next_critic_states'],
            sample_num=self.num_min_qs,
        )

        next_q_nan_mask = jnp.isnan(next_qs)
        next_q_nan_ratio = jnp.mean(next_q_nan_mask)
        next_qs = jnp.where(next_q_nan_mask, 0.0, next_qs)
        next_q = next_qs.min(axis=0)
        target_q = batch["rewards"] + (self.discount ** self.replan_steps) * batch["masks"] * next_q

        key, rng = jax.random.split(rng)

        params_dict = {"critic": self.critic.params}
        if not self.freeze_critic_encoder:
            params_dict["batch_encoder"] = self.batch_encoder.params

        def critic_loss_fn(params_dict) -> Tuple[jnp.ndarray, Dict[str, float]]:
            if self.freeze_critic_encoder:
                observations = batch_encode(
                    self.batch_encoder.apply_fn,
                    self.batch_encoder.params,
                    batch["observations"],
                    stop_gradient=True,
                )
            else:
                observations = batch_encode(
                    self.batch_encoder.apply_fn, params_dict["batch_encoder"], batch["observations"]
                )
            # Apply sharding constraint to encoded observations (works inside grad)
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)
            qs = self.critic.apply_fn(
                {"params": params_dict['critic']},
                observations,
                batch["actions"],
                True,
                p=batch['critic_states'], 
                rngs={"dropout": key},
            )
            critic_loss = (((qs - target_q) ** 2) * batch["valids"]).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
                "q_min": qs.min(),
                "q_max": qs.max(),
                "target_q_min": target_q.min(),
                "target_q_max": target_q.max(),
                "target_q_mean": target_q.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(params_dict)

        critic = self.critic.apply_gradients(grads=grads["critic"])

        if self.freeze_critic_encoder:
            batch_encoder = self.batch_encoder
        else:
            batch_encoder = self.batch_encoder.apply_gradients(grads=grads["batch_encoder"]) 

        critic_grad_norm = optax.global_norm(grads["critic"])
        info["critic_grad_norm"] = critic_grad_norm
        info["critic_param_norm"] = optax.global_norm(critic.params)
        info["next_q_nan_ratio"] = next_q_nan_ratio
        
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)
        info["target_critic_param_norm"] = optax.global_norm(target_critic_params)

        info.update(sample_info)

        return self.replace(critic=critic, target_critic=target_critic, batch_encoder=batch_encoder, rng=rng), info


    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        # Drop stale inference copies before JIT; rebuild after so rollouts use new weights.
        new_agent, info = self.replace(_infer_cache=None)._update_jit(
            agent.replace(_infer_cache=None), batch, utd_ratio, actor_batch
        )
        return new_agent.cache_infer_params(), info


    @partial(jax.jit, static_argnames="utd_ratio")
    def _update_jit(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        batch = batch.copy()
        rng, key1 = jax.random.split(agent.rng)
        rng, key2 = jax.random.split(rng)
        batch["image"] = self.data_augmentation_fn(key1, batch["image"])
        batch["next_image"] = self.data_augmentation_fn(key2, batch["next_image"])
        batch = prepare_critic_batch(batch, self.actor.model_config.action_dim, self.action_dim, self.state_dim, self.action_horizon, self.replan_steps, self.critic_camera_keys)
        new_agent = agent.replace(rng=rng)

        total_bs = batch["actions"].shape[0]
        assert total_bs % utd_ratio == 0, (
            f"Batch size ({total_bs}) must be a multiple of utd_ratio ({utd_ratio})"
        )
        minibatch_size = total_bs // utd_ratio

        def reshape_minibatch(x):
            return x.reshape((utd_ratio, minibatch_size) + x.shape[1:])

        minibatches = jax.tree_util.tree_map(reshape_minibatch, batch)

        def create_minibatch_sharding(x):
            # Create sharding spec: (None, DATA_AXIS, ...)
            # None for utd_ratio dimension, DATA_AXIS for minibatch dimension
            ndim = len(x.shape)
            spec_tuple = (None,) + (_sharding.DATA_AXIS,) + (None,) * (ndim - 2)
            new_sharding = jax.sharding.NamedSharding(
                self.data_sharding.mesh,
                jax.sharding.PartitionSpec(*spec_tuple)
            )
            return jax.device_put(x, new_sharding)

        minibatches = jax.tree_util.tree_map(create_minibatch_sharding, minibatches)

        def critic_update_step(carry, mb):
            (agent,) = carry
            agent, info = agent.update_critic(mb)
            return (agent,), info

        (new_agent,), critic_infos = jax.lax.scan(critic_update_step, (new_agent,), minibatches)

        # Use last minibatch for actor updates
        last_minibatch = jax.tree_util.tree_map(lambda x: x[-1] if x is not None and hasattr(x, "shape") else x, minibatches)

        # When actor_success_only, use the dedicated success-episode batch for
        # the Pi05 actor update; otherwise use the last critic minibatch.
        if self.actor_success_only:
            actor_batch = actor_batch.copy()
            rng, key = jax.random.split(new_agent.rng)
            actor_batch["image"] = self.data_augmentation_fn(key, actor_batch["image"])
            new_agent = new_agent.replace(rng=rng)
            actor_batch = prepare_critic_batch(actor_batch, self.actor.model_config.action_dim, self.action_dim, self.state_dim, self.action_horizon, self.replan_steps, self.critic_camera_keys)
            new_agent, actor_info = new_agent.update_actor(actor_batch)
        else:
            new_agent, actor_info = new_agent.update_actor(last_minibatch)

        actor_info = dict(actor_info)

        if self.n_edit_samples > 0:
            new_agent, r_actor_info = new_agent.update_edit_actor(last_minibatch)
            new_agent, temp_info = new_agent.update_temperature(r_actor_info["entropy"])
            actor_info = {**actor_info, **r_actor_info, **temp_info}

        critic_info = jax.tree_util.tree_map(lambda x: x[-1], critic_infos)
        return new_agent, {**actor_info, **critic_info}
