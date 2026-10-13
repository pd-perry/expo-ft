#! /usr/bin/env python
"""Online RL fine-tuning of a pi0.5 policy on the real robot, with env stepping and
gradient updates on separate threads (device 0 samples, the rest update)."""
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tqdm
from absl import app, flags

from ml_collections import config_flags

import jax
import etils.epath as epath

import wandb
from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.utils.log_utils import EpisodeState, TrainingStats
from expo_ft.utils.loop_utils import AsyncChunkSampler, EpisodeSink
from expo_ft.utils.timer import Timer
from expo_ft.utils.train_utils import (
    get_batch_info, init_logging, init_wandb, set_compilation_cache_dir,
)
from expo_ft.agents.vla.pi05 import build_pi05

import openpi.training.sharding as openpi_sharding

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo-ft", "wandb project name.")
flags.DEFINE_string("run_name", None, "Optional wandb run name.")
flags.DEFINE_float("offline_ratio", 0.0, "Offline batch fraction; 0 inserts dataset into online replay buffer.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_float("ep_timeout_secs", 120.0, "Pause update thread if no episode finishes within this many seconds. 0 to disable.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("max_steps", 100_000, "Number of training steps.")
flags.DEFINE_integer("num_data", 0, "Max number of offline demo episodes to load (0 = all).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint during training.")
flags.DEFINE_integer("checkpoint_interval", 0, "Save agent checkpoint every N steps. When 0 and checkpoint_model=True, no interval saving (save at end only).")
flags.DEFINE_boolean("checkpoint_buffer", False, "Save agent replay buffer on evaluation.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("keep_period", None, "Keep checkpoints every N steps.")
flags.DEFINE_boolean("overwrite", False, "Overwrite existing checkpoint directory.")
flags.DEFINE_boolean("resume", False, "Resume training from checkpoint.")
flags.DEFINE_string("output_dir", "./logs", "Directory for logs and checkpoints.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")

flags.DEFINE_string("client_host", "0.0.0.0", "Bind host to listen on; the rollout client dials in.")
flags.DEFINE_integer("client_port", 8102, "Bind port to listen on.")

flags.DEFINE_integer("replan_steps", 8, "Number of replan steps for evaluation.")
flags.DEFINE_integer(
    "delay", 0,
    "Simulated main-actor inference latency (env-steps) for RealTimeEXPOFTLearner. "
    "0 <= delay <= replan_steps. >0 enables the delayed + prefix-inpainted "
    "rollout and the matching delayed critic backup. Ignored by other learners.",
)
flags.DEFINE_float("sim_latency", 0.0, "Simulated extra inference latency in ms added to each sample_actions call; 0 disables.")

flags.DEFINE_string("dataset_path", "", "Path to the dataset.")
config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/pick.py",
    "File path to the task configuration.",
    lock_config=False,
)


def main(_):
    init_logging()
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0
    set_compilation_cache_dir(f"async-{FLAGS.config.model_cls}")

    num_gpus = jax.device_count()
    if num_gpus < 2:
        raise ValueError(
            f"At least 2 GPUs required (1 for sampling, rest for updates), got {num_gpus}"
        )
    sample_device = jax.devices()[0]
    update_devices = jax.devices()[1:]
    num_update = len(update_devices)
    if num_update % FLAGS.fsdp_devices != 0:
        raise ValueError(
            f"Number of update devices ({num_update}) must be divisible by "
            f"fsdp_devices ({FLAGS.fsdp_devices})"
        )
    if FLAGS.batch_size % num_update != 0:
        raise ValueError(
            f"Batch size {FLAGS.batch_size} must be divisible by "
            f"the number of update devices {num_update}"
        )
    mesh = jax.sharding.Mesh(
        np.array(update_devices).reshape(num_update // FLAGS.fsdp_devices, FLAGS.fsdp_devices),
        (openpi_sharding.BATCH_AXIS, openpi_sharding.FSDP_AXIS),
    )
    logging.info("Device layout: sampling on %s, updates on %s", sample_device, update_devices)

    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    os.makedirs(log_dir, exist_ok=True)
    train_video_dir = os.path.join(log_dir, "train_videos")
    os.makedirs(train_video_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_dir_path = epath.Path(checkpoint_dir)
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=FLAGS.keep_period,
        overwrite=FLAGS.overwrite,
        resume=FLAGS.resume,
    )

    init_wandb(checkpoint_dir_path, resuming, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=resuming)

    if FLAGS.config_task.env_type == "droid":
        dataset = process_droid_dataset(
            FLAGS.dataset_path,
            FLAGS.config_task,
            num_data=FLAGS.num_data,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {FLAGS.config_task.env_type}")
    if not dataset:
        raise ValueError(f"No demos loaded from {FLAGS.dataset_path}.")
    example_action = dataset[0]['actions'][np.newaxis]

    model_cls = FLAGS.config.model_cls
    # BCLearner uses human-intervention chunks for the actor batch only (no critic).
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "RealTimeEXPOFTLearner":
        from expo_ft.agents.alg.realtime_expo_ft import load_agent, restore_checkpoint, save_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    # Same string run_client returns as task_description, without needing the env yet.
    task_description = FLAGS.config_task.language_instruction

    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resuming, task_description,
    )

    critic_camera_keys = tuple(getattr(FLAGS.config_task, "critic_camera_keys", CRITIC_CAMERA_KEYS))
    rb_args = dict(
        config=FLAGS.config,
        example_action=example_action,
        capacity=FLAGS.max_steps,
        task_description=task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
        delay=FLAGS.delay,
        critic_camera_keys=critic_camera_keys,
    )
    replay_buffer = create_replay_buffer(**rb_args)
    offline_replay_buffer = create_replay_buffer(**rb_args)

    actor_success_only = getattr(FLAGS.config, "actor_success_only", False)
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        data_sharding=data_sharding,
        batch_size=FLAGS.batch_size,
        utd_ratio=FLAGS.utd_ratio,
        offline_ratio=FLAGS.offline_ratio,
        actor_success_only=actor_success_only,
        use_dagger_hil_sampling=use_dagger_hil_sampling,
        dataset=dataset,
    )

    critic_example = {
        _critic_key_to_storage(k): offline_replay_buffer.dataset_dict[_critic_key_to_storage(k)][0][np.newaxis]
        for k in critic_camera_keys
    }
    critic_example["state"] = offline_replay_buffer.dataset_dict['state'][0][np.newaxis]
    critic_example["actions"] = offline_replay_buffer.dataset_dict['actions'][0][np.newaxis]
    agent_example_observation, agent_example_state, agent_example_action = offline_replay_buffer.convert_to_critic_format(
        critic_example)
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]

    if model_cls == "RealTimeEXPOFTLearner":
        agent_kwargs["delay"] = FLAGS.delay
    agent_kwargs["critic_camera_keys"] = critic_camera_keys

    agent = load_agent(
        seed=FLAGS.seed,
        example_observation=agent_example_observation.squeeze(),
        example_action=agent_example_action.squeeze(),
        example_state=agent_example_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=vla_metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=resuming,
        replan_steps=FLAGS.replan_steps,
        default_prompt=task_description,
        edit_action_xyzg=FLAGS.config_task.edit_action_xyzg,
    )

    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if latest_step is not None:
            start_step = latest_step
            logging.info("Resuming from step %d", start_step)
        batch_processor.restore(checkpoint_dir_path, up_to_step=latest_step)

    # The env is created only now: EnvClientWrapper blocks until the rollout client
    # dials in, and the agent build above is the slow part.
    train_env_creation_request = {
        "example_action": example_action,
        "env_usage": "train",
        "video_dir": train_video_dir,
    }
    logging.info("Creating environment...")
    env = EnvClientWrapper(
        env_creation_request=train_env_creation_request,
        host=FLAGS.client_host,
        port=FLAGS.client_port
    )
    env.reset()
    logging.info(f"Created training environment {env.env_id}")

    episode_log = EpisodeState()
    training_log = TrainingStats(
        ep_count=replay_buffer.count_episodes_chronological() if resuming else 0,
    )
    if resuming:
        logging.info("Resuming: ep_count set to %d (episodes in replay buffer).", training_log.ep_count)

    # Guards the replay buffer: the update thread samples while the loop inserts.
    buffer_lock = threading.Lock()

    batch_processor.on_episode_start()
    timer = Timer()

    # --- Async update thread setup ---
    # Actor (main thread) samples on device[0] (the pi0.5 agent's infer_device), learner
    # (background thread) updates on device[1:]. Params published via atomic reference swap.
    _actor_agent = agent.cache_infer_params()
    _published = [None]
    _publish_lock = threading.Lock()
    _env_step = [start_step]
    _stop_event = threading.Event()
    _can_update = threading.Event()
    _episode_done = threading.Event()
    _update_count = [0]
    _ckpt_request = [None]  # main thread sets step number; update thread saves and clears
    _ckpt_done = threading.Event()

    def _save_from_update_thread(_manager, _agent, step):
        """Checkpoints hold the learner's train states, which only the update thread has."""
        if not _can_update.is_set():
            save_checkpoint(checkpoint_manager, agent, step)  # no update has run yet
            return
        _ckpt_done.clear()
        _ckpt_request[0] = step
        if not _ckpt_done.wait(timeout=600):
            raise RuntimeError("update thread did not save the checkpoint in time")

    # Runs main-actor inference in the background while the robot executes the last `delay` actions.
    sampler = AsyncChunkSampler(
        _actor_agent, delay=FLAGS.delay, replan_steps=FLAGS.replan_steps,
        sim_latency_ms=FLAGS.sim_latency, timer=timer,
    )
    # Buffer inserts, NFS buffer saves, wandb logs (both threads) and checkpoints queue here and flush at episode end.
    sink = EpisodeSink(
        batch_processor, buffer_lock, checkpoint_manager, checkpoint_dir_path, _save_from_update_thread,
        save_buffer=FLAGS.checkpoint_buffer, checkpoint_model=FLAGS.checkpoint_model,
        checkpoint_interval=FLAGS.checkpoint_interval, start_step=start_step,
    )
    # env.reset() runs in the background so the episode-end bookkeeping overlaps the robot homing.
    reset_executor = ThreadPoolExecutor(max_workers=1)

    def _update_worker():
        combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)
        learner_agent = agent
        last_episode_time = time.time()
        update_time = deque(maxlen=10)

        _can_update.wait()
        while not _stop_event.is_set():
            try:
                if FLAGS.ep_timeout_secs > 0 and time.time() - last_episode_time > FLAGS.ep_timeout_secs:
                    logging.info("No episode finished for %.1fs, pausing updates.", FLAGS.ep_timeout_secs)
                    sink.record_log(_env_step[0], {"training/update_paused": 1})
                    _episode_done.wait()
                    _episode_done.clear()
                    last_episode_time = time.time()
                    logging.info("Episode signal received, resuming updates.")
                    sink.record_log(_env_step[0], {"training/update_paused": 0})

                if _episode_done.is_set():
                    _episode_done.clear()
                    last_episode_time = time.time()

                with buffer_lock:
                    batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
                batch_info = get_batch_info(batch)

                t0 = time.time()
                learner_agent = learner_agent.replace(
                    rng=jax.device_put(learner_agent.rng, replicated_sharding)
                )
                learner_agent, update_info = learner_agent.update(
                    learner_agent, batch, FLAGS.utd_ratio, actor_batch
                )

                with _publish_lock:
                    _published[0] = learner_agent._infer_cache

                update_time.append(time.time() - t0)
                _update_count[0] += 1
                log_dict = {f"training/{k}": v for k, v in update_info.items()}
                log_dict["batch_info"] = batch_info
                log_dict["training/num_updates"] = _update_count[0]
                if _update_count[0] % 10 == 0 and len(update_time) == update_time.maxlen:
                    log_dict["training/update_time_avg_ms"] = float(np.mean(update_time)) * 1000.0
                sink.record_log(_env_step[0], log_dict)

                ckpt_step = _ckpt_request[0]
                if ckpt_step is not None:
                    _ckpt_request[0] = None
                    try:
                        save_checkpoint(checkpoint_manager, learner_agent, ckpt_step)
                        checkpoint_manager.wait_until_finished()
                    except Exception as e:
                        logging.error("Could not save model checkpoint: %s", e)
                    _ckpt_done.set()
            except Exception:
                logging.exception("Update thread crashed at update %d", _update_count[0])
                break
        # Handle checkpoint request after stop signal
        ckpt_step = _ckpt_request[0]
        if ckpt_step is not None:
            _ckpt_request[0] = None
            try:
                save_checkpoint(checkpoint_manager, learner_agent, ckpt_step)
                logging.info("Saved agent checkpoint at step %d (from update thread)", ckpt_step)
            except Exception as e:
                logging.error("Could not save checkpoint: %s", e)
            _ckpt_done.set()
        logging.info("Update thread exiting (updates=%d).", _update_count[0])

    _update_thread = threading.Thread(target=_update_worker, daemon=True)
    _update_thread.start()

    if resuming and replay_buffer._size >= FLAGS.batch_size:
        _can_update.set()
        logging.info("Resuming: replay buffer already warm, update thread starting immediately.")

    dt = 1.0 / FLAGS.config_task.control_hz
    done = False
    last_control_start = time.time()
    env.step(FLAGS.config_task.example_action.squeeze().tolist())
    action_plan = deque()
    action_type = "policy"
    first_step_of_episode = True

    for i in tqdm.tqdm(
        range(start_step, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        step_metrics = {}
        timer.reset()
        timer.tick("loop_time")
        timer.tick("total")
        _env_step[0] = i

        # Pick up the learner's newest params; an in-flight chunk keeps the ones it started with.
        with _publish_lock:
            new_cache = _published[0]
            _published[0] = None
        if new_cache is not None:
            _actor_agent = _actor_agent.replace(_infer_cache=new_cache)

        with timer.context("obs"):
            observation = env.get_observation()

        # Observation history for the replan_steps < delay fallback (no-op otherwise).
        sampler.observe(observation)
        with timer.context("info"):
            done, success, reward, mask = env.get_info_for_step()
        sink.label_last(reward, mask, done)  # outcome of the previous action

        if not done:
            # Start the next chunk's inference once exactly `delay` actions remain in the plan.
            sampler.launch(_actor_agent, observation, len(action_plan), action_type)

            with timer.context("plan"):
                # Skip model inference while human is controlling.
                if not action_plan and action_type != "human":
                    sample_start = time.time()
                    action_chunk, _actor_agent, sample_info = sampler.sample(_actor_agent, observation, step_metrics)
                    episode_log.sample_info_history.append(sample_info)
                    action_chunk = np.asarray(jax.device_get(action_chunk[:FLAGS.replan_steps]))
                    training_log.record_sample_time(time.time() - sample_start, step_metrics)
                    action_plan.extend(action_chunk)
                    # Re-check with the refilled plan (covers delay == replan_steps).
                    sampler.launch(_actor_agent, observation, len(action_plan), action_type)
                else:
                    episode_log.sample_info_history.append(
                        episode_log.sample_info_history[-1] if episode_log.sample_info_history else None
                    )

            # Pacing is anchored to the previous step's dispatch time, not its return, so latency does not accumulate.
            sleep_left = 0.0 if last_control_start is None else dt - (time.time() - last_control_start)
            step_metrics["timing/sleep_left_ms"] = sleep_left * 1000.0
            with timer.context("wait"):
                if sleep_left > 0:
                    time.sleep(sleep_left)

            has_action = bool(action_plan)
            action = action_plan.popleft() if has_action else np.zeros_like(example_action.squeeze())

            with timer.context("act"):
                last_control_start = time.time()
                real_action, action_type = env.step(action.tolist())

        # Wire time of this step's env RPCs, as measured by the client wrapper.
        step_metrics["timing/network_ms"] = env.pop_network_ms()
        timer.tock("total")
        timer.tick("post_step")
        if not done:
            episode_log.record_step(observation, len(action_plan), action_type, real_action, reward)
        else:
            # The terminal reward belongs to the preceding action, not a new step.
            episode_log.ep_return += reward

        # A human override discards the plan and any in-flight inference.
        if not done and action_type == "human":
            action_plan.clear()
            sampler.on_human_takeover()

        if not done and (has_action or action_type == "human"):
            transition = dict(
                observations=observation,
                actions=real_action,
                is_hil=(action_type == "human"),
            )
            # Queued only; the buffer insert happens at episode end.
            sink.record_transition(i, transition)

        log_timing = not done and not first_step_of_episode
        first_step_of_episode = done  # next iteration starts a new episode if this one ended
        if done:
            # Reset starts now; the bookkeeping below runs while the robot homes.
            pending_reset = reset_executor.submit(env.reset)
            sink.flush_transitions()
            with buffer_lock:
                batch_processor.on_episode_done(success)
            _episode_done.set()
            last_control_start = None
            sampler.on_episode_end()

            training_log.on_episode_done(episode_log, success, step_metrics)
            episode_log.reset()
            with buffer_lock:
                batch_processor.on_episode_start()

            # Drain queued inserts, NFS saves and logs; interval checkpoint if due.
            sink.flush_episode(i, _actor_agent)

            # Updates start once a batch fits; compiling is slow, so do not wait for more episodes.
            if not _can_update.is_set() and replay_buffer._size >= FLAGS.batch_size:
                _can_update.set()
                logging.info("Replay buffer ready (ep_count=%d), starting update thread.", training_log.ep_count)

            pending_reset.result()  # re-raises reset failures on the main thread
            env.pop_network_ms()  # discard the reset round-trip so it doesn't inflate the next step
            done = False
            action_type = "policy"
            action_plan.clear()

        timer.tock("post_step")
        timer.tock("loop_time")
        step_metrics.update({f"timing/{k}_ms": v for k, v in timer.get_times_ms().items()})

        if not log_timing:
            step_metrics = {k: v for k, v in step_metrics.items()
                            if not k.startswith("timing/")}
        training_log.maybe_add_success_rate(i, step_metrics)
        sink.record_log(i, step_metrics)

    sink.flush_all()

    if FLAGS.checkpoint_model:
        _ckpt_done.clear()
        _ckpt_request[0] = FLAGS.max_steps
    _stop_event.set()
    _can_update.set()
    _episode_done.set()
    _update_thread.join()

    if FLAGS.checkpoint_model:
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()

    sampler.close()
    reset_executor.shutdown(wait=True)


if __name__ == "__main__":
    app.run(main)
