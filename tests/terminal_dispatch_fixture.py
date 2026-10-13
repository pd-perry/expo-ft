"""CPU controls for the terminal boundary in the two training drivers.

This is a source-fragment check, not a training, policy or hardware run. External
I/O, inference, device transfer and reset concurrency are explicit test doubles.
The upstream episode logger, transition sink and timer remain unchanged.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

DRIVERS = ("train_pi_robo.py", "train_pi_robo_async.py")
SUPPORT = (
    "expo_ft/utils/log_utils.py",
    "expo_ft/utils/loop_utils.py",
    "expo_ft/utils/timer.py",
)
CASES = (
    "ongoing_queued",
    "ongoing_empty",
    "ongoing_human_takeover",
    "ongoing_human_control",
    "terminal_success_queued",
    "terminal_failure_empty",
    "terminal_timeout_human",
    "terminal_then_continue",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [plain(v) for v in value]
    return value


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract_loop(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loops = [
        n
        for n in main.body
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "i"
    ]
    assert len(loops) == 1, "Expected exactly one outer control loop"
    loop = loops[0]
    return compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), {
        "path": path.name,
        "first_line": loop.lineno,
        "last_line": loop.end_lineno,
        "ast_sha256": digest(ast.dump(loop, include_attributes=False).encode()),
    }


def sink_class(path, emitted_logs):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EpisodeSink")

    def unexpected_save(*args, **kwargs):
        raise AssertionError("Disk checkpointing is outside this control")

    namespace = {
        "sys": SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: False)),
        "logging": logging,
        "wandb": SimpleNamespace(log=lambda metrics, step: emitted_logs.append((step, metrics))),
        "save_replay_buffer_transition": unexpected_save,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["EpisodeSink"]


class EnvironmentDouble:
    def __init__(self, frames):
        self.frames = frames
        self.frame = -1
        self.episode = 0
        self.dispatches = []
        self.resets = []

    def get_observation(self):
        self.frame += 1
        return {"frame": self.frame, "episode": self.episode}

    def get_info_for_step(self):
        f = self.frames[self.frame]
        return f["done"], f["success"], float(f["success"]), float(not f["done"])

    def step(self, action):
        f = self.frames[self.frame]
        actual = [90.0, 91.0] if f["human"] else action
        self.dispatches.append(
            {"frame": self.frame, "done": f["done"], "requested": action, "actual": actual}
        )
        return np.asarray(actual), "human" if f["human"] else "policy"

    def pop_network_ms(self):
        return 0.0

    def reset(self):
        self.resets.append(self.frame)
        self.episode += 1


class SamplerDouble:
    def __init__(self, env):
        self.env = env
        self.samples = []
        self.launches = []
        self.takeovers = []
        self.ends = []

    def observe(self, observation):
        pass

    def launch(self, agent, observation, plan_len, action_type):
        self.launches.append(
            {"frame": self.env.frame, "done": self.env.frames[self.env.frame]["done"]}
        )

    def sample(self, agent, observation, metrics):
        self.samples.append(
            {"frame": self.env.frame, "done": self.env.frames[self.env.frame]["done"]}
        )
        index = self.env.frame
        return np.asarray([[10.0 + index, 20.0 + index], [30.0 + index, 40.0 + index]]), agent, {}

    def on_human_takeover(self):
        self.takeovers.append(self.env.frame)

    def on_episode_end(self):
        self.ends.append(self.env.frame)


class ImmediateResetDouble:
    def submit(self, function):
        result = function()
        return SimpleNamespace(result=lambda: result)


class BatchDouble:
    def __init__(self):
        self.inserted = []
        self.ends = []
        self.starts = 0

    def insert_transition(self, transition):
        self.inserted.append(copy.deepcopy(transition))

    def on_episode_done(self, success):
        self.ends.append(success)

    def on_episode_start(self):
        self.starts += 1


def run_case(source, driver, name, native_log, native_timer):
    is_terminal = name.startswith("terminal_")
    initial_human = name in {"ongoing_human_control", "terminal_timeout_human"}
    empty = initial_human or name in {"ongoing_empty", "terminal_failure_empty"}
    success = name in {"terminal_success_queued", "terminal_then_continue"}
    frames = [
        {"done": is_terminal, "success": success, "human": initial_human or "takeover" in name}
    ]
    if name == "terminal_then_continue":
        frames.append({"done": False, "success": False, "human": False})
    env = EnvironmentDouble(frames)
    sampler = SamplerDouble(env)
    batch = BatchDouble()
    emitted_logs = []
    sink_type = sink_class(source / SUPPORT[1], emitted_logs)
    sink = sink_type(
        batch,
        threading.Lock(),
        None,
        None,
        None,
        save_buffer=False,
        checkpoint_model=False,
        checkpoint_interval=0,
        start_step=0,
    )
    previous = {
        "observations": {"frame": -1, "episode": 0},
        "actions": np.asarray([-9.0, -8.0]),
        "is_hil": initial_human,
    }
    sink.record_transition(-1, previous)
    episode = native_log.EpisodeState(ep_len=0)
    episode.record_step(
        previous["observations"], 0, "human" if initial_human else "policy", previous["actions"], 0
    )
    stats = native_log.TrainingStats()
    initial_plan = [] if empty else [np.asarray([2.0, 3.0]), np.asarray([4.0, 5.0])]

    def unexpected_update(*args, **kwargs):
        raise AssertionError("Model updates are outside this control")

    flags = SimpleNamespace(
        max_steps=len(frames) - 1,
        replan_steps=2,
        tqdm=False,
        batch_size=1024,
        update_type="episode",
        num_updates=0,
        step_interval=1,
        num_batch=1,
    )
    scope = {
        "FLAGS": flags,
        "np": np,
        "time": time,
        "logging": logging,
        "tqdm": SimpleNamespace(tqdm=lambda sequence, **kwargs: sequence),
        "jax": SimpleNamespace(device_get=lambda value: value),
        "agent": "fixture-agent",
        "_actor_agent": "fixture-agent",
        "_publish_lock": threading.Lock(),
        "_published": [None],
        "_env_step": [0],
        "_episode_done": threading.Event(),
        "_can_update": threading.Event(),
        "replay_buffer": SimpleNamespace(_size=0),
        "env": env,
        "sampler": sampler,
        "sink": sink,
        "batch_processor": batch,
        "buffer_lock": threading.Lock(),
        "episode_log": episode,
        "training_log": stats,
        "timer": native_timer.Timer(),
        "reset_executor": ImmediateResetDouble(),
        "example_action": np.zeros((1, 2)),
        "start_step": 0,
        "dt": 0.0,
        "last_control_start": None,
        "done": False,
        "action_plan": deque(initial_plan),
        "action_type": "human" if initial_human else "policy",
        "first_step_of_episode": False,
        "transitions_since_update": 0,
        "episodes_since_update": 0,
        "run_agent_updates": unexpected_update,
        "updates_due": unexpected_update,
    }
    code, binding = extract_loop(source / driver)
    error = None
    try:
        exec(code, scope)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    metrics = [
        {"step": step, **{k: v for k, v in item.items() if not k.startswith("timing/")}}
        for step, item in emitted_logs + sink._logs
    ]
    completed = [item for item in metrics if "training/return" in item]
    transitions = batch.inserted + [item for _, item in sink._transitions]
    if is_terminal:
        expected_actions = [[11.0, 21.0]] if name == "terminal_then_continue" else []
    elif initial_human or "takeover" in name:
        expected_actions = [[90.0, 91.0]]
    else:
        expected_actions = [[10.0, 20.0]] if empty else [[2.0, 3.0]]
    terminal_frames = [i for i, frame in enumerate(frames) if frame["done"]]
    continuing_frames = [i for i, frame in enumerate(frames) if not frame["done"]]
    checks = {
        "execution_completed": error is None,
        "dispatch_only_on_continuing_frames": [item["frame"] for item in env.dispatches]
        == continuing_frames,
        "no_terminal_dispatch": not any(item["done"] for item in env.dispatches),
        "no_terminal_sampling": not any(item["done"] for item in sampler.samples),
        "no_terminal_launch": not any(item["done"] for item in sampler.launches),
        "exact_continuing_actions": [item["actual"] for item in env.dispatches] == expected_actions,
        "previous_transition_labels": (
            previous.get("rewards"),
            previous.get("masks"),
            previous.get("dones"),
        )
        == (float(success), float(not is_terminal), is_terminal),
        "one_reset_per_terminal": env.resets == terminal_frames,
        "one_episode_end_per_terminal": sampler.ends == terminal_frames
        and len(batch.ends) == len(terminal_frames),
        "terminal_return_preserved": [item["training/return"] for item in completed]
        == ([float(success)] if is_terminal else []),
        "terminal_length_has_no_extra_action": [item["training/length"] for item in completed]
        == ([1.0] if is_terminal else []),
        "no_terminal_transition": all(
            item["observations"]["frame"] not in terminal_frames for item in transitions
        ),
        "continuing_transitions_retained": sorted(
            item["observations"]["frame"] for item in transitions
        )
        == [-1, *continuing_frames],
        "human_takeover_preserved": sampler.takeovers
        == [i for i, frame in enumerate(frames) if frame["human"] and not frame["done"]],
        "human_transition_labels": all(
            item["is_hil"]
            == (
                initial_human
                if item["observations"]["frame"] == -1
                else frames[item["observations"]["frame"]]["human"]
            )
            for item in transitions
        ),
    }
    observables = plain(
        {
            "dispatches": env.dispatches,
            "samples": sampler.samples,
            "launches": sampler.launches,
            "resets": env.resets,
            "takeovers": sampler.takeovers,
            "ends": sampler.ends,
            "transitions": transitions,
            "episode_state": asdict(episode),
            "metrics": metrics,
            "remaining_plan": list(scope["action_plan"]),
            "action_type": scope["action_type"],
        }
    )
    return {
        "id": driver + ":" + name,
        "binding": binding,
        "inputs": frames,
        "scope": "constructed control-flow fixture; no learned inference or hardware",
        "checks": checks,
        "passed": all(checks.values()),
        "error": error,
        "observables": observables,
        "observables_sha256": digest(json.dumps(observables, sort_keys=True).encode()),
    }
