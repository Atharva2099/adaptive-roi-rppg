"""Read-only SB3-Contrib recurrent-policy adapter."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import numpy as np

from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.model_replay import CheckpointIdentity

def _fail(message: str) -> None: raise ContractValidationError(message)

class SB3RecurrentPolicy:
    def __init__(self, model: Any, identity: CheckpointIdentity): self._model, self._identity = model, identity
    @property
    def identity(self) -> CheckpointIdentity: return self._identity
    def initial_state(self): return None
    def predict(self, observation: np.ndarray, recurrent_state, *, episode_start: bool):
        if observation.shape != (1,101) or observation.dtype != np.float32: _fail("SB3 adapter requires float32[1,101]")
        action, state = self._model.predict(observation, state=recurrent_state, episode_start=np.asarray([episode_start]), deterministic=True)
        return int(np.asarray(action).reshape(-1)[0]), state

def load_frozen_recurrent_policy(spec: dict[str, Any]) -> SB3RecurrentPolicy:
    path=Path(spec["locator"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != spec["byte_size"]: _fail("checkpoint file identity is invalid")
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]: _fail("checkpoint SHA-256 mismatch")
    try: from sb3_contrib import RecurrentPPO
    except ImportError as exc: raise ContractValidationError("model-eval dependencies are not installed") from exc
    model=RecurrentPPO.load(str(path), device="cpu")
    if tuple(model.observation_space.shape) != (101,) or getattr(model.observation_space,"dtype",None) != np.dtype(np.float32) or getattr(model.action_space,"n",None) != 12: _fail("checkpoint space does not match current controller")
    policy=model.policy
    if type(policy).__name__ != "RecurrentActorCriticPolicy" or getattr(policy,"lstm_actor",None) is None or policy.lstm_actor.hidden_size != 128 or policy.lstm_actor.num_layers != 1: _fail("checkpoint recurrent architecture does not match manifest")
    identity=CheckpointIdentity(spec["method_id"],spec["family"],spec["seed"],spec["sha256"])
    return SB3RecurrentPolicy(model, identity)
