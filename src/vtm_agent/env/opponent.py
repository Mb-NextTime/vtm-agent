from typing import Protocol

import numpy as np

from vtm_agent.engine import Hunter, Vampire
from vtm_agent.env.action import Phase, Stance, WillpowerAction


class Opponent(Protocol):
    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        agent: Hunter | Vampire,
        opponent: Hunter | Vampire,
        phase: Phase,
    ) -> int: ...


class ScriptedOpponent:
    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        agent: Hunter | Vampire,
        opponent: Hunter | Vampire,
        phase: Phase,
    ) -> int:
        if phase == Phase.STANCE:
            return int(Stance.ATTACK)
        return int(WillpowerAction.SKIP)


class RandomOpponent:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        agent: Hunter | Vampire,
        opponent: Hunter | Vampire,
        phase: Phase,
    ) -> int:
        valid = [i for i, v in enumerate(action_mask) if v]
        if not valid:
            return 0
        return int(self._rng.choice(valid))
