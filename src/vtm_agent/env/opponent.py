from typing import Protocol

import numpy as np

from vtm_agent.engine import Hunter, Vampire
from vtm_agent.env.action import Action


class Opponent(Protocol):
    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        hunter: Hunter,
        vampire: Vampire,
    ) -> Action: ...


class ScriptedOpponent:
    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        hunter: Hunter,
        vampire: Vampire,
    ) -> Action:
        return Action.ATTACK


class RandomOpponent:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        hunter: Hunter,
        vampire: Vampire,
    ) -> Action:
        valid = [a for a in Action if action_mask[a]]
        if not valid:
            return Action.ATTACK
        return Action(self._rng.choice(valid))
