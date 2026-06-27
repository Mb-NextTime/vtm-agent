from enum import IntEnum

import numpy as np
from gymnasium import Env, spaces

from vtm_agent.engine import (
    Damage,
    DamageType,
    BarType,
    Hunter,
    Vampire,
    BloodRageError,
)


class Action(IntEnum):
    ATTACK = 0
    EVADE = 1
    WILLPOWER_ATTACK = 2
    WILLPOWER_EVADE = 3


class VTMCombatEnv(Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str | None = None,
        hunter_hp: int = 10,
        hunter_will: int = 10,
        hunter_attack: int = 6,
        hunter_evasion: int = 5,
        hunter_attack_mod: int = 0,
        vampire_hp: int = 10,
        vampire_will: int = 10,
        vampire_attack: int = 3,
        vampire_evasion: int = 3,
        vampire_attack_mod: int = 0,
        vampire_hunger: int = 1,
        max_rounds: int = 50,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        self._hunter_params = (hunter_hp, hunter_will, hunter_attack, hunter_attack_mod, hunter_evasion)
        self._vampire_params = (vampire_hp, vampire_will, vampire_attack, vampire_attack_mod, vampire_evasion, vampire_hunger)
        self.max_rounds = max_rounds

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(Action))

        self.hunter: Hunter | None = None
        self.vampire: Vampire | None = None
        self.round: int = 0

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        assert self.hunter is not None and self.vampire is not None
        h, v = self.hunter, self.vampire
        return np.array([
            (h.superficial_damage + h.aggravated_damage) / h.hp_cap,
            (h.will_superficial_damage + h.will_aggravated_damage) / h.will_cap,
            float(h.is_impaired),
            (v.superficial_damage + v.aggravated_damage) / v.hp_cap,
            (v.will_superficial_damage + v.will_aggravated_damage) / v.will_cap,
            float(v.is_impaired),
            v.hunger / 5.0,
            self.round / self.max_rounds,
        ], dtype=np.float32)

    def _action_mask(self) -> np.ndarray:
        assert self.hunter is not None
        mask = np.ones(len(Action), dtype=np.int8)
        will_filled = (
            self.hunter.will_superficial_damage + self.hunter.will_aggravated_damage
            >= self.hunter.will_cap
        )
        if will_filled:
            mask[Action.WILLPOWER_ATTACK] = 0
            mask[Action.WILLPOWER_EVADE] = 0
        return mask

    # ------------------------------------------------------------------
    # Combat resolution
    # ------------------------------------------------------------------

    def _resolve_attack(
        self,
        attacker: Hunter | Vampire,
        defender: Hunter | Vampire,
        willpower: bool,
    ) -> float:
        atk = attacker.roll(attacker.attack_pool)
        if willpower:
            sorted_idx = np.argsort(atk.common_pool)[:2].tolist()
            attacker.will_reroll(atk, sorted_idx)

        dfn = defender.roll(defender.evasion_pool)

        margin = atk.successes - dfn.successes
        if margin <= 0:
            return 0.0

        dmg_type = DamageType.AGGRAVATED if margin >= 5 else DamageType.SUPERFICIAL
        defender.apply_damage(Damage(margin, dmg_type, BarType.HP))
        sign = -1.0 if isinstance(attacker, Vampire) else 1.0
        return sign * float(margin) * 0.1

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)

        hp = self._hunter_params
        self.hunter = Hunter(hp=hp[0], will=hp[1], attack_pool=hp[2], attack_modifier=hp[3], evasion_pool=hp[4])
        vp = self._vampire_params
        self.vampire = Vampire(
            hp=vp[0], will=vp[1], attack_pool=vp[2], attack_modifier=vp[3], evasion_pool=vp[4], hunger=vp[5],
        )
        self.round = 0
        return self._observe(), {"action_mask": self._action_mask()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.hunter is not None and self.vampire is not None
        h, v = self.hunter, self.vampire
        action = Action(action)
        reward = 0.0

        try:
            if action in (Action.ATTACK, Action.WILLPOWER_ATTACK):
                will = action == Action.WILLPOWER_ATTACK
                reward += self._resolve_attack(h, v, will)
                if v.is_defeated:
                    return self._observe(), 1.0, True, False, {"action_mask": self._action_mask()}
                reward += self._resolve_attack(v, h, False)
            else:
                will = action == Action.WILLPOWER_EVADE
                reward += self._resolve_attack(v, h, False)
                if h.is_defeated:
                    return self._observe(), -1.0, True, False, {"action_mask": self._action_mask()}
                reward += self._resolve_attack(h, v, will)
        except BloodRageError:
            return self._observe(), -1.0, True, False, {"action_mask": self._action_mask()}

        if v.is_defeated:
            return self._observe(), 1.0, True, False, {"action_mask": self._action_mask()}
        if h.is_defeated:
            return self._observe(), -1.0, True, False, {"action_mask": self._action_mask()}

        self.round += 1
        truncated = self.round >= self.max_rounds

        if self.render_mode == "human":
            self._render()

        return self._observe(), reward, False, truncated, {"action_mask": self._action_mask()}

    def _render(self) -> None:
        assert self.hunter is not None and self.vampire is not None
        h, v = self.hunter, self.vampire
        print(f"--- Round {self.round + 1} ---")
        print(
            f"  Hunter  HP: {h.superficial_damage + h.aggravated_damage}/{h.hp_cap} "
            f"(S:{h.superficial_damage} A:{h.aggravated_damage})  "
            f"Will: {h.will_superficial_damage + h.will_aggravated_damage}/{h.will_cap}"
        )
        print(
            f"  Vampire HP: {v.superficial_damage + v.aggravated_damage}/{v.hp_cap} "
            f"(S:{v.superficial_damage} A:{v.aggravated_damage})  "
            f"Will: {v.will_superficial_damage + v.will_aggravated_damage}/{v.will_cap}  "
            f"Hunger: {v.hunger}"
        )
