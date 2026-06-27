from typing import NamedTuple

import numpy as np
from gymnasium import Env, spaces

from vtm_agent.engine import (
    BarType,
    BloodRageError,
    Damage,
    DamageType,
    Hunter,
    Roll,
    Vampire,
)
from vtm_agent.env.action import Phase, Stance, WillpowerAction
from vtm_agent.env.opponent import Opponent, ScriptedOpponent

MAX_POOL = 30


class VTMCombatEnv(Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str | None = None,
        opponent: Opponent | None = None,
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
        self._opponent = opponent if opponent is not None else ScriptedOpponent()

        self._hunter_params = _HunterParams(hunter_hp, hunter_will, hunter_attack, hunter_attack_mod, hunter_evasion)
        self._vampire_params = _VampireParams(
            vampire_hp, vampire_will, vampire_attack, vampire_attack_mod, vampire_evasion, vampire_hunger
        )
        self.max_rounds = max_rounds

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.hunter: Hunter | None = None
        self.vampire: Vampire | None = None
        self.round: int = 0
        self._phase: Phase = Phase.STANCE
        self._agent_stance: Stance | None = None
        self._agent_roll: Roll | None = None
        self._opponent_successes: int | None = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observe(self, pre_will_successes: int | None = None) -> np.ndarray:
        assert self.hunter is not None and self.vampire is not None
        h, v = self.hunter, self.vampire
        return np.array(
            [
                (h.superficial_damage + h.aggravated_damage) / h.hp_cap,
                (h.will_superficial_damage + h.will_aggravated_damage) / h.will_cap,
                float(h.is_impaired),
                (v.superficial_damage + v.aggravated_damage) / v.hp_cap,
                (v.will_superficial_damage + v.will_aggravated_damage) / v.will_cap,
                float(v.is_impaired),
                v.hunger / 5.0,
                self.round / self.max_rounds,
                float(self._phase),
                (pre_will_successes or 0) / MAX_POOL,
            ],
            dtype=np.float32,
        )

    def _action_mask(self) -> np.ndarray:
        mask = np.array([1, 1, 0, 0], dtype=np.int8)
        if self._phase == Phase.WILLPOWER:
            person = self.hunter
            assert person is not None
            if person.will_superficial_damage + person.will_aggravated_damage >= person.will_cap:
                mask[1] = 0
        return mask

    # ------------------------------------------------------------------
    # Combat resolution
    # ------------------------------------------------------------------

    def _resolve_opponent(self, h: Hunter, v: Vampire) -> int:
        obs = self._observe()

        stance_mask = np.array([1, 1, 0, 0], dtype=np.int8)
        stance = Stance(self._opponent.act(obs, stance_mask, h, v, Phase.STANCE))
        pool = v.attack_pool if stance == Stance.ATTACK else v.evasion_pool
        roll = v.roll(pool)

        wp_mask = np.array([1, 1, 0, 0], dtype=np.int8)
        if v.will_superficial_damage + v.will_aggravated_damage >= v.will_cap:
            wp_mask[1] = 0
        wp = WillpowerAction(self._opponent.act(obs, wp_mask, h, v, Phase.WILLPOWER))
        if wp == WillpowerAction.USE:
            v.will_reroll(roll)

        return roll.successes

    def _resolve_damage(self, margin: int, target: Hunter | Vampire) -> float:
        dmg_type = DamageType.AGGRAVATED if margin >= 5 else DamageType.SUPERFICIAL
        target.apply_damage(Damage(margin, dmg_type, BarType.HP))
        return float(margin) * 0.1

    def _reset_round_state(self) -> None:
        self._phase = Phase.STANCE
        self._agent_stance = None
        self._agent_roll = None
        self._opponent_successes = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)

        hp = self._hunter_params
        self.hunter = Hunter(
            hp=hp.hp, will=hp.will, attack_pool=hp.attack_pool, attack_modifier=hp.attack_mod, evasion_pool=hp.evasion
        )
        vp = self._vampire_params
        self.vampire = Vampire(
            hp=vp.hp,
            will=vp.will,
            attack_pool=vp.attack_pool,
            attack_modifier=vp.attack_mod,
            evasion_pool=vp.evasion,
            hunger=vp.hunger,
        )

        self.round = 0
        self._reset_round_state()
        return self._observe(), {"action_mask": self._action_mask()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.hunter is not None and self.vampire is not None
        h, v = self.hunter, self.vampire

        action = int(action)
        mask = self._action_mask()
        if not mask[action]:
            action = int(mask.argmax())

        if self._phase == Phase.STANCE:
            return self._step_stance(h, v, action)
        return self._step_willpower(h, v, action)

    def _step_stance(
        self, h: Hunter, v: Vampire, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self._agent_stance = Stance(action)
        pool = h.attack_pool if self._agent_stance == Stance.ATTACK else h.evasion_pool
        self._agent_roll = h.roll(pool)

        try:
            self._opponent_successes = self._resolve_opponent(h, v)
        except BloodRageError:
            self._reset_round_state()
            return self._observe(), -1.0, True, False, {"action_mask": self._action_mask()}

        self._phase = Phase.WILLPOWER
        return (
            self._observe(pre_will_successes=self._agent_roll.successes),
            0.0,
            False,
            False,
            {"action_mask": self._action_mask()},
        )

    def _step_willpower(
        self, h: Hunter, v: Vampire, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self._agent_roll is not None and self._opponent_successes is not None

        wp = WillpowerAction(action)
        if wp == WillpowerAction.USE:
            h.will_reroll(self._agent_roll)

        agent_succ = self._agent_roll.successes
        opp_succ = self._opponent_successes

        reward = 0.0
        if agent_succ > opp_succ:
            reward += self._resolve_damage(agent_succ - opp_succ, v)
        elif opp_succ > agent_succ:
            reward += self._resolve_damage(opp_succ - agent_succ, h)

        if v.is_defeated:
            self._reset_round_state()
            return self._observe(), 1.0, True, False, {"action_mask": self._action_mask()}
        if h.is_defeated:
            self._reset_round_state()
            return self._observe(), -1.0, True, False, {"action_mask": self._action_mask()}

        self.round += 1
        truncated = self.round >= self.max_rounds
        self._reset_round_state()

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


class _HunterParams(NamedTuple):
    hp: int
    will: int
    attack_pool: int
    attack_mod: int
    evasion: int


class _VampireParams(NamedTuple):
    hp: int
    will: int
    attack_pool: int
    attack_mod: int
    evasion: int
    hunger: int
