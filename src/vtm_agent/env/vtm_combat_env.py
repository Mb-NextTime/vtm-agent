import logging
from dataclasses import dataclass

import numpy as np
from gymnasium import Env, spaces

from vtm_agent.engine import (
    BarType,
    Damage,
    DamageType,
    Hunter,
    Person,
    Roll,
    Vampire,
)
from vtm_agent.env.action import Phase, Stance, WillpowerAction
from vtm_agent.env.opponent import Opponent, ScriptedOpponent

logger = logging.getLogger(__name__)

MAX_POOL = 30
MAX_STAT = 30
MAX_MOD = 15
OBS_DIM = 30
MAX_HUNGER = 5


@dataclass
class CharacterConfig:
    hp_range: tuple[int, int] = (8, 12)
    will_range: tuple[int, int] = (8, 12)
    attack_pool_range: tuple[int, int] = (3, 8)
    attack_mod_range: tuple[int, int] = (-2, 2)
    evasion_pool_range: tuple[int, int] = (2, 6)
    hunger_range: tuple[int, int] = (1, 5)
    is_vampire: bool = True


ObsType = np.ndarray


def _build_obs_block(p: Person) -> list[float]:
    hp_dmg = p.superficial_damage + p.aggravated_damage
    will_dmg = p.will_superficial_damage + p.will_aggravated_damage
    hp_cap = max(1, p.hp_cap)
    will_cap = max(1, p.will_cap)
    return [
        hp_dmg / hp_cap,
        will_dmg / will_cap,
        p.aggravated_damage / hp_cap,
        p.will_aggravated_damage / will_cap,
        float(p.is_impaired),
        float(p.is_impaired_will),
        float(p.is_vampire),
        p.hunger / MAX_HUNGER,
        p.base_attack_pool / MAX_POOL,
        p.base_evasion_pool / MAX_POOL,
        (p.attack_modifier + MAX_MOD) / (2 * MAX_MOD),
        p.hp_cap / MAX_STAT,
        p.will_cap / MAX_STAT,
    ]


class VTMCombatEnv(Env[ObsType, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str | None = None,
        opponent: Opponent | None = None,
        agent_config: CharacterConfig | None = None,
        opponent_config: CharacterConfig | None = None,
        max_rounds: int = 50,
        randomize_roles: bool = True,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._opponent = opponent if opponent is not None else ScriptedOpponent()
        self._agent_config = agent_config or CharacterConfig(is_vampire=False)
        self._opponent_config = opponent_config or CharacterConfig(is_vampire=True)
        self.max_rounds = max_rounds
        self.randomize_roles = randomize_roles

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self._agent_person: Hunter | Vampire | None = None
        self._opponent_person: Hunter | Vampire | None = None
        self.round: int = 0
        self._phase: Phase = Phase.STANCE
        self._agent_stance: Stance | None = None
        self._agent_roll: Roll | None = None
        self._opponent_successes: int | None = None

    # ------------------------------------------------------------------
    # Character factory
    # ------------------------------------------------------------------

    @staticmethod
    def _build_person(cfg: CharacterConfig, rng: np.random.Generator) -> Hunter | Vampire:
        hp = int(rng.integers(*cfg.hp_range))
        will = int(rng.integers(*cfg.will_range))
        attack = int(rng.integers(*cfg.attack_pool_range))
        attack_mod = int(rng.integers(*cfg.attack_mod_range))
        evasion = int(rng.integers(*cfg.evasion_pool_range))
        if cfg.is_vampire:
            hunger = int(rng.integers(*cfg.hunger_range))
            return Vampire(hp, will, attack, attack_mod, evasion, hunger)
        return Hunter(hp, will, attack, attack_mod, evasion)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observe(self, pre_will_roll: Roll | None = None) -> np.ndarray:
        assert self._agent_person is not None and self._opponent_person is not None
        agent = _build_obs_block(self._agent_person)
        opp = _build_obs_block(self._opponent_person)

        if pre_will_roll is not None:
            common_succ = pre_will_roll.common_successes / MAX_POOL
            blood_succ = pre_will_roll.blood_successes / MAX_POOL
        else:
            common_succ = 0.0
            blood_succ = 0.0

        context: list[float] = [
            float(self._phase),
            self.round / self.max_rounds,
            common_succ,
            blood_succ,
        ]
        return np.array(agent + opp + context, dtype=np.float32)

    def action_mask(self) -> np.ndarray:
        assert self._agent_person is not None
        if self._phase == Phase.STANCE:
            return np.array([1, 1, 0, 0], dtype=np.int8)
        mask = np.array([0, 0, 1, 1], dtype=np.int8)
        p = self._agent_person
        if p.will_superficial_damage + p.will_aggravated_damage >= p.will_cap:
            mask[3] = 0
        return mask

    # ------------------------------------------------------------------
    # Combat resolution
    # ------------------------------------------------------------------

    def _resolve_opponent(self) -> int:
        assert self._opponent_person is not None and self._agent_person is not None
        opp = self._opponent_person
        agent = self._agent_person
        obs = self._observe()

        stance_mask = np.array([1, 1, 0, 0], dtype=np.int8)
        stance = Stance(self._opponent.act(obs, stance_mask, agent, opp, Phase.STANCE))
        pool = opp.attack_pool if stance == Stance.ATTACK else opp.evasion_pool
        roll = opp.roll(pool)

        wp_mask = np.array([0, 0, 1, 1], dtype=np.int8)
        if opp.will_superficial_damage + opp.will_aggravated_damage >= opp.will_cap:
            wp_mask[3] = 0
        wp = WillpowerAction(self._opponent.act(obs, wp_mask, agent, opp, Phase.WILLPOWER))
        if wp == WillpowerAction.USE:
            opp.will_reroll(roll)

        return roll.successes

    @staticmethod
    def _resolve_damage(margin: int, target: Person) -> float:
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
        assert self.np_random is not None

        char_a = self._build_person(self._agent_config, self.np_random)
        char_b = self._build_person(self._opponent_config, self.np_random)

        if self.randomize_roles and self.np_random.random() < 0.5:
            self._agent_person, self._opponent_person = char_b, char_a
        else:
            self._agent_person, self._opponent_person = char_a, char_b

        self.round = 0
        self._reset_round_state()
        return self._observe(), {"action_mask": self.action_mask()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self._agent_person is not None and self._opponent_person is not None

        action = int(action)
        mask = self.action_mask()
        if not mask[action]:
            logger.warning("Invalid action %d in phase %s, falling back to %d", action, self._phase, int(mask.argmax()))
            action = int(mask.argmax())

        if self._phase == Phase.STANCE:
            return self._step_stance(action)
        return self._step_willpower(action)

    def _step_stance(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self._agent_person is not None and self._opponent_person is not None

        self._agent_stance = Stance(action)
        p = self._agent_person
        pool = p.attack_pool if self._agent_stance == Stance.ATTACK else p.evasion_pool
        self._agent_roll = p.roll(pool)

        self._opponent_successes = self._resolve_opponent()

        self._phase = Phase.WILLPOWER
        return (
            self._observe(pre_will_roll=self._agent_roll),
            0.0,
            False,
            False,
            {"action_mask": self.action_mask()},
        )

    def _step_willpower(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self._agent_person is not None and self._opponent_person is not None
        assert self._agent_roll is not None and self._opponent_successes is not None

        wp = WillpowerAction(action)
        if wp == WillpowerAction.USE:
            self._agent_person.will_reroll(self._agent_roll)

        agent_succ = self._agent_roll.successes
        opp_succ = self._opponent_successes

        reward = 0.0
        if agent_succ > opp_succ:
            reward += self._resolve_damage(agent_succ - opp_succ, self._opponent_person)
        elif opp_succ > agent_succ:
            reward += self._resolve_damage(opp_succ - agent_succ, self._agent_person)

        if self._opponent_person.is_defeated:
            self._reset_round_state()
            return self._observe(), 1.0, True, False, {"action_mask": self.action_mask()}
        if self._agent_person.is_defeated:
            self._reset_round_state()
            return self._observe(), -1.0, True, False, {"action_mask": self.action_mask()}

        self.round += 1
        truncated = self.round >= self.max_rounds
        self._reset_round_state()

        if self.render_mode == "human":
            self._render()

        return self._observe(), reward, False, truncated, {"action_mask": self.action_mask()}

    def _render(self) -> None:
        assert self._agent_person is not None and self._opponent_person is not None
        a, o = self._agent_person, self._opponent_person
        print(f"--- Round {self.round} ---")
        print(
            f"  Agent  HP: {a.superficial_damage + a.aggravated_damage}/{a.hp_cap} "
            f"(S:{a.superficial_damage} A:{a.aggravated_damage})  "
            f"Will: {a.will_superficial_damage + a.will_aggravated_damage}/{a.will_cap}"
        )
        print(
            f"  Opponent HP: {o.superficial_damage + o.aggravated_damage}/{o.hp_cap} "
            f"(S:{o.superficial_damage} A:{o.aggravated_damage})  "
            f"Will: {o.will_superficial_damage + o.will_aggravated_damage}/{o.will_cap}  "
            f"Hunger: {o.hunger}"
        )
