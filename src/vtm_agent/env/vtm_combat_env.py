import numpy as np
from gymnasium import Env, spaces

from vtm_agent.engine import (
    BarType,
    BloodRageError,
    Damage,
    DamageType,
    Hunter,
    Vampire,
)
from vtm_agent.env.action import Action
from vtm_agent.env.opponent import Opponent, ScriptedOpponent


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

        self._hunter_params = (hunter_hp, hunter_will, hunter_attack, hunter_attack_mod, hunter_evasion)
        self._vampire_params = (
            vampire_hp,
            vampire_will,
            vampire_attack,
            vampire_attack_mod,
            vampire_evasion,
            vampire_hunger,
        )
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
            ],
            dtype=np.float32,
        )

    def _action_mask(self, is_hunter: bool = True) -> np.ndarray:
        person = self.hunter if is_hunter else self.vampire
        assert person is not None
        mask = np.ones(len(Action), dtype=np.int8)
        will_filled = person.will_superficial_damage + person.will_aggravated_damage >= person.will_cap
        if will_filled:
            mask[Action.WILLPOWER_ATTACK] = 0
            mask[Action.WILLPOWER_EVADE] = 0
        return mask

    # ------------------------------------------------------------------
    # Combat resolution
    # ------------------------------------------------------------------

    def _resolve_stance(self, person: Hunter | Vampire, action: Action) -> tuple[int, bool]:
        """Execute one side's stance and return (roll_successes, used_willpower)."""
        if action in (Action.ATTACK, Action.WILLPOWER_ATTACK):
            roll = person.roll(person.attack_pool)
        else:
            roll = person.roll(person.evasion_pool)

        willpower = action in (Action.WILLPOWER_ATTACK, Action.WILLPOWER_EVADE)
        if willpower:
            person.will_reroll(roll)

        return roll.successes, willpower

    def _resolve_round(
        self,
        h: Hunter,
        v: Vampire,
        agent_action: Action,
        opp_action: Action,
    ) -> float:
        """Resolve a round where both sides pick a stance. Returns reward."""
        agent_succ, _ = self._resolve_stance(h, agent_action)
        opp_succ, _ = self._resolve_stance(v, opp_action)

        if agent_succ > opp_succ:
            margin = agent_succ - opp_succ
            dmg_type = DamageType.AGGRAVATED if margin >= 5 else DamageType.SUPERFICIAL
            v.apply_damage(Damage(margin, dmg_type, BarType.HP))
            return float(margin) * 0.1
        elif opp_succ > agent_succ:
            margin = opp_succ - agent_succ
            dmg_type = DamageType.AGGRAVATED if margin >= 5 else DamageType.SUPERFICIAL
            h.apply_damage(Damage(margin, dmg_type, BarType.HP))
            return -float(margin) * 0.1
        return 0.0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)

        hp = self._hunter_params
        self.hunter = Hunter(hp=hp[0], will=hp[1], attack_pool=hp[2], attack_modifier=hp[3], evasion_pool=hp[4])
        vp = self._vampire_params
        self.vampire = Vampire(
            hp=vp[0],
            will=vp[1],
            attack_pool=vp[2],
            attack_modifier=vp[3],
            evasion_pool=vp[4],
            hunger=vp[5],
        )
        self.round = 0
        return self._observe(), {"action_mask": self._action_mask(is_hunter=True)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.hunter is not None and self.vampire is not None
        h, v = self.hunter, self.vampire

        agent_action = Action(action)
        opp_action = self._opponent.act(self._observe(), self._action_mask(is_hunter=False), h, v)

        reward = 0.0
        try:
            reward += self._resolve_round(h, v, agent_action, opp_action)
        except BloodRageError:
            return self._observe(), -1.0, True, False, {"action_mask": self._action_mask(is_hunter=True)}

        if v.is_defeated:
            return self._observe(), 1.0, True, False, {"action_mask": self._action_mask(is_hunter=True)}
        if h.is_defeated:
            return self._observe(), -1.0, True, False, {"action_mask": self._action_mask(is_hunter=True)}

        self.round += 1
        truncated = self.round >= self.max_rounds

        if self.render_mode == "human":
            self._render()

        return self._observe(), reward, False, truncated, {"action_mask": self._action_mask(is_hunter=True)}

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
