from typing import cast

import numpy as np

from vtm_agent.env import Phase, VTMCombatEnv
from vtm_agent.env.vtm_combat_env import OBS_DIM, CharacterConfig


class TestVTMCombatEnv:
    def _mask(self, info: dict[str, object]) -> np.ndarray:
        return cast(np.ndarray, info["action_mask"])

    def _full_round(self, env: VTMCombatEnv, stance: int = 0, willpower: int = 2) -> None:
        env.step(stance)
        env.step(willpower)

    def test_reset_returns_30dim_obs(self) -> None:
        env = VTMCombatEnv()
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,)
        assert "action_mask" in info

    def test_stance_phase_mask(self) -> None:
        env = VTMCombatEnv()
        mask = self._mask(env.reset()[1])
        assert mask[0] == 1
        assert mask[1] == 1
        assert mask[2] == 0
        assert mask[3] == 0

    def test_willpower_phase_mask(self) -> None:
        env = VTMCombatEnv()
        env.reset()
        _, _, _, _, info = env.step(0)
        mask = self._mask(info)
        assert mask[0] == 0
        assert mask[1] == 0
        assert mask[2] == 1
        assert mask[3] == 1

    def test_willpower_use_masked_when_will_full(self) -> None:
        env = VTMCombatEnv(
            agent_config=CharacterConfig(
                hp_range=(10, 11),
                will_range=(1, 2),
                attack_pool_range=(5, 6),
                attack_mod_range=(0, 1),
                evasion_pool_range=(5, 6),
            ),
            randomize_roles=False,
        )
        env.reset()
        env.step(0)  # stance
        env.step(3)  # USE willpower, now will track is full
        _, _, _, _, info = env.step(0)  # new round stance → returns WILLPOWER mask
        mask = self._mask(info)
        assert mask[3] == 0  # USE must be masked

    def test_willpower_equiv_skip(self) -> None:
        env = VTMCombatEnv()
        env.reset()
        _, _, _, _, info = env.step(0)  # stance
        mask = self._mask(info)
        assert mask[2] == 1  # SKIP is valid

    def test_full_round_transitions(self) -> None:
        env = VTMCombatEnv()
        env.reset()

        obs, reward, done, truncated, info = env.step(0)
        assert obs[26] == float(Phase.WILLPOWER)  # context phase index
        assert reward == 0.0

        obs, reward, done, truncated, info = env.step(2)  # SKIP
        assert obs[26] == float(Phase.STANCE)
        assert env.round == 1

    def test_env_runs_stochastic_rollout(self) -> None:
        env = VTMCombatEnv()
        env.reset()
        obs1, _, _, _, _ = env.step(0)
        _, _, _, _, _ = env.step(2)
        env.reset()
        obs2, _, _, _, _ = env.step(0)
        _, _, _, _, _ = env.step(2)
        assert obs1.shape == obs2.shape

    def test_invalid_action_falls_back(self) -> None:
        env = VTMCombatEnv()
        env.reset()
        obs, _, _, _, _ = env.step(2)  # 2 is masked in STANCE, falls back to 0 (ATTACK)
        assert obs[26] == float(Phase.WILLPOWER)

    def test_truncation(self) -> None:
        env = VTMCombatEnv(max_rounds=1)
        env.reset()
        env.step(0)  # stance
        _, _, _, truncated, _ = env.step(2)  # willpower
        assert truncated

    def test_obs_blocks_order(self) -> None:
        env = VTMCombatEnv(randomize_roles=False)
        obs, _ = env.reset()
        # agent block = indices 0-12
        agent_is_vampire = obs[6]
        opp_is_vampire = obs[6 + 13]
        assert agent_is_vampire == 0.0  # agent is Hunter (default)
        assert opp_is_vampire == 1.0  # opponent is Vampire (default)
