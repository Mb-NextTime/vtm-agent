from typing import cast

import numpy as np

from vtm_agent.env import Phase, VTMCombatEnv


class TestVTMCombatEnv:
    def _mask(self, info: dict[str, object]) -> np.ndarray:
        return cast(np.ndarray, info["action_mask"])

    def _full_round(self, env: VTMCombatEnv, stance: int = 0, willpower: int = 0) -> None:
        env.step(stance)
        env.step(willpower)

    def test_reset_returns_10dim_obs(self) -> None:
        env = VTMCombatEnv()
        obs, info = env.reset()
        assert obs.shape == (10,)
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
        assert mask[0] == 1
        assert mask[1] == 1
        assert mask[2] == 0
        assert mask[3] == 0

    def test_willpower_use_masked_when_will_full(self) -> None:
        env = VTMCombatEnv(hunter_will=1)
        env.reset()
        env.step(0)
        env.step(1)
        _, _, _, _, info = env.step(0)
        mask = self._mask(info)
        assert mask[1] == 0

    def test_full_round_transitions(self) -> None:
        env = VTMCombatEnv()
        env.reset()

        obs, reward, done, truncated, info = env.step(0)
        assert obs[8] == float(Phase.WILLPOWER)
        assert reward == 0.0
        assert 0.0 <= obs[9] <= 1.0

        obs, reward, done, truncated, info = env.step(0)
        assert obs[8] == float(Phase.STANCE)
        assert env.round == 1

    def test_env_runs_stochastic_rollout(self) -> None:
        env = VTMCombatEnv()
        env.reset()
        obs1, _, _, _, _ = env.step(0)
        _, _, _, _, _ = env.step(0)
        env.reset()
        obs2, _, _, _, _ = env.step(0)
        _, _, _, _, _ = env.step(0)
        assert obs1.shape == obs2.shape

    def test_invalid_action_falls_back(self) -> None:
        env = VTMCombatEnv()
        env.reset()
        obs, _, _, _, _ = env.step(3)
        assert obs[8] == float(Phase.WILLPOWER)

    def test_truncation(self) -> None:
        env = VTMCombatEnv(max_rounds=1)
        env.reset()
        env.step(0)  # stance
        _, _, _, truncated, _ = env.step(0)  # willpower
        assert truncated
