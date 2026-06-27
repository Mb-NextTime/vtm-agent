import numpy as np

from vtm_agent.engine import Hunter, Vampire
from vtm_agent.env.action import Phase, Stance, WillpowerAction
from vtm_agent.env.opponent import RandomOpponent, ScriptedOpponent


class TestScriptedOpponent:
    def test_always_attacks_skips_willpower(self) -> None:
        opp = ScriptedOpponent()
        a1 = opp.act(None, None, None, None, Phase.STANCE)  # type: ignore[arg-type]
        a2 = opp.act(None, None, None, None, Phase.WILLPOWER)  # type: ignore[arg-type]
        assert a1 == int(Stance.ATTACK)
        assert a2 == int(WillpowerAction.SKIP)


class TestRandomOpponent:
    def test_respects_action_mask_stance(self) -> None:
        opp = RandomOpponent(rng=np.random.default_rng(0))
        mask = np.array([0, 1, 0, 0], dtype=np.int8)
        actions = {opp.act(None, mask, None, None, Phase.STANCE) for _ in range(100)}  # type: ignore[arg-type]
        assert actions == {1}

    def test_deterministic_seed(self) -> None:
        opp1 = RandomOpponent(rng=np.random.default_rng(42))
        opp2 = RandomOpponent(rng=np.random.default_rng(42))
        mask = np.ones(4, dtype=np.int8)
        for _ in range(20):
            a1 = opp1.act(None, mask, None, None, Phase.STANCE)  # type: ignore[arg-type]
            a2 = opp2.act(None, mask, None, None, Phase.STANCE)  # type: ignore[arg-type]
            assert a1 == a2

    def test_with_real_objects(self) -> None:
        opp = RandomOpponent(rng=np.random.default_rng(0))
        hunter = Hunter(hp=10, will=10, attack_pool=6, attack_modifier=0, evasion_pool=5)
        vampire = Vampire(hp=10, will=10, attack_pool=3, attack_modifier=0, evasion_pool=3, hunger=1)
        mask = np.ones(4, dtype=np.int8)
        obs = np.zeros(30, dtype=np.float32)
        action = opp.act(obs, mask, hunter, vampire, Phase.STANCE)
        assert isinstance(action, int)

    def test_fallback_on_empty_mask(self) -> None:
        opp = RandomOpponent(rng=np.random.default_rng(0))
        mask = np.zeros(4, dtype=np.int8)
        action = opp.act(None, mask, None, None, Phase.STANCE)  # type: ignore[arg-type]
        assert action == 0
