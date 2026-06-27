import pytest

from vtm_agent.engine.dice import IncorrectDataError, Roll


class TestRoll:
    def test_successes_range(self) -> None:
        for _ in range(100):
            r = Roll(10)
            assert 0 <= r.successes <= 10 * 3

    def test_reroll(self) -> None:
        r = Roll(5)
        old = r.common_pool[:]
        r.reroll([0, 2])
        assert r.common_pool[0] != old[0] or r.common_pool[2] != old[2]

    def test_reroll_out_of_range(self) -> None:
        r = Roll(3)
        with pytest.raises(IncorrectDataError):
            r.reroll([5])

    def test_blood_pool_accounted(self) -> None:
        r = Roll(2, blood_pool_size=1)
        assert len(r.blood_pool) == 1
        assert len(r.common_pool) == 1
