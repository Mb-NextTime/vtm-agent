from unittest.mock import patch

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
        with patch("random.randint", return_value=8):
            r.reroll([0, 2])
        assert r.common_pool[0] == 8
        assert r.common_pool[2] == 8
        # Other indices unchanged
        assert r.common_pool[1] == old[1]
        assert r.common_pool[3] == old[3]
        assert r.common_pool[4] == old[4]

    def test_reroll_out_of_range(self) -> None:
        r = Roll(3)
        with pytest.raises(IncorrectDataError):
            r.reroll([5])

    def test_reroll_failed_rerolls_non_successes(self) -> None:
        r = Roll(5)
        for i in range(len(r.common_pool)):
            r.common_pool[i] = 1
        with patch("random.randint", return_value=10):
            r.reroll_failed(max_count=3)
        changed = sum(1 for v in r.common_pool if v != 1)
        assert changed == 3

    def test_reroll_failed_max_count(self) -> None:
        r = Roll(10)
        for i in range(len(r.common_pool)):
            r.common_pool[i] = 1
        with patch("random.randint", return_value=10):
            r.reroll_failed(max_count=3)
        changed = sum(1 for v in r.common_pool if v != 1)
        assert changed == 3

    def test_reroll_failed_skips_successes(self) -> None:
        r = Roll(4)
        r.common_pool = [10, 10, 1, 1]
        r.reroll_failed(max_count=3)
        assert r.common_pool[0] == 10
        assert r.common_pool[1] == 10

    def test_blood_pool_accounted(self) -> None:
        r = Roll(2, blood_pool_size=1)
        assert len(r.blood_pool) == 1
        assert len(r.common_pool) == 1
