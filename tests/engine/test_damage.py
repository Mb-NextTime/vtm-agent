from vtm_agent.engine.damage import BarType, Damage, DamageType


class TestDamage:
    def test_frozen(self) -> None:
        d = Damage(3, DamageType.SUPERFICIAL, BarType.HP)
        assert d.value == 3
        assert d.dtype == DamageType.SUPERFICIAL
        assert d.bar == BarType.HP

    def test_aggravated(self) -> None:
        d = Damage(2, DamageType.AGGRAVATED, BarType.WILL)
        assert d.dtype == DamageType.AGGRAVATED
        assert d.bar == BarType.WILL
