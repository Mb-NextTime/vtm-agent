from vtm_agent.engine import BarType, Damage, DamageType, Person


class TestPerson:
    def test_initial_state(self) -> None:
        p = Person(hp=6, will=5, attack_pool=5, attack_modifier=0, evasion_pool=3)
        assert not p.is_impaired
        assert not p.is_impaired_will
        assert not p.is_defeated
        assert p.attack_pool == 5
        assert p.evasion_pool == 3

    def test_superficial_damage_hp(self) -> None:
        p = Person(hp=3, will=3, attack_pool=3, attack_modifier=0, evasion_pool=3)
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.HP))
        assert p.superficial_damage == 2
        assert p.aggravated_damage == 0
        assert not p.is_impaired

    def test_impairment_hp(self) -> None:
        p = Person(hp=3, will=3, attack_pool=3, attack_modifier=0, evasion_pool=3)
        p.apply_damage(Damage(3, DamageType.SUPERFICIAL, BarType.HP))
        assert p.is_impaired
        assert p.impare_penalty == 2

    def test_impairment_will(self) -> None:
        p = Person(hp=3, will=3, attack_pool=3, attack_modifier=0, evasion_pool=3)
        p.apply_damage(Damage(3, DamageType.SUPERFICIAL, BarType.WILL))
        assert p.is_impaired_will
        assert p.impare_penalty == 2

    def test_double_impairment_stacks(self) -> None:
        p = Person(hp=2, will=2, attack_pool=5, attack_modifier=0, evasion_pool=5)
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.HP))
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.WILL))
        assert p.impare_penalty == 4
        assert p.attack_pool == max(1, 5 - 4)

    def test_damage_overflow_converts_superficial_to_aggravated(self) -> None:
        p = Person(hp=2, will=2, attack_pool=3, attack_modifier=0, evasion_pool=3)
        # Fill with superficial
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.HP))
        assert p.superficial_damage == 2
        assert p.aggravated_damage == 0
        # Overflow damage: superficial -> aggravated
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.HP))
        # First point of overflow converts 1 superficial -> 1 aggravated
        # Second point of overflow converts 1 superficial -> 1 aggravated
        assert p.superficial_damage == 0
        assert p.aggravated_damage == 2

    def test_defeated_by_aggravated_hp(self) -> None:
        p = Person(hp=3, will=3, attack_pool=3, attack_modifier=0, evasion_pool=3)
        assert not p.is_defeated
        p.apply_damage(Damage(3, DamageType.AGGRAVATED, BarType.HP))
        assert p.is_defeated

    def test_defeated_by_aggravated_will(self) -> None:
        p = Person(hp=3, will=3, attack_pool=3, attack_modifier=0, evasion_pool=3)
        p.apply_damage(Damage(3, DamageType.AGGRAVATED, BarType.WILL))
        assert p.is_defeated

    def test_superficial_alone_does_not_defeat(self) -> None:
        p = Person(hp=2, will=2, attack_pool=3, attack_modifier=0, evasion_pool=3)
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.HP))
        assert not p.is_defeated

    def test_attack_pool_min_one(self) -> None:
        p = Person(hp=2, will=2, attack_pool=2, attack_modifier=0, evasion_pool=2)
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.HP))
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.WILL))
        # penalty = 4, pool goes to 1 minimum
        assert p.attack_pool == 1
        assert p.evasion_pool == 1

    def test_zero_damage_noop(self) -> None:
        p = Person(hp=5, will=5, attack_pool=5, attack_modifier=0, evasion_pool=5)
        p.apply_damage(Damage(0, DamageType.SUPERFICIAL, BarType.HP))
        assert p.superficial_damage == 0
