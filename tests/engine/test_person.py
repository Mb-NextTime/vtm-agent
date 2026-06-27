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
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.HP))
        assert p.superficial_damage == 2
        assert p.aggravated_damage == 0
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.HP))
        assert p.superficial_damage == 0
        assert p.aggravated_damage == 2

    def test_defeated_by_aggravated_hp(self) -> None:
        p = Person(hp=3, will=3, attack_pool=3, attack_modifier=0, evasion_pool=3)
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
        assert p.attack_pool == 1
        assert p.evasion_pool == 1

    def test_zero_damage_noop(self) -> None:
        p = Person(hp=5, will=5, attack_pool=5, attack_modifier=0, evasion_pool=5)
        p.apply_damage(Damage(0, DamageType.SUPERFICIAL, BarType.HP))
        assert p.superficial_damage == 0

    def test_attack_modifier_adds_to_pool(self) -> None:
        p = Person(hp=5, will=5, attack_pool=4, attack_modifier=2, evasion_pool=3)
        assert p.attack_pool == 6  # 4 + 2

    def test_attack_modifier_affected_by_impairment(self) -> None:
        p = Person(hp=2, will=2, attack_pool=5, attack_modifier=2, evasion_pool=3)
        p.apply_damage(Damage(2, DamageType.SUPERFICIAL, BarType.HP))
        assert p.is_impaired
        assert p.attack_pool == 5  # 5 + 2 - 2 = 5

    def test_overflow_aggravated_converts_superficial(self) -> None:
        p = Person(hp=5, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3)
        p.apply_damage(Damage(5, DamageType.SUPERFICIAL, BarType.HP))
        assert p.superficial_damage == 5
        assert p.aggravated_damage == 0
        p.apply_damage(Damage(1, DamageType.AGGRAVATED, BarType.HP))
        assert p.superficial_damage == 4  # 1 superficial converted
        assert p.aggravated_damage == 1  # aggravated applied

    def test_overflow_aggravated_full_aggravated(self) -> None:
        p = Person(hp=5, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3)
        p.apply_damage(Damage(5, DamageType.AGGRAVATED, BarType.HP))
        assert p.is_defeated
        p.apply_damage(Damage(2, DamageType.AGGRAVATED, BarType.HP))
        assert p.aggravated_damage == 7  # extra aggravated still counted

    def test_will_reroll_applies_will_damage(self) -> None:
        p = Person(hp=5, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3)
        roll = p.roll(5)
        p.will_reroll(roll)
        assert p.will_superficial_damage == 1

    def test_will_reroll_noop_when_will_track_full(self) -> None:
        p = Person(hp=5, will=1, attack_pool=3, attack_modifier=0, evasion_pool=3)
        p.apply_damage(Damage(1, DamageType.SUPERFICIAL, BarType.WILL))
        assert p.will_superficial_damage + p.will_aggravated_damage >= p.will_cap
        roll = p.roll(3)
        p.will_reroll(roll)
        assert p.will_superficial_damage == 1  # no additional damage
