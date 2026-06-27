from unittest.mock import patch

from vtm_agent.engine import BarType, Damage, DamageType, Vampire


class TestVampire:
    def test_vampire_inherits_person(self) -> None:
        v = Vampire(hp=5, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3, hunger=2)
        assert v.hunger == 2
        assert not v.is_impaired

    def test_vampire_superficial_hp_halved(self) -> None:
        v = Vampire(hp=10, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3)
        v.apply_damage(Damage(3, DamageType.SUPERFICIAL, BarType.HP))
        assert v.superficial_damage == 2

    def test_vampire_aggravated_hp_not_halved(self) -> None:
        v = Vampire(hp=10, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3)
        v.apply_damage(Damage(3, DamageType.AGGRAVATED, BarType.HP))
        assert v.aggravated_damage == 3

    def test_vampire_will_damage_not_halved(self) -> None:
        v = Vampire(hp=10, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3)
        v.apply_damage(Damage(3, DamageType.SUPERFICIAL, BarType.WILL))
        assert v.will_superficial_damage == 3

    def test_roll_with_hunger(self) -> None:
        v = Vampire(hp=5, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3, hunger=3)
        r = v.roll(5)
        assert len(r.blood_pool) == 3
        assert len(r.common_pool) == 2

    def test_rouse_check_increases_hunger(self) -> None:
        with patch("random.randint", return_value=1):
            v = Vampire(hp=5, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3, hunger=1)
            v.rouse_check()
        assert v.hunger == 2
