from unittest.mock import patch

import pytest

from vtm_agent.engine import BarType, BloodRageError, Damage, DamageType, Vampire


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

    def test_rouse_check_does_not_increase_hunger_on_success(self) -> None:
        with patch("random.randint", return_value=10):
            v = Vampire(hp=5, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3, hunger=1)
            v.rouse_check()
        assert v.hunger == 1

    def test_blood_surge_roll_increases_pool(self) -> None:
        v = Vampire(hp=5, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3, hunger=2)
        with patch("random.randint", return_value=10):  # rouse check succeeds
            r = v.blood_surge_roll(5)
        assert len(r.common_pool) + len(r.blood_pool) == 5 + v.surge_modifier
        assert v.hunger == 2  # hunger unchanged on success

    def test_blood_surge_roll_calls_rouse_check(self) -> None:
        v = Vampire(hp=5, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3, hunger=4)
        with patch("random.randint", return_value=1):  # rouse check fails
            v.blood_surge_roll(5)
        assert v.hunger == 5

    def test_blood_rage_error_raised(self) -> None:
        v = Vampire(hp=5, will=5, attack_pool=3, attack_modifier=0, evasion_pool=3, hunger=5)
        with patch("random.randint", return_value=1):  # fails, hunger would exceed 5
            with pytest.raises(BloodRageError):
                v.rouse_check()
        assert v.hunger == 5  # hunger not incremented on frenzy

    def test_attack_modifier_works(self) -> None:
        v = Vampire(hp=5, will=5, attack_pool=4, attack_modifier=1, evasion_pool=3, hunger=1)
        assert v.attack_pool == 5
