import random

from vtm_agent.engine.damage import BarType, Damage, DamageType
from vtm_agent.engine.dice import Roll
from vtm_agent.engine.person import Person


class BloodRageError(Exception):
    pass


class Vampire(Person):
    def __init__(
        self,
        hp: int,
        will: int,
        attack_pool: int,
        attack_modifier: int,
        evasion_pool: int,
        hunger: int = 1,
        surge_modifier: int = 2,
    ) -> None:
        super().__init__(hp, will, attack_pool, attack_modifier, evasion_pool)
        self.hunger = hunger
        self.surge_modifier = surge_modifier

    def rouse_check(self) -> None:
        if random.randint(1, 10) < 6:
            if self.hunger >= 5:
                raise BloodRageError()
            self.hunger += 1

    def roll(self, pool: int) -> Roll:
        return Roll(pool, self.hunger)

    def blood_surge_roll(self, base_pool: int) -> Roll:
        roll = Roll(base_pool + self.surge_modifier, self.hunger)
        self.rouse_check()
        return roll

    def apply_damage(self, damage: Damage) -> None:
        if damage.dtype == DamageType.SUPERFICIAL and damage.bar == BarType.HP:
            halved_val = (damage.value + 1) // 2
            super().apply_damage(Damage(halved_val, damage.dtype, damage.bar))
        else:
            super().apply_damage(damage)
