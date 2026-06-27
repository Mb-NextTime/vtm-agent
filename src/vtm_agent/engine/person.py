from vtm_agent.engine.damage import BarType, Damage, DamageType
from vtm_agent.engine.dice import Roll


class Person:
    def __init__(
        self,
        hp: int,
        will: int,
        attack_pool: int,
        attack_modifier: int,
        evasion_pool: int,
    ) -> None:
        self.hp_cap = hp
        self.superficial_damage = 0
        self.aggravated_damage = 0

        self.will_cap = will
        self.will_superficial_damage = 0
        self.will_aggravated_damage = 0

        self.base_attack_pool = attack_pool
        self.attack_modifier = attack_modifier
        self.base_evasion_pool = evasion_pool

    @property
    def is_impaired(self) -> bool:
        return (self.superficial_damage + self.aggravated_damage) >= self.hp_cap

    @property
    def is_impaired_will(self) -> bool:
        return (self.will_superficial_damage + self.will_aggravated_damage) >= self.will_cap

    @property
    def impare_penalty(self) -> int:
        return (2 if self.is_impaired else 0) + (2 if self.is_impaired_will else 0)

    @property
    def attack_pool(self) -> int:
        pool = self.base_attack_pool + self.attack_modifier - self.impare_penalty
        return max(1, pool)

    @property
    def evasion_pool(self) -> int:
        pool = self.base_evasion_pool - self.impare_penalty
        return max(1, pool)

    @property
    def is_defeated(self) -> bool:
        return self.aggravated_damage >= self.hp_cap or self.will_aggravated_damage >= self.will_cap

    def apply_damage(self, damage: Damage) -> None:
        val = damage.value
        if val <= 0:
            return

        if damage.bar == BarType.HP:
            superficial = self.superficial_damage
            aggravated = self.aggravated_damage
            bar_cap = self.hp_cap
        elif damage.bar == BarType.WILL:
            superficial = self.will_superficial_damage
            aggravated = self.will_aggravated_damage
            bar_cap = self.will_cap
        else:
            raise TypeError(f"Unexpected BarType: {damage.bar}")

        for _ in range(val):
            if superficial + aggravated < bar_cap:
                if damage.dtype == DamageType.SUPERFICIAL:
                    superficial += 1
                else:
                    aggravated += 1
            else:
                if superficial > 0:
                    superficial -= 1
                    aggravated += 1
                else:
                    aggravated += 1

        if damage.bar == BarType.HP:
            self.superficial_damage = superficial
            self.aggravated_damage = aggravated
        elif damage.bar == BarType.WILL:
            self.will_superficial_damage = superficial
            self.will_aggravated_damage = aggravated
        else:
            raise TypeError(f"Unexpected BarType: {damage.bar}")

    def roll(self, pool: int) -> Roll:
        return Roll(pool)

    def will_reroll(self, roll: Roll) -> None:
        if self.will_superficial_damage + self.will_aggravated_damage >= self.will_cap:
            return
        roll.reroll_failed(max_count=3)
        self.apply_damage(Damage(1, DamageType.SUPERFICIAL, BarType.WILL))
