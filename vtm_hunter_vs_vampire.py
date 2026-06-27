import math
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import copy


class IncorrectData(Exception):
    pass

class BloodRage(Exception):
    pass

class DamageType(Enum):
    SUPERFICIAL = auto()
    AGGRAVATED = auto()

class BarType(Enum):
    HP = auto()
    WILL = auto()

@dataclass(frozen=True)
class Damage:
    value: int
    dtype: DamageType
    bar: BarType


class Roll:
    def __init__(self, pool_size: int, blood_pool_size: int=0):
        pool = [random.randint(1, 10) for _ in range(pool_size)]
        self.blood_pool = pool[:blood_pool_size]
        self.common_pool = pool[blood_pool_size:]

    @property
    def successes(self):
        pool = self.blood_pool + self.common_pool
        # TODO Optimization
        base_successes = sum(1 for die in pool if die >= 6)
        tens = pool.count(10)
        critical_bonus = (tens // 2) * 2
        total_successes = base_successes + critical_bonus
        return total_successes

    def reroll(self, indices: List[int]):
        for i in indices:
            if i >= len(self.common_pool):
                raise IncorrectData(f"Reroll index ({i = }) out of range")
            self.common_pool[i] = random.randint(1, 10)


class Person:
    def __init__(self,
                 hp: int,
                 will: int,
                 attack_pool: int,
                 attack_modifier: int,
                 evasion_pool: int):
        """
        hp - число пустых ячеек HP
        will - число пустых ячеек воли
        """
        # Ячейки трека здоровья
        self.hp_cap = hp
        self.superficial_damage = 0
        self.aggravated_damage = 0

        # Ячейки трека воли
        self.will_cap = will
        self.will_superficial_damage = 0
        self.will_aggravated_damage = 0

        # Базовые параметры без учета штрафов здоровья
        self.base_attack_pool = attack_pool
        self.attack_modifier = attack_modifier
        self.base_evasion_pool = evasion_pool

    @property
    def is_impaired(self) -> bool:
        """Персонаж Изнурен (Impaired), когда сумма любого урона равна или больше трека HP."""
        return (self.superficial_damage + self.aggravated_damage) >= self.hp_cap

    @property
    def is_impaired_will(self) -> bool:
        """Персонаж изнурен по воле, когда сумма любого урона равна или больше трека воли."""
        return (self.superficial_damage + self.aggravated_damage) >= self.hp_cap

    @property
    def impare_penalty(self) -> int:
        """Штраф к проверкам в зависимости от изнурения"""
        return (2 if self.is_impaired else 0) + (2 if self.is_impaired_will else 0)

    @property
    def attack_pool(self) -> int:
        """Динамический пул атаки со штрафом за изнурение."""
        pool = self.base_attack_pool - self.impare_penalty
        return max(1, pool)  # По правилам V5 пул не может упасть ниже 1 куба

    @property
    def evasion_pool(self) -> int:
        """Динамический пул уклонения со штрафом за изнурение."""
        pool = self.base_evasion_pool - self.impare_penalty
        return max(1, pool)

    @property
    def is_defeated(self) -> bool:
        """Поражение наступает, когда весь трек заполнен исключительно тяжелым уроном."""
        return self.aggravated_damage >= self.hp_cap or self.will_aggravated_damage >= self.will_cap

    def apply_damage(self, damage: Damage) -> None:
        """Пошаговое заполнение фиксированных ячеек здоровья."""
        val = damage.value
        if val <= 0:
            # TODO можно конечно добавить лечение, как отрицательный урон...
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
            raise TypeError(f"Неподходящий тип BarType {damage.bar}")

        # TODO Optimization
        for _ in range(val):
            # Если в треке еще есть полностью пустые ячейки
            if superficial + aggravated < bar_cap:
                if damage.dtype == DamageType.SUPERFICIAL:
                    superficial += 1
                else:
                    aggravated += 1
            else:
                # Трек заполнен. Любой новый урон превращает поверхностный в агрессивный
                if superficial > 0:
                    superficial -= 1
                    aggravated += 1
                else:
                    # Если поверхностного уже нет, трек забит чистым агрессивным уроном
                    aggravated += 1

        if damage.bar == BarType.HP:
            self.superficial_damage = superficial
            self.aggravated_damage = aggravated
        elif damage.bar == BarType.WILL:
            self.will_superficial_damage = superficial
            self.will_aggravated_damage = aggravated
        else:
            raise TypeError(f"Неподходящий тип BarType {damage.bar}")

    def roll(self, pool: int) -> Roll:
        return Roll(pool)

    def will_reroll(self, roll: Roll, indices: List[int]) -> None:
        """
        Изменяет сам объект roll
        """
        roll.reroll(indices)
        self.apply_damage(Damage(1, DamageType.SUPERFICIAL, BarType.WILL))


class Hunter(Person):
    pass


class Vampire(Person):
    def __init__(self,
                 hp: int,
                 will: int,
                 attack_pool: int,
                 attack_modifier: int,
                 evasion_pool: int,
                 hunger: int=1,
                 surge_modifier: int=2):
        super().__init__(hp, will, attack_pool, attack_modifier, evasion_pool)
        self.hunger = hunger
        self.surge_modifier = surge_modifier

    def rouse_check(self):
        """Проверка крови (Rouse Check).
        1-5: Неудача (голод растет). 6-10: Успех (голод не растет).
        """
        if random.randint(1, 10) < 6:
            self.hunger += 1
        if self.hunger > 5:
            # TODO Как-то это обработать
            raise BloodRage()

    def roll(self, pool: int) -> Roll:
        return Roll(pool, self.hunger)

    def blood_surge_roll(self, base_pool: int) -> Roll:
        """Совершает бросок с приливом крови"""
        roll = Roll(base_pool + self.surge_modifier, self.hunger)
        self.rouse_check()
        return roll

    def will_reroll(self, roll: Roll, indices: List[int]) -> None:
        """
        Изменяет сам объект roll
        """
        roll.reroll(indices)
        self.apply_damage(Damage(1, DamageType.SUPERFICIAL, BarType.WILL))

    def apply_damage(self, damage: Damage) -> None:
        if damage.dtype == DamageType.SUPERFICIAL and damage.bar == BarType.HP:
            halved_val = (damage.value + 1) // 2
            super().apply_damage(Damage(halved_val, damage.dtype, damage.bar))
        else:
            super().apply_damage(damage)



# def execute_attack_phase(attacker: Person, defender: Person, is_hunter_attacking: bool) -> bool:
#     """Возвращает True, если Охотник нанес 5+ маржи колом до деления пополам."""
#     atk_successes = attacker.get_attack_successes()
#     def_successes = defender.get_evasion_successes()

#     if atk_successes > def_successes:
#         margin = atk_successes - def_successes

#         if is_hunter_attacking:
#             if margin >= 5:
#                 return True  # Мгновенный кол
#             defender.apply_damage(Damage(margin, DamageType.LIGHT))
#         else:
#             defender.apply_damage(Damage(margin, DamageType.LIGHT))

#     return False


# def fight_simulation(vampire: Vampire, hunter: Hunter) -> str:
#     rounds = 0
#     while rounds < 100 and not vampire.is_defeated and not hunter.is_defeated:
#         rounds += 1

#         # Фаза 1: Вампир бьет -> Охотник защищается
#         execute_attack_phase(vampire, hunter, is_hunter_attacking=False)
#         if hunter.is_defeated:
#             return "Vampire"

#         # Фаза 2: Охотник бьет колом -> Вампир защищается
#         staked = execute_attack_phase(hunter, vampire, is_hunter_attacking=True)
#         if staked:
#             return "Hunter (Staked)"
#         if vampire.is_defeated:
#             return "Hunter"

#     return "Tie"


# def run_monte_carlo(vamp_stats: tuple, hunter_stats: tuple, iterations: int = 10000):
#     results = {"Vampire": 0, "Hunter": 0, "Hunter (Staked)": 0, "Tie": 0}

#     for _ in range(iterations):
#         vamp = Vampire(*vamp_stats)
#         hunt = Hunter(*hunter_stats)

#         outcome = fight_simulation(vamp, hunt)
#         results[outcome] += 1

#     print(f"=== Результаты {iterations:,} симуляций (с учетом Impairment) ===")
#     for outcome, count in results.items():
#         percentage = (count / iterations) * 100
#         print(f"-> {outcome}: {count} ({percentage:.2f}%)")


# # Параметры: (Здоровье, Воля, Пул Атаки, Пул Уклонения)
# VAMP_PARAMS = (5 * 2, 5 * 2, 3, 3)
# HUNTER_PARAMS = (6 * 2, 8 * 2, 6, 5)

# run_monte_carlo(VAMP_PARAMS, HUNTER_PARAMS, iterations=10000)
