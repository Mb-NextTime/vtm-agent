import random


class IncorrectDataError(Exception):
    pass


class Roll:
    def __init__(self, pool_size: int, blood_pool_size: int = 0) -> None:
        pool = [random.randint(1, 10) for _ in range(pool_size)]
        self.blood_pool = pool[:blood_pool_size]
        self.common_pool = pool[blood_pool_size:]

    @property
    def successes(self) -> int:
        pool = self.blood_pool + self.common_pool
        base_successes = sum(1 for die in pool if die >= 6)
        tens = pool.count(10)
        critical_bonus = (tens // 2) * 2
        return base_successes + critical_bonus

    def reroll(self, indices: list[int]) -> None:
        for i in indices:
            if i >= len(self.common_pool):
                raise IncorrectDataError(f"Reroll index ({i = }) out of range")
            self.common_pool[i] = random.randint(1, 10)

    def reroll_failed(self, max_count: int = 3) -> None:
        failed = [i for i, v in enumerate(self.common_pool) if v <= 5]
        self.reroll(failed[:max_count])
