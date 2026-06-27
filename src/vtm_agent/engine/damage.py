from dataclasses import dataclass
from enum import Enum, auto


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
