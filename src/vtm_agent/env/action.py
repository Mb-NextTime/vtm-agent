from enum import IntEnum


class Phase(IntEnum):
    STANCE = 0
    WILLPOWER = 1


class Stance(IntEnum):
    ATTACK = 0
    EVADE = 1


class WillpowerAction(IntEnum):
    SKIP = 0
    USE = 1
