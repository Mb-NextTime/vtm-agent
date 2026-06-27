# vtm-agent

RL agent for 1v1 combat in *Vampire: The Masquerade* (V5).

## Stack

- Python 3.11+, managed via `uv`.
- Core: `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`.

## Engine

`src/vtm_agent/engine/` implements V5 combat rules.

### Damage

Damage is a frozen dataclass with three fields: `value` (int), `dtype` (`SUPERFICIAL` or `AGGRAVATED`), `bar` (`HP` or `WILL`). Applied via `Person.apply_damage()` which fills track cells one by one. When a track overflows, each excess point converts 1 superficial damage to aggravated (1:1), regardless of incoming damage type.

Defeat occurs when aggravated damage fills a track completely. A track full of superficial damage alone causes impairment (−2 penalty) but not defeat. This is a deliberate design choice for RL — in V5 a full track of any damage type would incapacitate.

### Dice

`Roll(pool_size, blood_pool_size=0)` generates d10s. Successes are counted per V5: each die ≥6 is one success; each pair of 10s adds 2 critical successes. Blood pool dice (hunger dice for vampires) are kept separate from the common pool and are not eligible for willpower rerolls.

### Characters

| Class | Base | Extras |
|---|---|---|
| `Person` | HP/will tracks, attack/evasion pools, impairment penalty (−2 per filled track, stacks) | — |
| `Hunter` | extends Person | — |
| `Vampire` | extends Person | Hunger, Rouse Check (d10 < 6 → hunger+1), Blood Surge (+2 dice, triggers Rouse Check), superficial HP damage halved `(value+1)//2` |

- Impairment penalty applies to both attack and evasion pools. Minimum pool is 1.
- `attack_modifier` is added to the base attack pool before impairment.

### Willpower

Spending 1 point of willpower (`will_reroll`) rerolls up to 3 dice that are not successes (values ≤5) and deals 1 superficial will damage.

### Known simplifications

- Critical successes: each pair of 10s adds +2, not the full V5 critical hit mechanic with hunger dice.
- No willpower attacks against opponents (damage is always applied to the HP bar).
- Vampire scripted opponent does not use Blood Surge, Willpower, or Rouse Check.

## Development

```sh
make fmt      # ruff format + ruff check --fix
make lint     # ruff check + mypy
make test     # pytest
make test-cov # pytest with coverage
make test tests/engine/test_person.py::TestPerson::test_impairment_hp  # single test
```
