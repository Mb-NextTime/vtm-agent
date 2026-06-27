# AGENTS.md — vtm-agent

## Stack

- Python 3.11+, managed via `uv`.
- Single package `vtm_agent` under `src/`.
- Core deps: `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`.
- Run entry: `uv run python3 -c "..."` or `source .venv/bin/activate && python3 ...`.

## Project layout

```
src/vtm_agent/
  engine/          — V5 combat engine (refactored from vtm_hunter_vs_vampire.py)
    damage.py      — Damage, DamageType, BarType
    dice.py        — Roll, IncorrectDataError
    person.py      — Person (base class with damage/will tracking, impairment, is_vampire, hunger)
    hunter.py      — Hunter (extends Person, is_vampire=False, hunger=0)
    vampire.py     — Vampire (extends Person, adds Hunger/Rouse Check/blood surge)
  env/
    action.py          — Phase, Stance, WillpowerAction enums
    opponent.py        — Opponent protocol + ScriptedOpponent, RandomOpponent
    vtm_combat_env.py  — Gymnasium Env for 1v1 combat (generic agent vs opponent)
```

## Engine details

- `Person.is_impaired_will` was reading HP fields instead of will fields — fixed (uses `will_superficial_damage`/`will_aggravated_damage`).
- `attack_modifier` is now applied in `attack_pool` property (`base_attack_pool + attack_modifier - penalty`).
- Willpower rerolls up to 3 non-success dice (values ≤5) via `Roll.reroll_failed(max_count=3)`.
- `BloodRageError` exists; env catches it as defeat (vampire scripted opponent does not raise it).
- Vampire superficial HP damage halved: `(value + 1) // 2`.
- Old monolithic file `vtm_hunter_vs_vampire.py` kept for reference but superseded by `src/vtm_agent/engine/`.
- `Person.is_vampire` — `False` base, `True` in Vampire.
- `Person.hunger` — `0` base, property in Vampire (backed by `_hunger`).

## Environment design

### Action space (4 discrete)
- **STANCE phase** (index 0–1): `ATTACK=0`, `EVADE=1`
- **WILLPOWER phase** (index 2–3): `SKIP=2`, `USE=3`
- Action mask switches: STANCE → `[1,1,0,0]`, WILLPOWER → `[0,0,1,1]`
- `USE` masked when agent's will track is full.

### Observation (30-dim Box(0,1))
| Indices | Field |
|---|---|
| 0–12 | **Self block** — HP ratio, Will ratio, aggravated ratios, impairment, is_vampire, hunger, base pools, modifier, caps |
| 13–25 | **Opponent block** — same 13 fields |
| 26–29 | **Context** — phase, round_progress, common_successes, blood_successes |

### Generic agent/opponent
- No fixed "agent = Hunter, opponent = Vampire".
- `randomize_roles=True` — 50% chance agent/opponent swap on reset().
- `CharacterConfig` controls stat ranges, is_vampire flag.
- Characters generated randomly per `reset()` from ranges.
- All combos possible: Hunter vs Hunter, Vampire vs Vampire, Hunter vs Vampire (both ways).

### Opponent protocol
```python
def act(obs, action_mask, agent, opponent, phase) -> int
```
- `agent` / `opponent` are `Hunter | Vampire`.
- ScriptedOpponent: always ATTACK + SKIP.
- RandomOpponent: random valid action from mask.

## Development workflow

Run these after any change:

```sh
make fmt      # ruff format + ruff check --fix
make lint     # ruff check + mypy
make test     # pytest
make test-cov # pytest with coverage
```

Single test: `make test tests/engine/test_person.py::TestPerson::test_impairment_hp`

Lint+typecheck must pass before pushing.

## RL training

- Environment: `VTMCombatEnv` in `src/vtm_agent/env/vtm_combat_env.py`.
- Action space: 4 discrete actions (STANCE 0–1, WILLPOWER 2–3).
- Action masking used for willpower actions (disabled when will track full).
- Observation: 30-dim Box(0,1) — full self + opponent + context.
- Use `MaskablePPO` from `sb3_contrib` for training with action masks.
- Training entry point: not yet added (no `__main__` or training script yet).
