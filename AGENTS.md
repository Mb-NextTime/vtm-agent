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
    person.py      — Person (base class with damage/will tracking, impairment)
    hunter.py      — Hunter (extends Person)
    vampire.py     — Vampire (extends Person, adds Hunger/Rouse Check/blood surge)
  env/
    vtm_combat_env.py — Gymnasium Env for 1v1 combat (Hunter vs scripted Vampire)
```

## Engine details

- `Person.is_impaired_will` was reading HP fields instead of will fields — fixed (uses `will_superficial_damage`/`will_aggravated_damage`).
- `attack_modifier` is now applied in `attack_pool` property (`base_attack_pool + attack_modifier - penalty`).
- Willpower rerolls up to 3 non-success dice (values ≤5) via `Roll.reroll_failed(max_count=3)`.
- `BloodRageError` exists; env catches it as defeat (vampire scripted opponent does not raise it).
- Vampire superficial HP damage halved: `(value + 1) // 2`.
- Old monolithic file `vtm_hunter_vs_vampire.py` kept for reference but superseded by `src/vtm_agent/engine/`.

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
- Action space: 4 discrete actions (`ATTACK`, `EVADE`, `WILLPOWER_ATTACK`, `WILLPOWER_EVADE`).
- Action masking used for willpower actions (disabled when will track full).
- Observation: 8-dim Box(0,1) — HP ratios, will ratios, impairment flags, normalized hunger, round progress.
- Use `MaskablePPO` from `sb3_contrib` for training with action masks.
- Training entry point: not yet added (no `__main__` or training script yet).
