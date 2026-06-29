import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
from sb3_contrib import MaskablePPO

from vtm_agent.engine import Hunter, Vampire
from vtm_agent.env import CharacterConfig, Opponent, RandomOpponent, ScriptedOpponent, VTMCombatEnv
from vtm_agent.env.action import Phase, Stance


class RLOpponent:
    def __init__(self, model_path: str) -> None:
        self.model = MaskablePPO.load(model_path)

    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        agent: Hunter | Vampire,
        opponent: Hunter | Vampire,
        phase: Phase,
    ) -> int:
        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        return int(action)


def display_state(env: VTMCombatEnv, round_num: int) -> None:
    a = env.agent_person
    o = env.opponent_person
    assert a is not None and o is not None

    a_cls = "Vampire" if a.is_vampire else "Hunter"
    o_cls = "Vampire" if o.is_vampire else "Hunter"

    hp_a = a.superficial_damage + a.aggravated_damage
    will_a = a.will_superficial_damage + a.will_aggravated_damage
    hp_o = o.superficial_damage + o.aggravated_damage
    will_o = o.will_superficial_damage + o.will_aggravated_damage

    print(f"\n═══ Round {round_num} ═══\n")
    print(f"YOU ({a_cls}):")
    print(f"  HP: {hp_a}/{a.hp_cap} (S:{a.superficial_damage} A:{a.aggravated_damage})")
    print(
        f"  Will: {will_a}/{a.will_cap} (S:{a.will_superficial_damage} A:{a.will_aggravated_damage})",
    )
    print(
        f"  Attack pool: {a.attack_pool}  |  Evasion pool: {a.evasion_pool}  |  Atk mod: {a.attack_modifier:+d}",
    )
    if a.is_vampire:
        print(f"  Hunger: {a.hunger}/5")
    print()
    print(f"OPPONENT ({o_cls}):")
    print(f"  HP: {hp_o}/{o.hp_cap} (S:{o.superficial_damage} A:{o.aggravated_damage})")
    print(
        f"  Will: {will_o}/{o.will_cap} (S:{o.will_superficial_damage} A:{o.will_aggravated_damage})",
    )
    print(f"  Hunger: {o.hunger}/5")
    print(
        f"  Attack pool: {o.attack_pool}  |  Evasion pool: {o.evasion_pool}  |  Atk mod: {o.attack_modifier:+d}",
    )


def get_stance_input() -> int:
    while True:
        try:
            raw = input("Choose stance (0=ATTACK, 1=EVADE): ").strip()
            val = int(raw)
            if val in (0, 1):
                return val
            print("Invalid input. Enter 0 for ATTACK or 1 for EVADE.")
        except ValueError:
            print("Invalid input. Enter a number (0 or 1).")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)


def get_willpower_input(can_use: bool) -> int:
    prompt = "Use willpower? (0=SKIP" + (", 1=USE" if can_use else "") + "): "
    while True:
        try:
            raw = input(prompt).strip()
            val = int(raw)
            if val == 0:
                return 2
            if val == 1 and can_use:
                return 3
            if can_use:
                print("Invalid input. Enter 0 to skip or 1 to use willpower.")
            else:
                print("Invalid input. Enter 0 to skip.")
        except ValueError:
            print("Invalid input. Enter a number.")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)


def play_game(env: VTMCombatEnv, seed: int | None) -> None:
    env.reset(seed=seed)
    done = False
    round_num = 0

    while not done:
        round_num += 1
        display_state(env, round_num)

        # --- STANCE phase ---
        print("\n-- STANCE --")
        stance_action = get_stance_input()
        stance_name = Stance(stance_action).name

        obs, _, terminated, truncated, info = env.step(stance_action)
        round_res = cast("dict[str, object]", info.get("round_result", {}))
        agent_succ = cast(int, round_res.get("agent_successes", 0))
        opp_succ = cast(int, round_res.get("opponent_successes", 0))

        print(f"Your roll ({stance_name}): {agent_succ} successes")
        print(f"Opponent roll: {opp_succ} successes")

        # --- WILLPOWER phase ---
        print("\n-- WILLPOWER --")
        mask = info.get("action_mask")
        assert isinstance(mask, np.ndarray), f"Expected action_mask in info, got {info}"
        can_use = bool(mask[3])

        print(f"Your successes: {agent_succ}" + ("  (can reroll up to 3 failed dice)" if can_use else ""))

        wp_action = get_willpower_input(can_use)

        obs, reward, terminated, truncated, info = env.step(wp_action)
        round_res = cast("dict[str, object]", info.get("round_result", {}))
        final_agent_succ = cast(int, round_res.get("agent_successes", 0))
        opp_succ_final = cast(int, round_res.get("opponent_successes", 0))
        margin = cast(int, round_res.get("margin", 0))

        if wp_action == 3:
            print(f"Willpower reroll: now {final_agent_succ} successes")

        if margin > 0:
            dmg_type = "aggravated" if margin >= 5 else "superficial"
            print(f"\n>>> You deal {margin} {dmg_type} damage to opponent!")
        elif margin < 0:
            dmg_type = "aggravated" if abs(margin) >= 5 else "superficial"
            print(f"\n>>> Opponent deals {abs(margin)} {dmg_type} damage to you!")
        else:
            print("\n>>> Draw — no damage.")

        dmg_target = "opponent" if margin > 0 else ("you" if margin < 0 else "no one")
        print(
            f"Round {round_num}: Agent {final_agent_succ} succ vs Opponent "
            f"{opp_succ_final} succ — {abs(margin)} damage to {dmg_target}"
        )

        if terminated or truncated:
            done = True

    # GAME OVER
    print("\n=== GAME OVER ===")
    assert env.agent_person is not None and env.opponent_person is not None
    if env.agent_person.is_defeated:
        print("Result: YOU LOSE")
    elif env.opponent_person.is_defeated:
        print("Result: YOU WIN")
    else:
        print("Result: DRAW (max rounds reached)")
    print(f"Total rounds: {round_num}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VTM Combat — terminal game")
    parser.add_argument(
        "--opponent",
        "-o",
        choices=("random", "scripted", "rl"),
        default="scripted",
        help="Opponent type (default: scripted)",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="checkpoints/vtm_ppo_final.zip",
        help="Path to trained model .zip (required for rl opponent)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max-rounds", type=int, default=50, help="Max rounds (default: 50)")
    parser.add_argument("--as-vampire", action="store_true", help="Play as Vampire instead of Hunter")
    args = parser.parse_args()

    opponent: Opponent
    if args.opponent == "rl":
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: model not found at {model_path}", file=sys.stderr)
            sys.exit(1)
        opponent = RLOpponent(str(model_path))
    elif args.opponent == "random":
        opponent = RandomOpponent()
    else:
        opponent = ScriptedOpponent()

    agent_config = CharacterConfig(is_vampire=args.as_vampire)
    opponent_config = CharacterConfig(is_vampire=True)

    env = VTMCombatEnv(
        render_mode=None,
        opponent=opponent,
        agent_config=agent_config,
        opponent_config=opponent_config,
        max_rounds=args.max_rounds,
        randomize_roles=False,
    )

    while True:
        play_game(env, seed=args.seed)
        if args.seed is not None:
            args.seed += 1
        try:
            again = input("\nPlay again? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if again != "y":
            break


if __name__ == "__main__":
    main()
