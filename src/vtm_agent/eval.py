import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from vtm_agent.env import CharacterConfig, Opponent, RandomOpponent, ScriptedOpponent, VTMCombatEnv

AgentFn = Callable[[np.ndarray, np.ndarray], int]


@dataclass
class MatchupStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)


@dataclass
class EvalResults:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    per_config: dict[str, MatchupStats] = field(default_factory=dict)


@dataclass
class MatchupConfig:
    agent: CharacterConfig
    opponent: CharacterConfig
    label: str


DEFAULT_MATCHUPS: list[MatchupConfig] = [
    MatchupConfig(CharacterConfig(is_vampire=False), CharacterConfig(is_vampire=True), "HvV"),
    MatchupConfig(CharacterConfig(is_vampire=True), CharacterConfig(is_vampire=True), "VvV"),
    MatchupConfig(CharacterConfig(is_vampire=False), CharacterConfig(is_vampire=False), "HvH"),
    MatchupConfig(CharacterConfig(is_vampire=True), CharacterConfig(is_vampire=False), "VvH"),
]


def create_agent_fn(agent_type: str, model_path: str | None) -> AgentFn:
    if agent_type == "random":

        def _random_agent(obs: np.ndarray, mask: np.ndarray) -> int:
            valid = np.where(mask == 1)[0]
            return int(np.random.choice(valid))

        return _random_agent

    if agent_type == "scripted":

        def _scripted_agent(obs: np.ndarray, mask: np.ndarray) -> int:
            if mask[0] == 1:
                return 0
            return 2

        return _scripted_agent

    if model_path is None:
        raise ValueError("model_path is required when agent_type='rl'")
    model = MaskablePPO.load(model_path)

    def _rl_agent(obs: np.ndarray, mask: np.ndarray) -> int:
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    return _rl_agent


def evaluate(
    agent_fn: AgentFn,
    episodes: int,
    seed: int,
    max_rounds: int = 50,
    opponent: Opponent = ScriptedOpponent(),
    matchups: Sequence[MatchupConfig] = DEFAULT_MATCHUPS,
) -> EvalResults:
    results = EvalResults(
        per_config={m.label: MatchupStats() for m in matchups},
    )

    envs = {
        m.label: VTMCombatEnv(
            opponent=opponent,
            agent_config=m.agent,
            opponent_config=m.opponent,
            max_rounds=max_rounds,
            randomize_roles=False,
        )
        for m in matchups
    }

    try:
        for ep in range(episodes):
            m = matchups[ep % len(matchups)]
            env = envs[m.label]

            obs, info = env.reset(seed=seed + ep)
            done = False
            truncated = False
            ep_reward = 0.0
            ep_length = 0

            while not done and not truncated:
                mask = info.get("action_mask")
                assert isinstance(mask, np.ndarray)
                action = agent_fn(obs, mask)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1

            results.rewards.append(ep_reward)
            results.lengths.append(ep_length)

            cfg = results.per_config[m.label]
            cfg.rewards.append(ep_reward)
            cfg.lengths.append(ep_length)

            if env.opponent_defeated:
                results.wins += 1
                cfg.wins += 1
            elif env.agent_defeated:
                results.losses += 1
                cfg.losses += 1
            else:
                results.draws += 1
                cfg.draws += 1
    finally:
        for e in envs.values():
            e.close()

    return results


def _fmt_config_table(per_config: dict[str, MatchupStats]) -> str:
    lines = ["  Config    Episodes    Wins      Losses    Draws    Win rate    Avg reward    Avg length"]
    lines.append("  " + "-" * 95)
    for label in ("HvV", "VvV", "HvH", "VvH"):
        s = per_config[label]
        n = s.wins + s.losses + s.draws
        if n == 0:
            continue
        wr = s.wins / n * 100
        avg_r = float(np.mean(s.rewards)) if s.rewards else 0.0
        avg_l = float(np.mean(s.lengths)) if s.lengths else 0.0
        lines.append(
            f"  {label:<9} {n:<10} {s.wins:<10} {s.losses:<10} {s.draws:<8} "
            f"{wr:>6.1f}%    {avg_r:>8.4f}    {avg_l:>8.2f}"
        )
    return "\n".join(lines)


def print_results(results: EvalResults, agent_name: str, opponent_name: str) -> None:
    total: int = results.wins + results.losses + results.draws
    if total == 0:
        return

    wr = results.wins / total * 100
    avg_r = float(np.mean(results.rewards))
    avg_l = float(np.mean(results.lengths))

    print(f"Agent: {agent_name:<8} | Opponent: {opponent_name}")
    print("-" * 60)
    print(f"  Total episodes:     {total}")
    print(f"  Wins:               {results.wins:<5} ({wr:.1f}%)")
    print(f"  Losses:             {results.losses:<5} ({results.losses / total * 100:.1f}%)")
    print(f"  Draws:              {results.draws:<5} ({results.draws / total * 100:.1f}%)")
    print(f"  Win rate:           {wr:.1f}%")
    print(f"  Average reward:     {avg_r:.4f}")
    print(f"  Average length:     {avg_l:.2f}")
    print()
    print("Per match-up:")
    print(_fmt_config_table(results.per_config))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained VTM agent against various opponents")
    parser.add_argument("-a", "--agent", choices=["random", "scripted", "rl"], default="rl")
    parser.add_argument("-o", "--opponent", choices=["random", "scripted"], default="scripted")
    parser.add_argument("-m", "--model-path", default="checkpoints/vtm_ppo_final.zip")
    parser.add_argument("-e", "--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    model_path: str | None = str(Path(args.model_path)) if args.agent == "rl" else None
    opponent: Opponent = ScriptedOpponent() if args.opponent == "scripted" else RandomOpponent()
    agent_fn = create_agent_fn(args.agent, model_path)

    results = evaluate(agent_fn, args.episodes, args.seed, args.max_rounds, opponent)

    print_results(results, args.agent.upper(), args.opponent.upper())


if __name__ == "__main__":
    main(sys.argv[1:])
