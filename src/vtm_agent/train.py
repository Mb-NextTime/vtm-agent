import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gymnasium import Env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from vtm_agent.env import VTMCombatEnv


def make_env(seed: int, rank: int) -> Callable[[], Env[Any, Any]]:
    def _init() -> Env[Any, Any]:
        e = VTMCombatEnv(randomize_roles=True)
        e.reset(seed=seed + rank * 1000)
        return ActionMasker(e, "action_mask")

    return _init


def train(args: argparse.Namespace) -> None:
    env = VecMonitor(SubprocVecEnv([make_env(args.seed, i) for i in range(args.n_envs)]))
    env.seed(args.seed)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        seed=args.seed,
        device="auto",
        tensorboard_log=str(args.checkpoint_dir / "tensorboard"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(args.checkpoint_dir),
        name_prefix="vtm_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_path = args.checkpoint_dir / "vtm_ppo_final.zip"
    model.save(str(final_path))
    print(f"Model saved to {final_path}")

    env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskablePPO agent for VTM combat")

    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint save frequency (steps)")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per env per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")

    args = parser.parse_args(argv)
    if args.n_envs < 1:
        parser.error("--n-envs must be >= 1")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
