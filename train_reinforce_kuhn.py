"""Train a tabular REINFORCE policy for Kuhn Poker."""

from __future__ import annotations

import argparse

from kuhn_poker import evaluate_vs_random, train_self_play
from kuhn_poker.rl import policy_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=150_000, help="Self-play episodes")
    parser.add_argument("--lr", type=float, default=0.05, help="Policy learning rate")
    parser.add_argument(
        "--baseline-lr",
        type=float,
        default=0.01,
        help="Moving-baseline learning rate",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10_000,
        help="Log metrics every N episodes (0 disables)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20_000,
        help="Evaluation hands vs random policy",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="models/kuhn_reinforce_policy.json",
        help="Path for learned policy JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy, logs = train_self_play(
        episodes=args.episodes,
        learning_rate=args.lr,
        baseline_lr=args.baseline_lr,
        seed=args.seed,
        log_every=args.log_every,
    )

    if logs:
        print("Training progress:")
        for row in logs:
            print(
                f"  ep={int(row['episode'])} "
                f"avg_p0_reward={row['avg_p0_reward']:+.4f} "
                f"baseline_p0={row['baseline_p0']:+.4f} "
                f"baseline_p1={row['baseline_p1']:+.4f}"
            )

    avg_p0 = evaluate_vs_random(
        policy, episodes=args.eval_episodes, seed=args.seed + 1, as_player=0
    )
    avg_p1 = evaluate_vs_random(
        policy, episodes=args.eval_episodes, seed=args.seed + 2, as_player=1
    )
    print("\nEvaluation vs random:")
    print(f"  as player 0: {avg_p0:+.4f} chips/hand")
    print(f"  as player 1: {avg_p1:+.4f} chips/hand")

    policy.save(args.save)
    print(f"\nSaved policy to: {args.save}")

    print("\nLearned policy (stochastic action probs):")
    for info_state, probs in policy_table(policy):
        pretty = ", ".join(f"{action}={prob:.3f}" for action, prob in probs.items())
        print(f"  {info_state:>5} -> {pretty}")


if __name__ == "__main__":
    main()
