"""Train tabular blackjack policies."""

from __future__ import annotations

import argparse

from blackjack import (
    BasicStrategyPolicy,
    RandomPolicy,
    TabularActionValuePolicy,
    evaluate_policy,
    policy_table,
    train_monte_carlo_control,
    train_q_learning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algo",
        type=str,
        choices=("mc", "q"),
        default="mc",
        help="Training algorithm",
    )
    parser.add_argument("--episodes", type=int, default=200_000, help="Training episodes")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Q-learning step size (Q-learning only)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor (Q-learning only)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10_000,
        help="Log every N episodes (0 disables)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20_000,
        help="Evaluation hands per policy",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Output JSON path for the learned policy",
    )
    parser.add_argument(
        "--print-rows",
        type=int,
        default=40,
        help="Max learned rows to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.algo == "mc":
        policy, logs = train_monte_carlo_control(
            episodes=args.episodes,
            epsilon=args.epsilon,
            seed=args.seed,
            log_every=args.log_every,
        )
        default_save = "models/blackjack_mc_policy.json"
    else:
        policy, logs = train_q_learning(
            episodes=args.episodes,
            epsilon=args.epsilon,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            seed=args.seed,
            log_every=args.log_every,
        )
        default_save = "models/blackjack_q_policy.json"

    if logs:
        print("Training progress:")
        for row in logs:
            print(
                f"  ep={int(row['episode'])} "
                f"avg_reward={row['avg_reward']:+.4f}"
            )

    learned_score = evaluate_policy(
        policy,
        episodes=args.eval_episodes,
        seed=args.seed + 1,
        greedy=True,
    )
    basic_score = evaluate_policy(
        BasicStrategyPolicy(),
        episodes=args.eval_episodes,
        seed=args.seed + 2,
        greedy=True,
    )
    random_score = evaluate_policy(
        RandomPolicy(),
        episodes=args.eval_episodes,
        seed=args.seed + 3,
        greedy=True,
    )
    print("\nEvaluation:")
    print(f"  learned policy: {learned_score:+.4f} chips/hand")
    print(f"  basic strategy: {basic_score:+.4f} chips/hand")
    print(f"  random policy:  {random_score:+.4f} chips/hand")

    save_path = args.save or default_save
    policy.save(save_path)
    print(f"\nSaved policy to: {save_path}")

    rows = policy_table(policy)
    if rows:
        print(f"\nLearned policy table sample ({min(len(rows), args.print_rows)} rows):")
        for key, values in rows[: args.print_rows]:
            pretty = ", ".join(f"{action}={value:+.3f}" for action, value in values.items())
            print(f"  {key} -> {pretty}")


if __name__ == "__main__":
    main()
