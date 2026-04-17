"""Train a tabular NFSP-style policy for heads-up limit Hold'em."""

from __future__ import annotations

import argparse

from holdem_limit.nfsp import evaluate_vs_random, policy_table, train_self_play


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=350_000, help="Self-play episodes")
    parser.add_argument(
        "--q-lr",
        type=float,
        default=0.08,
        help="Best-response Q-learning rate",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=0.20,
        help="Starting epsilon for BR exploration",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.02,
        help="Final epsilon for BR exploration",
    )
    parser.add_argument(
        "--anticipatory",
        type=float,
        default=0.10,
        help="Probability each player uses best response in a hand",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=1.0,
        help="Discount factor for TD targets",
    )
    parser.add_argument(
        "--reservoir-capacity",
        type=int,
        default=200_000,
        help="Reservoir size for average-policy action memory",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log-every", type=int, default=10_000, help="Log every N episodes")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20_000,
        help="Evaluation hands vs random",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="models/holdem_limit_nfsp_policy.json",
        help="Output policy JSON path",
    )
    parser.add_argument("--print-rows", type=int, default=30, help="Max rows to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy, logs = train_self_play(
        episodes=args.episodes,
        q_learning_rate=args.q_lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        anticipatory=args.anticipatory,
        discount=args.discount,
        reservoir_capacity=args.reservoir_capacity,
        seed=args.seed,
        log_every=args.log_every,
    )

    if logs:
        print("Training progress:")
        for row in logs:
            print(
                f"  ep={int(row['episode'])} "
                f"avg_p0_reward={row['avg_p0_reward']:+.4f} "
                f"eps={row['epsilon']:.3f} "
                f"q_states={int(row['q_states'])} "
                f"avg_states={int(row['avg_states'])} "
                f"reservoir={int(row['reservoir_size'])}"
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

    rows = policy_table(policy)
    print(f"\nLearned policy sample ({min(len(rows), args.print_rows)} rows):")
    for key, probs in rows[: args.print_rows]:
        pretty = ", ".join(f"{action}={prob:.3f}" for action, prob in probs.items())
        print(f"  {key} -> {pretty}")


if __name__ == "__main__":
    main()
