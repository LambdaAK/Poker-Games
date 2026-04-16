"""Train a CFR baseline strategy for Leduc poker."""

from __future__ import annotations

import argparse

from leduc_poker.cfr import CFRTrainer, evaluate_vs_random, policy_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=4_000, help="CFR iterations")
    parser.add_argument(
        "--log-every",
        type=int,
        default=400,
        help="Log every N iterations (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=30_000,
        help="Evaluation hands vs random",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="models/leduc_cfr_policy.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--print-rows",
        type=int,
        default=50,
        help="Max policy rows to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = CFRTrainer()
    logs = trainer.train(iterations=args.iterations, log_every=args.log_every)
    policy = trainer.average_policy()

    if logs:
        print("Training progress:")
        for row in logs:
            print(
                f"  iter={int(row['iteration'])} "
                f"avg_p0_value={row['avg_p0_value']:+.4f} "
                f"infosets={int(row['infosets'])}"
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
    print(f"\nAverage strategy sample ({min(len(rows), args.print_rows)} rows):")
    for key, probs in rows[: args.print_rows]:
        pretty = ", ".join(f"{action}={prob:.3f}" for action, prob in probs.items())
        print(f"  {key} -> {pretty}")


if __name__ == "__main__":
    main()
