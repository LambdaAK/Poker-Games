"""Train a small Hold'em policy league and keep the top performer.

This script trains multiple candidate bots (REINFORCE, CFR, NFSP) across
seeds, benchmarks them head-to-head, and copies the best non-random policy
to a stable output path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil

import evaluate_agents as bench
from holdem_limit import cfr as holdem_cfr
from holdem_limit import nfsp as holdem_nfsp
from holdem_limit import rl as holdem_rl


@dataclass
class TrainedArtifact:
    name: str
    algo: str
    seed: int
    path: Path
    eval_p0: float
    eval_p1: float
    train_log_last: dict[str, float] | None


def parse_int_list(csv_text: str) -> list[int]:
    values: list[int] = []
    for raw in csv_text.split(","):
        token = raw.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds for training candidates",
    )
    parser.add_argument(
        "--benchmark-seeds",
        type=str,
        default="11,12,13",
        help="Comma-separated seeds for round-robin benchmarking",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/holdem_league",
        help="Output directory for policies and report",
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Include random baseline in final benchmark table",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy policy action selection for benchmark/eval",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5000,
        help="Episodes for vs-random quality checks",
    )
    parser.add_argument(
        "--benchmark-episodes",
        type=int,
        default=3000,
        help="Episodes per matchup per benchmark seed",
    )
    parser.add_argument(
        "--elo-k",
        type=float,
        default=24.0,
        help="Elo K-factor for benchmark ranking",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="league_summary.json",
        help="Summary JSON filename in out-dir",
    )

    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip REINFORCE candidates",
    )
    parser.add_argument(
        "--skip-cfr",
        action="store_true",
        help="Skip CFR candidates",
    )
    parser.add_argument(
        "--skip-nfsp",
        action="store_true",
        help="Skip NFSP candidates",
    )

    parser.add_argument("--rl-episodes", type=int, default=140_000, help="REINFORCE episodes")
    parser.add_argument("--rl-lr", type=float, default=0.02, help="REINFORCE learning rate")
    parser.add_argument("--rl-baseline-lr", type=float, default=0.01, help="REINFORCE baseline lr")
    parser.add_argument("--rl-log-every", type=int, default=10_000, help="REINFORCE log frequency")

    parser.add_argument("--cfr-iterations", type=int, default=4_000, help="CFR iterations")
    parser.add_argument("--cfr-log-every", type=int, default=1_000, help="CFR log frequency")

    parser.add_argument("--nfsp-episodes", type=int, default=220_000, help="NFSP episodes")
    parser.add_argument("--nfsp-q-lr", type=float, default=0.08, help="NFSP Q-learning rate")
    parser.add_argument("--nfsp-eps-start", type=float, default=0.20, help="NFSP epsilon start")
    parser.add_argument("--nfsp-eps-end", type=float, default=0.02, help="NFSP epsilon end")
    parser.add_argument(
        "--nfsp-anticipatory",
        type=float,
        default=0.10,
        help="NFSP anticipatory parameter",
    )
    parser.add_argument("--nfsp-discount", type=float, default=1.0, help="NFSP discount factor")
    parser.add_argument(
        "--nfsp-reservoir-capacity",
        type=int,
        default=200_000,
        help="NFSP reservoir capacity",
    )
    parser.add_argument("--nfsp-log-every", type=int, default=10_000, help="NFSP log frequency")
    return parser.parse_args()


def train_rl_artifact(args: argparse.Namespace, seed: int, out_dir: Path) -> TrainedArtifact:
    policy, logs = holdem_rl.train_self_play(
        episodes=args.rl_episodes,
        learning_rate=args.rl_lr,
        baseline_lr=args.rl_baseline_lr,
        seed=seed,
        log_every=args.rl_log_every,
    )
    p0 = holdem_rl.evaluate_vs_random(
        policy,
        episodes=args.eval_episodes,
        seed=seed * 2 + 101,
        as_player=0,
        greedy=args.greedy,
    )
    p1 = holdem_rl.evaluate_vs_random(
        policy,
        episodes=args.eval_episodes,
        seed=seed * 2 + 102,
        as_player=1,
        greedy=args.greedy,
    )
    path = out_dir / f"holdem_rl_seed{seed}.json"
    policy.save(path)
    return TrainedArtifact(
        name=f"rl_s{seed}",
        algo="rl",
        seed=seed,
        path=path,
        eval_p0=p0,
        eval_p1=p1,
        train_log_last=logs[-1] if logs else None,
    )


def train_cfr_artifact(args: argparse.Namespace, seed: int, out_dir: Path) -> TrainedArtifact:
    trainer = holdem_cfr.CFRTrainer(seed=seed)
    logs = trainer.train(iterations=args.cfr_iterations, log_every=args.cfr_log_every)
    policy = trainer.average_policy()
    p0 = holdem_cfr.evaluate_vs_random(
        policy,
        episodes=args.eval_episodes,
        seed=seed * 2 + 201,
        as_player=0,
        greedy=args.greedy,
    )
    p1 = holdem_cfr.evaluate_vs_random(
        policy,
        episodes=args.eval_episodes,
        seed=seed * 2 + 202,
        as_player=1,
        greedy=args.greedy,
    )
    path = out_dir / f"holdem_cfr_seed{seed}.json"
    policy.save(path)
    return TrainedArtifact(
        name=f"cfr_s{seed}",
        algo="cfr",
        seed=seed,
        path=path,
        eval_p0=p0,
        eval_p1=p1,
        train_log_last=logs[-1] if logs else None,
    )


def train_nfsp_artifact(args: argparse.Namespace, seed: int, out_dir: Path) -> TrainedArtifact:
    policy, logs = holdem_nfsp.train_self_play(
        episodes=args.nfsp_episodes,
        q_learning_rate=args.nfsp_q_lr,
        epsilon_start=args.nfsp_eps_start,
        epsilon_end=args.nfsp_eps_end,
        anticipatory=args.nfsp_anticipatory,
        discount=args.nfsp_discount,
        reservoir_capacity=args.nfsp_reservoir_capacity,
        seed=seed,
        log_every=args.nfsp_log_every,
    )
    p0 = holdem_nfsp.evaluate_vs_random(
        policy,
        episodes=args.eval_episodes,
        seed=seed * 2 + 301,
        as_player=0,
        greedy=args.greedy,
    )
    p1 = holdem_nfsp.evaluate_vs_random(
        policy,
        episodes=args.eval_episodes,
        seed=seed * 2 + 302,
        as_player=1,
        greedy=args.greedy,
    )
    path = out_dir / f"holdem_nfsp_seed{seed}.json"
    policy.save(path)
    return TrainedArtifact(
        name=f"nfsp_s{seed}",
        algo="nfsp",
        seed=seed,
        path=path,
        eval_p0=p0,
        eval_p1=p1,
        train_log_last=logs[-1] if logs else None,
    )


def build_benchmark_agents(
    artifacts: list[TrainedArtifact],
    *,
    greedy: bool,
    include_random: bool,
) -> dict[str, bench.Agent]:
    agents: dict[str, bench.Agent] = {}
    if include_random:
        agents["random"] = bench.RandomAgent()
    for artifact in artifacts:
        if artifact.algo == "rl":
            policy = holdem_rl.TabularSoftmaxPolicy.load(artifact.path)
            agents[artifact.name] = bench.HoldemRlAgent(policy, greedy)
        elif artifact.algo == "cfr":
            policy = holdem_cfr.AverageStrategyPolicy.load(artifact.path)
            agents[artifact.name] = bench.HoldemCfrAgent(policy, greedy)
        elif artifact.algo == "nfsp":
            policy = holdem_nfsp.NfspAveragePolicy.load(artifact.path)
            agents[artifact.name] = bench.HoldemNfspAgent(policy, greedy)
        else:
            raise ValueError(f"Unknown artifact algo: {artifact.algo}")
    return agents


def select_best_artifact(
    artifacts: list[TrainedArtifact], benchmark: dict[str, object]
) -> TrainedArtifact:
    by_name = {artifact.name: artifact for artifact in artifacts}
    leaderboard = benchmark.get("leaderboard", [])
    if not isinstance(leaderboard, list):
        raise ValueError("Benchmark result missing leaderboard.")
    for row in leaderboard:
        if not isinstance(row, dict):
            continue
        name = row.get("agent")
        if isinstance(name, str) and name in by_name:
            return by_name[name]
    raise RuntimeError("Could not find a trained agent in leaderboard.")


def artifact_row(artifact: TrainedArtifact) -> dict[str, object]:
    return {
        "name": artifact.name,
        "algo": artifact.algo,
        "seed": artifact.seed,
        "path": str(artifact.path),
        "eval_vs_random_p0": artifact.eval_p0,
        "eval_vs_random_p1": artifact.eval_p1,
        "train_log_last": artifact.train_log_last,
    }


def play_flag_for_algo(algo: str) -> str:
    if algo == "rl":
        return "--rl-policy"
    if algo == "cfr":
        return "--cfr-policy"
    if algo == "nfsp":
        return "--nfsp-policy"
    raise ValueError(f"Unknown algo {algo!r}")


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    benchmark_seeds = parse_int_list(args.benchmark_seeds)
    if args.skip_rl and args.skip_cfr and args.skip_nfsp:
        raise ValueError("At least one training family must be enabled.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[TrainedArtifact] = []

    for seed in seeds:
        if not args.skip_rl:
            print(f"[train] RL seed={seed}")
            artifacts.append(train_rl_artifact(args, seed, out_dir))
        if not args.skip_cfr:
            print(f"[train] CFR seed={seed}")
            artifacts.append(train_cfr_artifact(args, seed, out_dir))
        if not args.skip_nfsp:
            print(f"[train] NFSP seed={seed}")
            artifacts.append(train_nfsp_artifact(args, seed, out_dir))

    print("\nCandidate quality vs random:")
    for artifact in artifacts:
        print(
            "  "
            f"{artifact.name}: "
            f"p0={artifact.eval_p0:+.4f} chips/hand, "
            f"p1={artifact.eval_p1:+.4f} chips/hand"
        )

    agents = build_benchmark_agents(
        artifacts,
        greedy=args.greedy,
        include_random=args.include_random,
    )
    if len(agents) < 2:
        raise ValueError("Need at least two agents for benchmarking.")

    holdem_spec = bench.build_game_specs()["holdem"]
    benchmark = bench.evaluate_game(
        spec=holdem_spec,
        agents=agents,
        seeds=benchmark_seeds,
        episodes=args.benchmark_episodes,
        elo_k=args.elo_k,
    )

    print("\nLeague benchmark:")
    bench.print_game_report(benchmark)

    best = select_best_artifact(artifacts, benchmark)
    best_copy = out_dir / "best_holdem_policy.json"
    shutil.copyfile(best.path, best_copy)
    best_meta = out_dir / "best_holdem_policy.meta.json"
    best_flag = play_flag_for_algo(best.algo)
    best_meta_payload = {
        "name": best.name,
        "algo": best.algo,
        "seed": best.seed,
        "policy_path": str(best_copy),
        "play_flag": best_flag,
        "play_command": f"python3 play_holdem.py {best_flag} {best_copy}",
    }
    best_meta.write_text(json.dumps(best_meta_payload, indent=2), encoding="utf-8")

    summary = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "seeds": seeds,
            "benchmark_seeds": benchmark_seeds,
            "include_random": args.include_random,
            "greedy": args.greedy,
            "eval_episodes": args.eval_episodes,
            "benchmark_episodes": args.benchmark_episodes,
            "elo_k": args.elo_k,
        },
        "artifacts": [artifact_row(artifact) for artifact in artifacts],
        "benchmark": benchmark,
        "best": {
            "name": best.name,
            "algo": best.algo,
            "seed": best.seed,
            "source_path": str(best.path),
            "copied_path": str(best_copy),
            "play_flag": best_flag,
            "meta_path": str(best_meta),
        },
    }
    summary_path = out_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest model:")
    print(f"  {best.name} ({best.algo}, seed={best.seed})")
    print(f"  copied to: {best_copy}")
    print(f"  play cmd:  python3 play_holdem.py {best_flag} {best_copy}")
    print(f"  meta:      {best_meta}")
    print(f"  summary:   {summary_path}")


if __name__ == "__main__":
    main()
