"""Benchmark poker agents across Kuhn, Leduc, and limit Hold'em."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Callable

import argparse

from kuhn_poker import KuhnPokerGame as KuhnGame
from kuhn_poker import rl as kuhn_rl
from leduc_poker import LeducPokerGame as LeducGame
from leduc_poker import cfr as leduc_cfr
from leduc_poker import rl as leduc_rl
from holdem_limit import HoldemLimitGame as HoldemGame
from holdem_limit import abstraction as holdem_abstraction
from holdem_limit import cfr as holdem_cfr
from holdem_limit import nfsp as holdem_nfsp
from holdem_limit import rl as holdem_rl


@dataclass
class HandStats:
    chips_sum: float = 0.0
    hands: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0

    def add_reward(self, reward: float) -> None:
        self.chips_sum += reward
        self.hands += 1
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.ties += 1

    def score_rate(self) -> float:
        if self.hands == 0:
            return 0.5
        return (self.wins + 0.5 * self.ties) / self.hands


class Agent:
    """Policy wrapper interface."""

    def choose(
        self,
        *,
        game: Any,
        state: Any,
        legal_actions: tuple[Any, ...],
        player_index: int,
        rng: random.Random,
    ) -> Any:
        raise NotImplementedError


class RandomAgent(Agent):
    def choose(
        self,
        *,
        game: Any,
        state: Any,
        legal_actions: tuple[Any, ...],
        player_index: int,
        rng: random.Random,
    ) -> Any:
        return rng.choice(list(legal_actions))


class KuhnRlAgent(Agent):
    def __init__(self, policy: kuhn_rl.TabularSoftmaxPolicy, greedy: bool) -> None:
        self._policy = policy
        self._greedy = greedy

    def choose(
        self,
        *,
        game: KuhnGame,
        state: Any,
        legal_actions: tuple[Any, ...],
        player_index: int,
        rng: random.Random,
    ) -> Any:
        key = kuhn_rl.info_state_key(state.player_cards[player_index], game.history_label())
        if self._greedy:
            return self._policy.greedy_action(key, legal_actions)
        return self._policy.sample_action(key, legal_actions, rng)


class LeducRlAgent(Agent):
    def __init__(self, policy: leduc_rl.TabularSoftmaxPolicy, greedy: bool) -> None:
        self._policy = policy
        self._greedy = greedy

    def choose(
        self,
        *,
        game: LeducGame,
        state: Any,
        legal_actions: tuple[Any, ...],
        player_index: int,
        rng: random.Random,
    ) -> Any:
        key = leduc_rl.info_state_key(state, player_index)
        if self._greedy:
            return self._policy.greedy_action(key, legal_actions)
        return self._policy.sample_action(key, legal_actions, rng)


class LeducCfrAgent(Agent):
    def __init__(self, policy: leduc_cfr.AverageStrategyPolicy, greedy: bool) -> None:
        self._policy = policy
        self._greedy = greedy

    def choose(
        self,
        *,
        game: LeducGame,
        state: Any,
        legal_actions: tuple[Any, ...],
        player_index: int,
        rng: random.Random,
    ) -> Any:
        key = leduc_cfr.info_state_key(state, player_index)
        if self._greedy:
            return self._policy.greedy_action(key, legal_actions)
        return self._policy.sample_action(key, legal_actions, rng)


class HoldemRlAgent(Agent):
    def __init__(self, policy: holdem_rl.TabularSoftmaxPolicy, greedy: bool) -> None:
        self._policy = policy
        self._greedy = greedy

    def choose(
        self,
        *,
        game: HoldemGame,
        state: Any,
        legal_actions: tuple[Any, ...],
        player_index: int,
        rng: random.Random,
    ) -> Any:
        key = holdem_abstraction.info_state_key(state, player_index)
        if self._greedy:
            return self._policy.greedy_action(key, legal_actions)
        return self._policy.sample_action(key, legal_actions, rng)


class HoldemCfrAgent(Agent):
    def __init__(self, policy: holdem_cfr.AverageStrategyPolicy, greedy: bool) -> None:
        self._policy = policy
        self._greedy = greedy

    def choose(
        self,
        *,
        game: HoldemGame,
        state: Any,
        legal_actions: tuple[Any, ...],
        player_index: int,
        rng: random.Random,
    ) -> Any:
        key = holdem_abstraction.info_state_key(state, player_index)
        if self._greedy:
            return self._policy.greedy_action(key, legal_actions)
        return self._policy.sample_action(key, legal_actions, rng)


class HoldemNfspAgent(Agent):
    def __init__(self, policy: holdem_nfsp.NfspAveragePolicy, greedy: bool) -> None:
        self._policy = policy
        self._greedy = greedy

    def choose(
        self,
        *,
        game: HoldemGame,
        state: Any,
        legal_actions: tuple[Any, ...],
        player_index: int,
        rng: random.Random,
    ) -> Any:
        key = holdem_abstraction.info_state_key(state, player_index)
        if self._greedy:
            return self._policy.greedy_action(key, legal_actions)
        return self._policy.sample_action(key, legal_actions, rng)


@dataclass
class GameSpec:
    key: str
    label: str
    big_blind: float
    make_game: Callable[[random.Random], Any]
    build_agents: Callable[[argparse.Namespace, list[str]], dict[str, Agent]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games",
        type=str,
        default="kuhn,leduc,holdem",
        help="Comma-separated games: kuhn,leduc,holdem",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated integer seeds",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Hands per matchup per seed (split equally across seats)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy action selection for loaded policies",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if a requested policy file is missing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON path (default: results/benchmark_<timestamp>.json)",
    )
    parser.add_argument(
        "--elo-k",
        type=float,
        default=24.0,
        help="Elo K-factor for pairwise updates",
    )

    parser.add_argument(
        "--kuhn-rl-policy",
        type=str,
        default="models/kuhn_reinforce_policy.json",
        help="Kuhn RL policy JSON path",
    )

    parser.add_argument(
        "--leduc-rl-policy",
        type=str,
        default="models/leduc_reinforce_policy.json",
        help="Leduc RL policy JSON path",
    )
    parser.add_argument(
        "--leduc-cfr-policy",
        type=str,
        default="models/leduc_cfr_policy.json",
        help="Leduc CFR policy JSON path",
    )

    parser.add_argument(
        "--holdem-rl-policy",
        type=str,
        default="models/holdem_limit_reinforce_policy.json",
        help="Hold'em RL policy JSON path",
    )
    parser.add_argument(
        "--holdem-cfr-policy",
        type=str,
        default="models/holdem_limit_cfr_policy.json",
        help="Hold'em CFR policy JSON path",
    )
    parser.add_argument(
        "--holdem-nfsp-policy",
        type=str,
        default="models/holdem_limit_nfsp_policy.json",
        help="Hold'em NFSP policy JSON path",
    )
    return parser.parse_args()


def _load_or_warn(
    *,
    path_text: str,
    strict_missing: bool,
    warnings: list[str],
    label: str,
) -> Path | None:
    path = Path(path_text)
    if path.exists():
        return path
    msg = f"Missing {label}: {path}"
    if strict_missing:
        raise FileNotFoundError(msg)
    warnings.append(msg)
    return None


def build_kuhn_agents(args: argparse.Namespace, warnings: list[str]) -> dict[str, Agent]:
    agents: dict[str, Agent] = {"random": RandomAgent()}
    rl_path = _load_or_warn(
        path_text=args.kuhn_rl_policy,
        strict_missing=args.strict_missing,
        warnings=warnings,
        label="Kuhn RL policy",
    )
    if rl_path is not None:
        agents["rl"] = KuhnRlAgent(kuhn_rl.TabularSoftmaxPolicy.load(rl_path), args.greedy)
    return agents


def build_leduc_agents(args: argparse.Namespace, warnings: list[str]) -> dict[str, Agent]:
    agents: dict[str, Agent] = {"random": RandomAgent()}

    rl_path = _load_or_warn(
        path_text=args.leduc_rl_policy,
        strict_missing=args.strict_missing,
        warnings=warnings,
        label="Leduc RL policy",
    )
    if rl_path is not None:
        agents["rl"] = LeducRlAgent(leduc_rl.TabularSoftmaxPolicy.load(rl_path), args.greedy)

    cfr_path = _load_or_warn(
        path_text=args.leduc_cfr_policy,
        strict_missing=args.strict_missing,
        warnings=warnings,
        label="Leduc CFR policy",
    )
    if cfr_path is not None:
        agents["cfr"] = LeducCfrAgent(leduc_cfr.AverageStrategyPolicy.load(cfr_path), args.greedy)
    return agents


def build_holdem_agents(args: argparse.Namespace, warnings: list[str]) -> dict[str, Agent]:
    agents: dict[str, Agent] = {"random": RandomAgent()}

    rl_path = _load_or_warn(
        path_text=args.holdem_rl_policy,
        strict_missing=args.strict_missing,
        warnings=warnings,
        label="Hold'em RL policy",
    )
    if rl_path is not None:
        agents["rl"] = HoldemRlAgent(holdem_rl.TabularSoftmaxPolicy.load(rl_path), args.greedy)

    cfr_path = _load_or_warn(
        path_text=args.holdem_cfr_policy,
        strict_missing=args.strict_missing,
        warnings=warnings,
        label="Hold'em CFR policy",
    )
    if cfr_path is not None:
        agents["cfr"] = HoldemCfrAgent(holdem_cfr.AverageStrategyPolicy.load(cfr_path), args.greedy)

    nfsp_path = _load_or_warn(
        path_text=args.holdem_nfsp_policy,
        strict_missing=args.strict_missing,
        warnings=warnings,
        label="Hold'em NFSP policy",
    )
    if nfsp_path is not None:
        agents["nfsp"] = HoldemNfspAgent(holdem_nfsp.NfspAveragePolicy.load(nfsp_path), args.greedy)
    return agents


def build_game_specs() -> dict[str, GameSpec]:
    return {
        "kuhn": GameSpec(
            key="kuhn",
            label="Kuhn Poker",
            big_blind=1.0,
            make_game=lambda rng: KuhnGame(rng=rng),
            build_agents=build_kuhn_agents,
        ),
        "leduc": GameSpec(
            key="leduc",
            label="Leduc Poker",
            big_blind=1.0,
            make_game=lambda rng: LeducGame(rng=rng),
            build_agents=build_leduc_agents,
        ),
        "holdem": GameSpec(
            key="holdem",
            label="Heads-up Limit Hold'em",
            big_blind=2.0,
            make_game=lambda rng: HoldemGame(rng=rng),
            build_agents=build_holdem_agents,
        ),
    }


def parse_seeds(seed_text: str) -> list[int]:
    values: list[int] = []
    for raw in seed_text.split(","):
        stripped = raw.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def run_match(
    *,
    spec: GameSpec,
    agent_a: Agent,
    agent_b: Agent,
    episodes: int,
    seed: int,
) -> HandStats:
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    rng = random.Random(seed)
    game = spec.make_game(rng)
    stats = HandStats()

    first_count = episodes // 2
    second_count = episodes - first_count
    seat_configs = ((0, first_count), (1, second_count))

    for a_player_index, count in seat_configs:
        for _ in range(count):
            state = game.reset()
            while True:
                legal = game.legal_actions()
                current = state.current_player
                if current == a_player_index:
                    action = agent_a.choose(
                        game=game,
                        state=state,
                        legal_actions=legal,
                        player_index=current,
                        rng=rng,
                    )
                else:
                    action = agent_b.choose(
                        game=game,
                        state=state,
                        legal_actions=legal,
                        player_index=current,
                        rng=rng,
                    )

                state, rewards, done = game.step(action)
                if done:
                    stats.add_reward(float(rewards[a_player_index]))
                    break
    return stats


def mean_and_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    mean = statistics.fmean(values)
    if len(values) < 2:
        return (mean, 0.0)
    stdev = statistics.stdev(values)
    ci95 = 1.96 * stdev / math.sqrt(len(values))
    return (mean, ci95)


def apply_elo_update(ratings: dict[str, float], a: str, b: str, score_a: float, k: float) -> None:
    ra = ratings[a]
    rb = ratings[b]
    expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    expected_b = 1.0 - expected_a
    ratings[a] = ra + k * (score_a - expected_a)
    ratings[b] = rb + k * ((1.0 - score_a) - expected_b)


def evaluate_game(
    *,
    spec: GameSpec,
    agents: dict[str, Agent],
    seeds: list[int],
    episodes: int,
    elo_k: float,
) -> dict[str, Any]:
    names = sorted(agents.keys())
    if len(names) < 2:
        return {
            "game": spec.key,
            "label": spec.label,
            "big_blind": spec.big_blind,
            "agents": names,
            "matches": [],
            "leaderboard": [],
            "elo": [],
            "skipped": True,
            "reason": "Need at least two agents for a matchup.",
        }

    match_rows: list[dict[str, Any]] = []
    by_agent: dict[str, list[float]] = {name: [] for name in names}
    ratings: dict[str, float] = {name: 1500.0 for name in names}

    pair_index = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name = names[i]
            b_name = names[j]
            pair_index += 1

            seed_means: list[float] = []
            combined = HandStats()
            for seed in seeds:
                match_seed = (
                    seed * 1_000_003 + pair_index * 97_013 + len(spec.key) * 389
                ) % (2**32)
                stats = run_match(
                    spec=spec,
                    agent_a=agents[a_name],
                    agent_b=agents[b_name],
                    episodes=episodes,
                    seed=match_seed,
                )
                seed_means.append(stats.chips_sum / stats.hands)
                combined.chips_sum += stats.chips_sum
                combined.hands += stats.hands
                combined.wins += stats.wins
                combined.losses += stats.losses
                combined.ties += stats.ties

            mean_chips, ci95 = mean_and_ci95(seed_means)
            bb100 = (mean_chips / spec.big_blind) * 100.0
            score_rate = combined.score_rate()
            apply_elo_update(ratings, a_name, b_name, score_rate, elo_k)

            row = {
                "agent_a": a_name,
                "agent_b": b_name,
                "episodes_per_seed": episodes,
                "seeds": seeds,
                "mean_chips_per_hand": mean_chips,
                "ci95_chips_per_hand": ci95,
                "bb_per_100": bb100,
                "hands": combined.hands,
                "wins_a": combined.wins,
                "losses_a": combined.losses,
                "ties": combined.ties,
                "score_rate_a": score_rate,
            }
            match_rows.append(row)
            by_agent[a_name].append(mean_chips)
            by_agent[b_name].append(-mean_chips)

    leaderboard = []
    for name in names:
        values = by_agent[name]
        mean = statistics.fmean(values) if values else 0.0
        leaderboard.append(
            {
                "agent": name,
                "avg_chips_per_hand_vs_pool": mean,
                "avg_bb_per_100_vs_pool": (mean / spec.big_blind) * 100.0,
                "opponents": len(values),
            }
        )
    leaderboard.sort(key=lambda row: row["avg_chips_per_hand_vs_pool"], reverse=True)

    elo_table = [
        {"agent": name, "elo": ratings[name]}
        for name in sorted(ratings.keys(), key=lambda key: ratings[key], reverse=True)
    ]

    return {
        "game": spec.key,
        "label": spec.label,
        "big_blind": spec.big_blind,
        "agents": names,
        "matches": match_rows,
        "leaderboard": leaderboard,
        "elo": elo_table,
        "skipped": False,
    }


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"benchmark_{stamp}.json"


def print_game_report(game_result: dict[str, Any]) -> None:
    print(f"\n== {game_result['label']} ==")
    if game_result.get("skipped"):
        print(f"Skipped: {game_result.get('reason', 'n/a')}")
        return

    print("Matchups:")
    for row in game_result["matches"]:
        print(
            "  "
            f"{row['agent_a']} vs {row['agent_b']}: "
            f"{row['mean_chips_per_hand']:+.4f} chips/hand "
            f"(CI95 ±{row['ci95_chips_per_hand']:.4f}), "
            f"{row['bb_per_100']:+.2f} bb/100"
        )

    print("Leaderboard (avg vs pool):")
    for row in game_result["leaderboard"]:
        print(
            "  "
            f"{row['agent']}: "
            f"{row['avg_chips_per_hand_vs_pool']:+.4f} chips/hand, "
            f"{row['avg_bb_per_100_vs_pool']:+.2f} bb/100"
        )

    print("Elo:")
    for row in game_result["elo"]:
        print(f"  {row['agent']}: {row['elo']:.1f}")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    specs = build_game_specs()

    requested = [name.strip().lower() for name in args.games.split(",") if name.strip()]
    if not requested:
        raise ValueError("No games selected.")

    unknown = [name for name in requested if name not in specs]
    if unknown:
        raise ValueError(f"Unknown games: {unknown}. Valid: {sorted(specs)}")

    all_results: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "games": requested,
            "seeds": seeds,
            "episodes": args.episodes,
            "greedy": args.greedy,
            "elo_k": args.elo_k,
            "strict_missing": args.strict_missing,
        },
        "warnings": [],
        "games": {},
    }

    for key in requested:
        spec = specs[key]
        local_warnings: list[str] = []
        agents = spec.build_agents(args, local_warnings)
        if local_warnings:
            all_results["warnings"].extend([f"{spec.key}: {text}" for text in local_warnings])

        result = evaluate_game(
            spec=spec,
            agents=agents,
            seeds=seeds,
            episodes=args.episodes,
            elo_k=args.elo_k,
        )
        all_results["games"][key] = result
        print_game_report(result)

    if all_results["warnings"]:
        print("\nWarnings:")
        for warning in all_results["warnings"]:
            print(f"  - {warning}")

    output = Path(args.output) if args.output else default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark report: {output}")


if __name__ == "__main__":
    main()
