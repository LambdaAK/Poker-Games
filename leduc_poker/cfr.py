"""Counterfactual Regret Minimization (CFR) for Leduc poker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

from .game import (
    Action,
    LeducPokerGame,
    LeducPokerState,
    all_deals,
    card_rank,
    history_label,
    initial_state,
    legal_actions_for_state,
    rank_symbol,
    step_state,
    terminal_rewards,
)


def info_state_key(state: LeducPokerState, player: int) -> str:
    """Information-set key based on private rank and public state."""
    private_rank = rank_symbol(card_rank(state.player_cards[player]))
    board_rank = "?" if state.board_card is None else rank_symbol(card_rank(state.board_card))
    return f"{private_rank}/{board_rank}|r{state.round_index}|{history_label(state)}"


@dataclass
class _Node:
    legal_actions: tuple[Action, ...]
    regret_sum: dict[Action, float]
    strategy_sum: dict[Action, float]

    @classmethod
    def create(cls, legal_actions: tuple[Action, ...]) -> "_Node":
        return cls(
            legal_actions=legal_actions,
            regret_sum={action: 0.0 for action in legal_actions},
            strategy_sum={action: 0.0 for action in legal_actions},
        )

    def current_strategy(self) -> dict[Action, float]:
        positives = {
            action: max(0.0, self.regret_sum[action]) for action in self.legal_actions
        }
        normalizer = sum(positives.values())
        if normalizer <= 0.0:
            uniform = 1.0 / len(self.legal_actions)
            return {action: uniform for action in self.legal_actions}
        return {action: positives[action] / normalizer for action in self.legal_actions}

    def average_strategy(self) -> dict[Action, float]:
        normalizer = sum(self.strategy_sum.values())
        if normalizer <= 0.0:
            uniform = 1.0 / len(self.legal_actions)
            return {action: uniform for action in self.legal_actions}
        return {
            action: self.strategy_sum[action] / normalizer for action in self.legal_actions
        }


class AverageStrategyPolicy:
    """A fixed policy (typically from CFR average strategy)."""

    def __init__(self, strategy: dict[str, dict[Action, float]]) -> None:
        self._strategy = strategy

    def action_probabilities(
        self, info_state: str, legal_actions: tuple[Action, ...]
    ) -> dict[Action, float]:
        if not legal_actions:
            return {}
        row = self._strategy.get(info_state)
        if not row:
            uniform = 1.0 / len(legal_actions)
            return {action: uniform for action in legal_actions}

        probs = {action: max(0.0, row.get(action, 0.0)) for action in legal_actions}
        total = sum(probs.values())
        if total <= 0.0:
            uniform = 1.0 / len(legal_actions)
            return {action: uniform for action in legal_actions}
        return {action: probs[action] / total for action in legal_actions}

    def sample_action(
        self, info_state: str, legal_actions: tuple[Action, ...], rng: random.Random
    ) -> Action:
        probs = self.action_probabilities(info_state, legal_actions)
        threshold = rng.random()
        running = 0.0
        chosen = legal_actions[-1]
        for action in legal_actions:
            running += probs[action]
            if threshold <= running:
                chosen = action
                break
        return chosen

    def greedy_action(self, info_state: str, legal_actions: tuple[Action, ...]) -> Action:
        probs = self.action_probabilities(info_state, legal_actions)
        return max(legal_actions, key=lambda action: probs[action])

    def to_json_dict(self) -> dict[str, dict[str, float]]:
        payload: dict[str, dict[str, float]] = {}
        for key, row in self._strategy.items():
            payload[key] = {action.value: value for action, value in row.items()}
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, dict[str, float]]) -> "AverageStrategyPolicy":
        strategy: dict[str, dict[Action, float]] = {}
        for key, row in payload.items():
            strategy[key] = {Action(action_text): float(value) for action_text, value in row.items()}
        return cls(strategy)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AverageStrategyPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Policy file must contain a JSON object.")
        return cls.from_json_dict(payload)

    def info_states(self) -> list[str]:
        return sorted(self._strategy.keys())


class CFRTrainer:
    """Tabular CFR trainer over the full Leduc chance tree."""

    def __init__(self) -> None:
        self._nodes: dict[str, _Node] = {}
        self._deals = all_deals()

    def train(
        self, *, iterations: int = 4_000, log_every: int = 400
    ) -> list[dict[str, float]]:
        if iterations <= 0:
            raise ValueError("iterations must be > 0")

        logs: list[dict[str, float]] = []
        for iteration in range(1, iterations + 1):
            value_accumulator = 0.0
            for traverser in (0, 1):
                for card0, card1, board in self._deals:
                    state = initial_state(cards=(card0, card1), board_card=board)
                    value = self._cfr(state, traverser, 1.0, 1.0)
                    if traverser == 0:
                        value_accumulator += value

            if log_every > 0 and iteration % log_every == 0:
                logs.append(
                    {
                        "iteration": float(iteration),
                        "avg_p0_value": value_accumulator / len(self._deals),
                        "infosets": float(len(self._nodes)),
                    }
                )
        return logs

    def average_policy(self) -> AverageStrategyPolicy:
        strategy: dict[str, dict[Action, float]] = {}
        for key, node in self._nodes.items():
            strategy[key] = node.average_strategy()
        return AverageStrategyPolicy(strategy)

    def _node_for(self, key: str, legal_actions: tuple[Action, ...]) -> _Node:
        existing = self._nodes.get(key)
        if existing is None:
            created = _Node.create(legal_actions)
            self._nodes[key] = created
            return created
        if existing.legal_actions != legal_actions:
            raise RuntimeError(
                f"Inconsistent legal actions for infoset {key!r}: "
                f"{existing.legal_actions} vs {legal_actions}"
            )
        return existing

    def _cfr(self, state: LeducPokerState, traverser: int, reach0: float, reach1: float) -> float:
        if state.terminal:
            return float(terminal_rewards(state)[traverser])

        current = state.current_player
        legal = legal_actions_for_state(state)
        key = info_state_key(state, current)
        node = self._node_for(key, legal)
        strategy = node.current_strategy()

        # Accumulate average strategy weighted by player's reach probability.
        player_reach = reach0 if current == 0 else reach1
        for action in legal:
            node.strategy_sum[action] += player_reach * strategy[action]

        if current == traverser:
            action_utilities: dict[Action, float] = {}
            node_utility = 0.0

            for action in legal:
                next_state, _, _ = step_state(state, action)
                next_reach0 = reach0
                next_reach1 = reach1
                if current == 0:
                    next_reach0 *= strategy[action]
                else:
                    next_reach1 *= strategy[action]
                utility = self._cfr(next_state, traverser, next_reach0, next_reach1)
                action_utilities[action] = utility
                node_utility += strategy[action] * utility

            opp_reach = reach1 if traverser == 0 else reach0
            for action in legal:
                regret = action_utilities[action] - node_utility
                node.regret_sum[action] += opp_reach * regret
            return node_utility

        node_utility = 0.0
        for action in legal:
            next_state, _, _ = step_state(state, action)
            next_reach0 = reach0
            next_reach1 = reach1
            if current == 0:
                next_reach0 *= strategy[action]
            else:
                next_reach1 *= strategy[action]
            node_utility += strategy[action] * self._cfr(
                next_state, traverser, next_reach0, next_reach1
            )
        return node_utility


def evaluate_vs_random(
    policy: AverageStrategyPolicy,
    *,
    episodes: int = 30_000,
    seed: int | None = None,
    as_player: int = 0,
    greedy: bool = False,
) -> float:
    """Average chips/hand against a random opponent."""
    if as_player not in (0, 1):
        raise ValueError("as_player must be 0 or 1")
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    rng = random.Random(seed)
    game = LeducPokerGame(rng=rng)
    total = 0.0

    for _ in range(episodes):
        state = game.reset()
        while True:
            legal = game.legal_actions()
            player = state.current_player
            if player == as_player:
                key = info_state_key(state, player)
                if greedy:
                    action = policy.greedy_action(key, legal)
                else:
                    action = policy.sample_action(key, legal, rng)
            else:
                action = rng.choice(list(legal))

            state, rewards, done = game.step(action)
            if done:
                total += rewards[as_player]
                break
    return total / episodes


def policy_table(policy: AverageStrategyPolicy) -> list[tuple[str, dict[str, float]]]:
    rows: list[tuple[str, dict[str, float]]] = []
    payload = policy.to_json_dict()
    for key in policy.info_states():
        row = payload.get(key, {})
        if row:
            rows.append((key, row))
    return rows
