"""Chance-sampled CFR for heads-up limit Hold'em."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

from .abstraction import info_state_key
from .game import Action, HoldemLimitGame, HoldemLimitState, legal_actions_for_state, step_state


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
        total = sum(positives.values())
        if total <= 0.0:
            uniform = 1.0 / len(self.legal_actions)
            return {action: uniform for action in self.legal_actions}
        return {action: positives[action] / total for action in self.legal_actions}

    def average_strategy(self) -> dict[Action, float]:
        total = sum(self.strategy_sum.values())
        if total <= 0.0:
            uniform = 1.0 / len(self.legal_actions)
            return {action: uniform for action in self.legal_actions}
        return {action: self.strategy_sum[action] / total for action in self.legal_actions}


class AverageStrategyPolicy:
    """Fixed policy (typically CFR average strategy)."""

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
        clipped = {action: max(0.0, row.get(action, 0.0)) for action in legal_actions}
        total = sum(clipped.values())
        if total <= 0.0:
            uniform = 1.0 / len(legal_actions)
            return {action: uniform for action in legal_actions}
        return {action: clipped[action] / total for action in legal_actions}

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

    def info_states(self) -> list[str]:
        return sorted(self._strategy.keys())

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


class CFRTrainer:
    """Chance-sampled CFR over abstract infosets."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._game = HoldemLimitGame(rng=self._rng)
        self._nodes: dict[str, _Node] = {}

    def train(
        self, *, iterations: int = 5_000, log_every: int = 500
    ) -> list[dict[str, float]]:
        if iterations <= 0:
            raise ValueError("iterations must be > 0")

        logs: list[dict[str, float]] = []
        window_value = 0.0
        window_count = 0

        for iteration in range(1, iterations + 1):
            for traverser in (0, 1):
                state = self._game.reset()
                value = self._cfr(state, traverser, 1.0, 1.0)
                if traverser == 0:
                    window_value += value
                    window_count += 1

            if log_every > 0 and iteration % log_every == 0:
                logs.append(
                    {
                        "iteration": float(iteration),
                        "avg_p0_value": window_value / max(1, window_count),
                        "infosets": float(len(self._nodes)),
                    }
                )
                window_value = 0.0
                window_count = 0

        return logs

    def average_policy(self) -> AverageStrategyPolicy:
        strategy: dict[str, dict[Action, float]] = {}
        for key, node in self._nodes.items():
            strategy[key] = node.average_strategy()
        return AverageStrategyPolicy(strategy)

    def _node_for(self, key: str, legal: tuple[Action, ...]) -> _Node:
        node = self._nodes.get(key)
        if node is None:
            created = _Node.create(legal)
            self._nodes[key] = created
            return created
        if node.legal_actions != legal:
            raise RuntimeError(
                f"Inconsistent legal action set for infoset {key!r}: "
                f"{node.legal_actions} vs {legal}"
            )
        return node

    def _cfr(self, state: HoldemLimitState, traverser: int, reach0: float, reach1: float) -> float:
        if state.terminal:
            # rewards are computed by step transition already embedded in terminal state
            from .game import terminal_rewards

            return float(terminal_rewards(state)[traverser])

        current = state.current_player
        legal = legal_actions_for_state(state)
        key = info_state_key(state, current)
        node = self._node_for(key, legal)
        strategy = node.current_strategy()

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
                util = self._cfr(next_state, traverser, next_reach0, next_reach1)
                action_utilities[action] = util
                node_utility += strategy[action] * util

            opp_reach = reach1 if traverser == 0 else reach0
            for action in legal:
                regret = action_utilities[action] - node_utility
                node.regret_sum[action] += opp_reach * regret
            return node_utility

        # External-sampling MCCFR: sample one opponent action instead of full expectation.
        sampled = _sample_action(strategy, legal, self._rng)
        next_state, _, _ = step_state(state, sampled)
        next_reach0 = reach0
        next_reach1 = reach1
        if current == 0:
            next_reach0 *= strategy[sampled]
        else:
            next_reach1 *= strategy[sampled]
        return self._cfr(next_state, traverser, next_reach0, next_reach1)


def evaluate_vs_random(
    policy: AverageStrategyPolicy,
    *,
    episodes: int = 20_000,
    seed: int | None = None,
    as_player: int = 0,
    greedy: bool = False,
) -> float:
    """Average chips/hand against random opponent."""
    if as_player not in (0, 1):
        raise ValueError("as_player must be 0 or 1")
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    rng = random.Random(seed)
    game = HoldemLimitGame(rng=rng)
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


def _sample_action(
    strategy: dict[Action, float], legal_actions: tuple[Action, ...], rng: random.Random
) -> Action:
    threshold = rng.random()
    running = 0.0
    chosen = legal_actions[-1]
    for action in legal_actions:
        running += strategy[action]
        if threshold <= running:
            chosen = action
            break
    return chosen
