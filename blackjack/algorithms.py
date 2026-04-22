"""Blackjack policies and tabular learning algorithms."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

from .game import Action, BlackjackGame, BlackjackState, blackjack_value, hand_value


@dataclass(frozen=True)
class Decision:
    """One decision made during an episode."""

    player: int
    info_state: str
    legal_actions: tuple[Action, ...]
    action: Action


def info_state_key(state: BlackjackState) -> str:
    """Return the visible blackjack information state."""
    total, soft = hand_value(state.player_hand)
    dealer_upcard = blackjack_value(state.dealer_hand[0].rank)
    dealer_token = "A" if dealer_upcard == 11 else "T" if dealer_upcard == 10 else str(dealer_upcard)
    softness = "s" if soft else "h"
    return f"p{total}{softness}|d{dealer_token}|n{len(state.player_hand)}|dbl{int(state.can_double)}"


def _choose_double_action(total: int, soft: bool, dealer_upcard: int) -> Action | None:
    if soft:
        if total == 18 and 3 <= dealer_upcard <= 6:
            return Action.DOUBLE
        if total in (15, 16, 17) and 4 <= dealer_upcard <= 6:
            return Action.DOUBLE
    else:
        if total == 11:
            return Action.DOUBLE
        if total == 10 and 2 <= dealer_upcard <= 9:
            return Action.DOUBLE
        if total == 9 and 3 <= dealer_upcard <= 6:
            return Action.DOUBLE
    return None


def _basic_strategy_action(state: BlackjackState, legal_actions: tuple[Action, ...]) -> Action:
    total, soft = hand_value(state.player_hand)
    dealer_upcard = blackjack_value(state.dealer_hand[0].rank)

    double_choice = _choose_double_action(total, soft, dealer_upcard)
    if double_choice is not None and double_choice in legal_actions:
        return double_choice

    if soft:
        if total >= 19:
            return Action.STAND
        if total == 18:
            return Action.STAND if dealer_upcard in (2, 7, 8) else Action.HIT
        return Action.HIT

    if total >= 17:
        return Action.STAND
    if total in (13, 14, 15, 16):
        return Action.STAND if 2 <= dealer_upcard <= 6 else Action.HIT
    if total == 12:
        return Action.STAND if 4 <= dealer_upcard <= 6 else Action.HIT
    return Action.HIT


class RandomPolicy:
    """Uniformly random policy over legal actions."""

    def choose_action(
        self,
        state: BlackjackState,
        legal_actions: tuple[Action, ...],
        rng: random.Random,
        *,
        greedy: bool = False,
        epsilon: float = 0.0,
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        return rng.choice(list(legal_actions))


class BasicStrategyPolicy:
    """Heuristic strategy approximating common blackjack basic strategy."""

    def choose_action(
        self,
        state: BlackjackState,
        legal_actions: tuple[Action, ...],
        rng: random.Random,
        *,
        greedy: bool = False,
        epsilon: float = 0.0,
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        chosen = _basic_strategy_action(state, legal_actions)
        if chosen in legal_actions:
            return chosen
        if Action.STAND in legal_actions:
            return Action.STAND
        return legal_actions[0]


class TabularActionValuePolicy:
    """Tabular Q-value policy for blackjack."""

    def __init__(self) -> None:
        self._q_values: dict[str, dict[Action, float]] = {}
        self._counts: dict[str, dict[Action, int]] = {}

    def _ensure(self, info_state: str, legal_actions: tuple[Action, ...]) -> None:
        q_row = self._q_values.setdefault(info_state, {})
        c_row = self._counts.setdefault(info_state, {})
        for action in legal_actions:
            q_row.setdefault(action, 0.0)
            c_row.setdefault(action, 0)

    def action_values(self, info_state: str, legal_actions: tuple[Action, ...]) -> dict[Action, float]:
        if not legal_actions:
            return {}
        self._ensure(info_state, legal_actions)
        return {action: self._q_values[info_state][action] for action in legal_actions}

    def greedy_action(self, info_state: str, legal_actions: tuple[Action, ...]) -> Action:
        values = self.action_values(info_state, legal_actions)
        return max(legal_actions, key=lambda action: values[action])

    def choose_action(
        self,
        state: BlackjackState,
        legal_actions: tuple[Action, ...],
        rng: random.Random,
        *,
        greedy: bool = False,
        epsilon: float = 0.0,
    ) -> Action:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        if not greedy and epsilon > 0.0 and rng.random() < epsilon:
            return rng.choice(list(legal_actions))
        return self.greedy_action(info_state_key(state), legal_actions)

    def update_monte_carlo(
        self,
        info_state: str,
        legal_actions: tuple[Action, ...],
        action: Action,
        return_value: float,
    ) -> None:
        self._ensure(info_state, legal_actions)
        self._counts[info_state][action] += 1
        count = self._counts[info_state][action]
        current = self._q_values[info_state][action]
        self._q_values[info_state][action] = current + (return_value - current) / count

    def update_q_learning(
        self,
        info_state: str,
        legal_actions: tuple[Action, ...],
        action: Action,
        target: float,
        learning_rate: float,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")
        self._ensure(info_state, legal_actions)
        current = self._q_values[info_state][action]
        self._q_values[info_state][action] = current + learning_rate * (target - current)

    def info_states(self) -> list[str]:
        return sorted(self._q_values.keys())

    def to_json_dict(self) -> dict[str, dict[str, dict[str, float | int]]]:
        q_payload: dict[str, dict[str, float]] = {}
        count_payload: dict[str, dict[str, int]] = {}
        for info_state, row in self._q_values.items():
            q_payload[info_state] = {action.value: value for action, value in row.items()}
        for info_state, row in self._counts.items():
            count_payload[info_state] = {action.value: count for action, count in row.items()}
        return {"q": q_payload, "counts": count_payload}

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "TabularActionValuePolicy":
        policy = cls()
        if "q" in payload:
            q_payload = payload.get("q", {})
            count_payload = payload.get("counts", {})
        else:
            q_payload = payload
            count_payload = {}

        if not isinstance(q_payload, dict) or not isinstance(count_payload, dict):
            raise ValueError("Policy file has an unexpected structure.")

        for info_state, row in q_payload.items():
            if not isinstance(row, dict):
                raise ValueError("Q-value rows must be objects.")
            parsed_row: dict[Action, float] = {}
            for action_text, value in row.items():
                parsed_row[Action(action_text)] = float(value)
            policy._q_values[info_state] = parsed_row

        for info_state, row in count_payload.items():
            if not isinstance(row, dict):
                raise ValueError("Count rows must be objects.")
            parsed_row: dict[Action, int] = {}
            for action_text, value in row.items():
                parsed_row[Action(action_text)] = int(value)
            policy._counts[info_state] = parsed_row

        return policy

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TabularActionValuePolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Policy file must contain a JSON object.")
        return cls.from_json_dict(payload)


def train_monte_carlo_control(
    *,
    episodes: int = 200_000,
    epsilon: float = 0.1,
    seed: int | None = None,
    log_every: int = 10_000,
) -> tuple[TabularActionValuePolicy, list[dict[str, float]]]:
    """Train a first-visit Monte Carlo control policy."""
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if epsilon < 0.0:
        raise ValueError("epsilon must be >= 0")

    rng = random.Random(seed)
    game = BlackjackGame(rng=rng)
    policy = TabularActionValuePolicy()
    logs: list[dict[str, float]] = []
    window_total = 0.0
    window_count = 0

    for episode in range(1, episodes + 1):
        rewards, decisions = _run_episode(game, policy, rng, epsilon=epsilon)
        reward = rewards[0]
        seen: set[tuple[str, Action]] = set()
        for decision in decisions:
            key = (decision.info_state, decision.action)
            if key in seen:
                continue
            seen.add(key)
            policy.update_monte_carlo(
                decision.info_state,
                decision.legal_actions,
                decision.action,
                reward,
            )

        window_total += reward
        window_count += 1
        if log_every > 0 and episode % log_every == 0:
            logs.append(
                {
                    "episode": float(episode),
                    "avg_reward": window_total / window_count,
                }
            )
            window_total = 0.0
            window_count = 0

    return policy, logs


def train_q_learning(
    *,
    episodes: int = 200_000,
    epsilon: float = 0.1,
    learning_rate: float = 0.05,
    gamma: float = 1.0,
    seed: int | None = None,
    log_every: int = 10_000,
) -> tuple[TabularActionValuePolicy, list[dict[str, float]]]:
    """Train a tabular Q-learning policy."""
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if epsilon < 0.0:
        raise ValueError("epsilon must be >= 0")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if gamma < 0.0:
        raise ValueError("gamma must be >= 0")

    rng = random.Random(seed)
    game = BlackjackGame(rng=rng)
    policy = TabularActionValuePolicy()
    logs: list[dict[str, float]] = []
    window_total = 0.0
    window_count = 0

    for episode in range(1, episodes + 1):
        state = game.reset()
        if state.terminal:
            reward = state.rewards[0]
            window_total += reward
            window_count += 1
            if log_every > 0 and episode % log_every == 0:
                logs.append(
                    {
                        "episode": float(episode),
                        "avg_reward": window_total / window_count,
                    }
                )
                window_total = 0.0
                window_count = 0
            continue

        while True:
            legal = game.legal_actions()
            info_state = info_state_key(state)
            action = policy.choose_action(state, legal, rng, epsilon=epsilon)
            next_state, rewards, done = game.step(action)
            target = rewards[0]
            if not done:
                next_legal = game.legal_actions()
                next_values = policy.action_values(info_state_key(next_state), next_legal)
                if next_values:
                    target += gamma * max(next_values.values())
            policy.update_q_learning(info_state, legal, action, target, learning_rate)
            state = next_state
            if done:
                reward = rewards[0]
                window_total += reward
                window_count += 1
                break

        if log_every > 0 and episode % log_every == 0:
            logs.append(
                {
                    "episode": float(episode),
                    "avg_reward": window_total / window_count if window_count else 0.0,
                }
            )
            window_total = 0.0
            window_count = 0

    return policy, logs


def evaluate_policy(
    policy: object,
    *,
    episodes: int = 20_000,
    seed: int | None = None,
    greedy: bool = True,
) -> float:
    """Average chips per hand for a policy."""
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    rng = random.Random(seed)
    game = BlackjackGame(rng=rng)
    total = 0.0

    for _ in range(episodes):
        state = game.reset()
        if state.terminal:
            total += state.rewards[0]
            continue

        while True:
            legal = game.legal_actions()
            action = policy.choose_action(state, legal, rng, greedy=greedy)
            state, rewards, done = game.step(action)
            if done:
                total += rewards[0]
                break

    return total / episodes


def policy_table(policy: TabularActionValuePolicy) -> list[tuple[str, dict[str, float]]]:
    """Readable table of learned Q-values."""
    rows: list[tuple[str, dict[str, float]]] = []
    payload = policy.to_json_dict().get("q", {})
    if not isinstance(payload, dict):
        return rows
    for info_state in policy.info_states():
        row = payload.get(info_state, {})
        if not row:
            continue
        rows.append((info_state, {action_text: float(value) for action_text, value in row.items()}))
    return rows


def _run_episode(
    game: BlackjackGame,
    policy: TabularActionValuePolicy,
    rng: random.Random,
    *,
    epsilon: float,
) -> tuple[tuple[float, float], list[Decision]]:
    state = game.reset()
    decisions: list[Decision] = []
    if state.terminal:
        return state.rewards, decisions

    while True:
        legal = game.legal_actions()
        info_state = info_state_key(state)
        action = policy.choose_action(state, legal, rng, epsilon=epsilon)
        decisions.append(
            Decision(
                player=0,
                info_state=info_state,
                legal_actions=legal,
                action=action,
            )
        )
        state, rewards, done = game.step(action)
        if done:
            return rewards, decisions
