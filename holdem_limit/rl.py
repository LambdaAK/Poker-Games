"""Tabular REINFORCE self-play for heads-up limit Hold'em."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

from .abstraction import info_state_key
from .game import Action, HoldemLimitGame


@dataclass(frozen=True)
class Decision:
    """One decision made during an episode."""

    player: int
    info_state: str
    legal_actions: tuple[Action, ...]
    action: Action


class TabularSoftmaxPolicy:
    """Softmax policy over abstract infosets."""

    def __init__(self) -> None:
        self._preferences: dict[str, dict[Action, float]] = {}

    def _ensure(self, key: str, legal: tuple[Action, ...]) -> None:
        prefs = self._preferences.setdefault(key, {})
        for action in legal:
            prefs.setdefault(action, 0.0)

    def action_probabilities(self, key: str, legal: tuple[Action, ...]) -> dict[Action, float]:
        if not legal:
            return {}
        self._ensure(key, legal)
        prefs = self._preferences[key]
        logits = [prefs[action] for action in legal]
        peak = max(logits)
        exp_values = [math.exp(value - peak) for value in logits]
        total = sum(exp_values)
        return {action: value / total for action, value in zip(legal, exp_values)}

    def sample_action(self, key: str, legal: tuple[Action, ...], rng: random.Random) -> Action:
        probs = self.action_probabilities(key, legal)
        t = rng.random()
        run = 0.0
        chosen = legal[-1]
        for action in legal:
            run += probs[action]
            if t <= run:
                chosen = action
                break
        return chosen

    def greedy_action(self, key: str, legal: tuple[Action, ...]) -> Action:
        probs = self.action_probabilities(key, legal)
        return max(legal, key=lambda action: probs[action])

    def update_episode(
        self,
        decisions: list[Decision],
        rewards: tuple[int, int],
        learning_rate: float,
        baseline: dict[int, float] | None = None,
    ) -> None:
        for decision in decisions:
            probs = self.action_probabilities(decision.info_state, decision.legal_actions)
            advantage = float(rewards[decision.player])
            if baseline is not None:
                advantage -= baseline[decision.player]
            prefs = self._preferences[decision.info_state]
            for action in decision.legal_actions:
                indicator = 1.0 if action == decision.action else 0.0
                grad = indicator - probs[action]
                prefs[action] += learning_rate * advantage * grad

    def info_states(self) -> list[str]:
        return sorted(self._preferences.keys())

    def to_json_dict(self) -> dict[str, dict[str, float]]:
        payload: dict[str, dict[str, float]] = {}
        for key, prefs in self._preferences.items():
            payload[key] = {action.value: value for action, value in prefs.items()}
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, dict[str, float]]) -> "TabularSoftmaxPolicy":
        policy = cls()
        for key, row in payload.items():
            parsed: dict[Action, float] = {}
            for action_text, value in row.items():
                parsed[Action(action_text)] = float(value)
            policy._preferences[key] = parsed
        return policy

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TabularSoftmaxPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Policy file must contain a JSON object.")
        return cls.from_json_dict(payload)


def train_self_play(
    *,
    episodes: int = 250_000,
    learning_rate: float = 0.02,
    baseline_lr: float = 0.01,
    seed: int | None = None,
    log_every: int = 10_000,
) -> tuple[TabularSoftmaxPolicy, list[dict[str, float]]]:
    """Train tabular policy with self-play REINFORCE."""
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if baseline_lr <= 0.0:
        raise ValueError("baseline_lr must be > 0")

    rng = random.Random(seed)
    game = HoldemLimitGame(rng=rng)
    policy = TabularSoftmaxPolicy()
    baselines = {0: 0.0, 1: 0.0}
    logs: list[dict[str, float]] = []
    window_sum = 0.0
    window_count = 0

    for episode in range(1, episodes + 1):
        rewards, decisions = _play_training_hand(game, policy, rng)
        policy.update_episode(decisions, rewards, learning_rate, baseline=baselines)
        for player in (0, 1):
            baselines[player] += baseline_lr * (rewards[player] - baselines[player])

        window_sum += rewards[0]
        window_count += 1

        if log_every > 0 and episode % log_every == 0:
            logs.append(
                {
                    "episode": float(episode),
                    "avg_p0_reward": window_sum / window_count,
                    "baseline_p0": baselines[0],
                    "baseline_p1": baselines[1],
                }
            )
            window_sum = 0.0
            window_count = 0

    return policy, logs


def _play_training_hand(
    game: HoldemLimitGame, policy: TabularSoftmaxPolicy, rng: random.Random
) -> tuple[tuple[int, int], list[Decision]]:
    state = game.reset()
    decisions: list[Decision] = []

    while True:
        player = state.current_player
        legal = game.legal_actions()
        key = info_state_key(state, player)
        action = policy.sample_action(key, legal, rng)
        decisions.append(
            Decision(
                player=player,
                info_state=key,
                legal_actions=legal,
                action=action,
            )
        )
        state, rewards, done = game.step(action)
        if done:
            return rewards, decisions


def evaluate_vs_random(
    policy: TabularSoftmaxPolicy,
    *,
    episodes: int = 20_000,
    seed: int | None = None,
    as_player: int = 0,
    greedy: bool = False,
) -> float:
    """Average chips/hand vs random opponent."""
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


def policy_table(policy: TabularSoftmaxPolicy) -> list[tuple[str, dict[str, float]]]:
    """Readable infoset -> action probabilities table."""
    rows: list[tuple[str, dict[str, float]]] = []
    payload = policy.to_json_dict()
    for key in policy.info_states():
        row = payload.get(key, {})
        if not row:
            continue
        legal = tuple(Action(action_text) for action_text in row.keys())
        probs = policy.action_probabilities(key, legal)
        rows.append((key, {action.value: probs[action] for action in legal}))
    return rows
