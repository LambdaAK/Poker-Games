"""Tabular policy-gradient (REINFORCE) training for Kuhn Poker."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

from .game import Action, Card, KuhnPokerGame


@dataclass(frozen=True)
class Decision:
    """One policy decision made during an episode."""

    player: int
    info_state: str
    legal_actions: tuple[Action, ...]
    action: Action


def info_state_key(card: Card, history_label: str) -> str:
    """Information-set key visible to a single player."""
    return f"{card.name}|{history_label}"


def legal_actions_for_history(history_label: str) -> tuple[Action, ...]:
    """Map compact history labels to legal actions."""
    if history_label in ("", "c"):
        return (Action.CHECK, Action.BET)
    if history_label in ("b", "cb"):
        return (Action.CALL, Action.FOLD)
    return ()


class TabularSoftmaxPolicy:
    """Softmax policy over information states."""

    def __init__(self) -> None:
        self._preferences: dict[str, dict[Action, float]] = {}

    def _ensure(self, info_state: str, actions: tuple[Action, ...]) -> None:
        prefs = self._preferences.setdefault(info_state, {})
        for action in actions:
            prefs.setdefault(action, 0.0)

    def action_probabilities(
        self, info_state: str, legal_actions: tuple[Action, ...]
    ) -> dict[Action, float]:
        """Return action probabilities on legal actions."""
        if not legal_actions:
            return {}
        self._ensure(info_state, legal_actions)
        prefs = self._preferences[info_state]
        logits = [prefs[a] for a in legal_actions]
        max_logit = max(logits)
        exps = [math.exp(x - max_logit) for x in logits]
        total = sum(exps)
        return {action: value / total for action, value in zip(legal_actions, exps)}

    def sample_action(
        self,
        info_state: str,
        legal_actions: tuple[Action, ...],
        rng: random.Random,
    ) -> Action:
        """Sample action from current policy."""
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
        """Choose highest-probability legal action."""
        probs = self.action_probabilities(info_state, legal_actions)
        return max(legal_actions, key=lambda action: probs[action])

    def update_episode(
        self,
        decisions: list[Decision],
        rewards: tuple[int, int],
        learning_rate: float,
        baseline: dict[int, float] | None = None,
    ) -> None:
        """Apply REINFORCE update from a full episode."""
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
        for info_state, prefs in self._preferences.items():
            payload[info_state] = {action.value: value for action, value in prefs.items()}
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, dict[str, float]]) -> "TabularSoftmaxPolicy":
        policy = cls()
        for info_state, prefs in payload.items():
            parsed: dict[Action, float] = {}
            for action_text, value in prefs.items():
                parsed[Action(action_text)] = float(value)
            policy._preferences[info_state] = parsed
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
    episodes: int = 200_000,
    learning_rate: float = 0.05,
    baseline_lr: float = 0.01,
    seed: int | None = None,
    log_every: int = 10_000,
) -> tuple[TabularSoftmaxPolicy, list[dict[str, float]]]:
    """Train a shared policy with self-play REINFORCE."""
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if baseline_lr <= 0.0:
        raise ValueError("baseline_lr must be > 0")

    rng = random.Random(seed)
    game = KuhnPokerGame(rng=rng)
    policy = TabularSoftmaxPolicy()
    baselines = {0: 0.0, 1: 0.0}

    logs: list[dict[str, float]] = []
    window_sum_p0 = 0.0
    window_count = 0

    for episode in range(1, episodes + 1):
        rewards, decisions = _play_training_hand(game, policy, rng)
        policy.update_episode(decisions, rewards, learning_rate, baseline=baselines)

        for player in (0, 1):
            baselines[player] += baseline_lr * (rewards[player] - baselines[player])

        window_sum_p0 += rewards[0]
        window_count += 1

        if log_every > 0 and episode % log_every == 0:
            logs.append(
                {
                    "episode": float(episode),
                    "avg_p0_reward": window_sum_p0 / window_count,
                    "baseline_p0": baselines[0],
                    "baseline_p1": baselines[1],
                }
            )
            window_sum_p0 = 0.0
            window_count = 0

    return policy, logs


def _play_training_hand(
    game: KuhnPokerGame,
    policy: TabularSoftmaxPolicy,
    rng: random.Random,
) -> tuple[tuple[int, int], list[Decision]]:
    state = game.reset()
    decisions: list[Decision] = []

    while True:
        player = state.current_player
        legal_actions = game.legal_actions()
        history = game.history_label()
        info_state = info_state_key(state.player_cards[player], history)
        action = policy.sample_action(info_state, legal_actions, rng)
        decisions.append(
            Decision(
                player=player,
                info_state=info_state,
                legal_actions=legal_actions,
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
    """Average reward per hand vs a random opponent."""
    if as_player not in (0, 1):
        raise ValueError("as_player must be 0 or 1")
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    rng = random.Random(seed)
    game = KuhnPokerGame(rng=rng)
    total = 0.0

    for _ in range(episodes):
        state = game.reset()
        while True:
            legal_actions = game.legal_actions()
            player = state.current_player
            if player == as_player:
                info_state = info_state_key(state.player_cards[player], game.history_label())
                if greedy:
                    action = policy.greedy_action(info_state, legal_actions)
                else:
                    action = policy.sample_action(info_state, legal_actions, rng)
            else:
                action = rng.choice(list(legal_actions))

            state, rewards, done = game.step(action)
            if done:
                total += rewards[as_player]
                break

    return total / episodes


def policy_table(policy: TabularSoftmaxPolicy) -> list[tuple[str, dict[str, float]]]:
    """Readable policy table with legal-action probabilities."""
    rows: list[tuple[str, dict[str, float]]] = []
    for info_state in policy.info_states():
        _, history = info_state.split("|", 1)
        legal = legal_actions_for_history(history)
        if not legal:
            continue
        probs = policy.action_probabilities(info_state, legal)
        rows.append((info_state, {a.value: probs[a] for a in legal}))
    return rows
