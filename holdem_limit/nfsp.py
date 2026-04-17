"""Tabular NFSP-style training for heads-up limit Hold'em.

This implementation keeps:
- a tabular best-response Q learner (epsilon-greedy)
- a tabular average policy trained from a reservoir of BR actions
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

from .abstraction import info_state_key
from .game import Action, HoldemLimitGame


@dataclass(frozen=True)
class _ReservoirEntry:
    info_state: str
    action: Action


@dataclass(frozen=True)
class _PendingTransition:
    info_state: str
    action: Action


class TabularBestResponseQ:
    """Tabular Q values used as the best-response policy in NFSP."""

    def __init__(self) -> None:
        self._q_values: dict[str, dict[Action, float]] = {}

    def _ensure(self, info_state: str, legal_actions: tuple[Action, ...]) -> None:
        row = self._q_values.setdefault(info_state, {})
        for action in legal_actions:
            row.setdefault(action, 0.0)

    def q_values(self, info_state: str, legal_actions: tuple[Action, ...]) -> dict[Action, float]:
        if not legal_actions:
            return {}
        self._ensure(info_state, legal_actions)
        row = self._q_values[info_state]
        return {action: row[action] for action in legal_actions}

    def greedy_action(self, info_state: str, legal_actions: tuple[Action, ...]) -> Action:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty.")
        values = self.q_values(info_state, legal_actions)
        return max(legal_actions, key=lambda action: values[action])

    def sample_action(
        self,
        info_state: str,
        legal_actions: tuple[Action, ...],
        rng: random.Random,
        epsilon: float,
    ) -> Action:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty.")
        if rng.random() < epsilon:
            return rng.choice(list(legal_actions))
        return self.greedy_action(info_state, legal_actions)

    def best_value(self, info_state: str, legal_actions: tuple[Action, ...]) -> float:
        if not legal_actions:
            return 0.0
        values = self.q_values(info_state, legal_actions)
        return max(values.values())

    def td_update(self, info_state: str, action: Action, target: float, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        row = self._q_values.setdefault(info_state, {})
        current = row.get(action, 0.0)
        row[action] = current + learning_rate * (target - current)

    def info_states(self) -> list[str]:
        return sorted(self._q_values.keys())


class NfspAveragePolicy:
    """Average policy trained from a reservoir of best-response actions."""

    def __init__(
        self, *, reservoir_capacity: int = 200_000, rng: random.Random | None = None
    ) -> None:
        if reservoir_capacity <= 0:
            raise ValueError("reservoir_capacity must be > 0")
        self._reservoir_capacity = reservoir_capacity
        self._rng = rng if rng is not None else random.Random()
        self._samples_seen = 0
        self._reservoir: list[_ReservoirEntry] = []
        self._counts: dict[str, dict[Action, float]] = {}

    def _bump(self, info_state: str, action: Action, delta: float) -> None:
        row = self._counts.setdefault(info_state, {})
        row[action] = row.get(action, 0.0) + delta
        if row[action] <= 0.0:
            row.pop(action, None)
        if not row:
            self._counts.pop(info_state, None)

    def observe(self, info_state: str, action: Action) -> None:
        """Record one best-response action into the reservoir."""
        item = _ReservoirEntry(info_state=info_state, action=action)
        self._samples_seen += 1

        if len(self._reservoir) < self._reservoir_capacity:
            self._reservoir.append(item)
            self._bump(info_state, action, 1.0)
            return

        index = self._rng.randrange(self._samples_seen)
        if index >= self._reservoir_capacity:
            return

        removed = self._reservoir[index]
        self._bump(removed.info_state, removed.action, -1.0)
        self._reservoir[index] = item
        self._bump(info_state, action, 1.0)

    def action_probabilities(
        self, info_state: str, legal_actions: tuple[Action, ...]
    ) -> dict[Action, float]:
        if not legal_actions:
            return {}

        row = self._counts.get(info_state)
        if not row:
            uniform = 1.0 / len(legal_actions)
            return {action: uniform for action in legal_actions}

        positive = {action: max(0.0, row.get(action, 0.0)) for action in legal_actions}
        total = sum(positive.values())
        if total <= 0.0:
            uniform = 1.0 / len(legal_actions)
            return {action: uniform for action in legal_actions}
        return {action: positive[action] / total for action in legal_actions}

    def sample_action(
        self,
        info_state: str,
        legal_actions: tuple[Action, ...],
        rng: random.Random,
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
        return sorted(self._counts.keys())

    def reservoir_size(self) -> int:
        return len(self._reservoir)

    def to_json_dict(self) -> dict[str, dict[str, float]]:
        payload: dict[str, dict[str, float]] = {}
        for key, row in self._counts.items():
            payload[key] = {action.value: value for action, value in row.items()}
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, dict[str, float]]) -> "NfspAveragePolicy":
        policy = cls(reservoir_capacity=1)
        policy._reservoir = []
        policy._samples_seen = 0
        policy._counts = {}
        for key, row in payload.items():
            parsed: dict[Action, float] = {}
            for action_text, value in row.items():
                parsed[Action(action_text)] = float(value)
            policy._counts[key] = parsed
        return policy

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NfspAveragePolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Policy file must contain a JSON object.")
        return cls.from_json_dict(payload)


def train_self_play(
    *,
    episodes: int = 350_000,
    q_learning_rate: float = 0.08,
    epsilon_start: float = 0.20,
    epsilon_end: float = 0.02,
    anticipatory: float = 0.10,
    discount: float = 1.0,
    reservoir_capacity: int = 200_000,
    seed: int | None = None,
    log_every: int = 10_000,
) -> tuple[NfspAveragePolicy, list[dict[str, float]]]:
    """Train an NFSP-style average policy via self-play."""
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if q_learning_rate <= 0.0:
        raise ValueError("q_learning_rate must be > 0")
    if not 0.0 <= epsilon_start <= 1.0:
        raise ValueError("epsilon_start must be in [0, 1]")
    if not 0.0 <= epsilon_end <= 1.0:
        raise ValueError("epsilon_end must be in [0, 1]")
    if not 0.0 <= anticipatory <= 1.0:
        raise ValueError("anticipatory must be in [0, 1]")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    if reservoir_capacity <= 0:
        raise ValueError("reservoir_capacity must be > 0")

    rng = random.Random(seed)
    game = HoldemLimitGame(rng=rng)
    br_q = TabularBestResponseQ()
    avg_policy = NfspAveragePolicy(reservoir_capacity=reservoir_capacity, rng=rng)

    logs: list[dict[str, float]] = []
    window_sum = 0.0
    window_count = 0

    for episode in range(1, episodes + 1):
        if episodes == 1:
            epsilon = epsilon_start
        else:
            ratio = (episode - 1) / (episodes - 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * ratio

        use_br = {
            0: rng.random() < anticipatory,
            1: rng.random() < anticipatory,
        }
        rewards = _play_training_hand(
            game=game,
            rng=rng,
            br_q=br_q,
            avg_policy=avg_policy,
            use_br=use_br,
            epsilon=epsilon,
            q_learning_rate=q_learning_rate,
            discount=discount,
        )
        window_sum += rewards[0]
        window_count += 1

        if log_every > 0 and episode % log_every == 0:
            logs.append(
                {
                    "episode": float(episode),
                    "avg_p0_reward": window_sum / max(1, window_count),
                    "epsilon": epsilon,
                    "q_states": float(len(br_q.info_states())),
                    "avg_states": float(len(avg_policy.info_states())),
                    "reservoir_size": float(avg_policy.reservoir_size()),
                }
            )
            window_sum = 0.0
            window_count = 0

    return avg_policy, logs


def _play_training_hand(
    *,
    game: HoldemLimitGame,
    rng: random.Random,
    br_q: TabularBestResponseQ,
    avg_policy: NfspAveragePolicy,
    use_br: dict[int, bool],
    epsilon: float,
    q_learning_rate: float,
    discount: float,
) -> tuple[int, int]:
    state = game.reset()
    pending: dict[int, _PendingTransition | None] = {0: None, 1: None}

    while True:
        player = state.current_player
        legal = game.legal_actions()
        key = info_state_key(state, player)

        prev = pending[player]
        if prev is not None:
            target = discount * br_q.best_value(key, legal)
            br_q.td_update(prev.info_state, prev.action, target, q_learning_rate)
            pending[player] = None

        if use_br[player]:
            action = br_q.sample_action(key, legal, rng, epsilon)
            avg_policy.observe(key, action)
            pending[player] = _PendingTransition(info_state=key, action=action)
        else:
            action = avg_policy.sample_action(key, legal, rng)

        state, rewards, done = game.step(action)
        if done:
            for update_player in (0, 1):
                prev = pending[update_player]
                if prev is not None:
                    br_q.td_update(
                        prev.info_state,
                        prev.action,
                        float(rewards[update_player]),
                        q_learning_rate,
                    )
                    pending[update_player] = None
            return rewards


def evaluate_vs_random(
    policy: NfspAveragePolicy,
    *,
    episodes: int = 20_000,
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


def policy_table(policy: NfspAveragePolicy) -> list[tuple[str, dict[str, float]]]:
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
