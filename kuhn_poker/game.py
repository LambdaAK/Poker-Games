"""Minimal Kuhn Poker environment for simulation and RL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import random


class Card(IntEnum):
    """Kuhn Poker card ranks."""

    J = 0
    Q = 1
    K = 2


class Action(str, Enum):
    """Actions available in Kuhn Poker."""

    CHECK = "check"
    BET = "bet"
    CALL = "call"
    FOLD = "fold"


_START_ACTIONS = (Action.CHECK, Action.BET)
_RESPONSE_ACTIONS = (Action.CALL, Action.FOLD)
_TERMINAL_HISTORIES = {
    (Action.CHECK, Action.CHECK),
    (Action.BET, Action.CALL),
    (Action.BET, Action.FOLD),
    (Action.CHECK, Action.BET, Action.CALL),
    (Action.CHECK, Action.BET, Action.FOLD),
}


@dataclass(frozen=True)
class KuhnPokerState:
    """Immutable environment state."""

    player_cards: tuple[Card, Card]
    history: tuple[Action, ...]
    current_player: int
    pot: int
    terminal: bool


class KuhnPokerGame:
    """Stateful Kuhn Poker environment."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._state: KuhnPokerState | None = None

    @property
    def state(self) -> KuhnPokerState:
        """Return the current state."""
        if self._state is None:
            raise RuntimeError("Game has not been reset. Call reset() first.")
        return self._state

    def reset(
        self, *, seed: int | None = None, cards: tuple[Card, Card] | None = None
    ) -> KuhnPokerState:
        """Start a new hand and return initial state."""
        if seed is not None:
            self._rng.seed(seed)

        if cards is None:
            shuffled = list(Card)
            self._rng.shuffle(shuffled)
            dealt = (shuffled[0], shuffled[1])
        else:
            dealt = cards

        if dealt[0] == dealt[1]:
            raise ValueError("Kuhn Poker requires distinct private cards.")

        self._state = KuhnPokerState(
            player_cards=dealt,
            history=(),
            current_player=0,
            pot=2,  # antes only
            terminal=False,
        )
        return self._state

    def legal_actions(self) -> tuple[Action, ...]:
        """Return legal actions for the current player."""
        state = self.state
        return _legal_actions(state.history, state.terminal)

    def step(self, action: Action | str) -> tuple[KuhnPokerState, tuple[int, int], bool]:
        """Apply one action and return (state, rewards, done)."""
        state = self.state
        parsed_action = _parse_action(action)

        legal = _legal_actions(state.history, state.terminal)
        if parsed_action not in legal:
            raise ValueError(
                f"Illegal action {parsed_action.value!r} for history "
                f"{_history_label(state.history)!r}. Legal: {[a.value for a in legal]}"
            )

        history = state.history + (parsed_action,)
        pot = state.pot + (1 if parsed_action in (Action.BET, Action.CALL) else 0)
        terminal = history in _TERMINAL_HISTORIES
        next_player = -1 if terminal else 1 - state.current_player

        self._state = KuhnPokerState(
            player_cards=state.player_cards,
            history=history,
            current_player=next_player,
            pot=pot,
            terminal=terminal,
        )

        rewards = _terminal_rewards(self._state)
        return self._state, rewards, terminal

    def history_label(self) -> str:
        """Compact history label, e.g., '', 'cbf', 'bc'."""
        return _history_label(self.state.history)


def _parse_action(action: Action | str) -> Action:
    if isinstance(action, Action):
        return action
    try:
        return Action(action)
    except ValueError as exc:
        raise ValueError(f"Unknown action {action!r}.") from exc


def _legal_actions(history: tuple[Action, ...], terminal: bool) -> tuple[Action, ...]:
    if terminal:
        return ()
    if history == ():
        return _START_ACTIONS
    if history == (Action.CHECK,):
        return _START_ACTIONS
    if history == (Action.BET,):
        return _RESPONSE_ACTIONS
    if history == (Action.CHECK, Action.BET):
        return _RESPONSE_ACTIONS
    raise RuntimeError(f"Unexpected non-terminal history: {_history_label(history)!r}.")


def _history_label(history: tuple[Action, ...]) -> str:
    token = {
        Action.CHECK: "c",
        Action.BET: "b",
        Action.CALL: "c",
        Action.FOLD: "f",
    }
    return "".join(token[a] for a in history)


def _terminal_rewards(state: KuhnPokerState) -> tuple[int, int]:
    if not state.terminal:
        return (0, 0)

    history = state.history
    if history == (Action.BET, Action.FOLD):
        return (1, -1)
    if history == (Action.CHECK, Action.BET, Action.FOLD):
        return (-1, 1)

    if history not in {
        (Action.CHECK, Action.CHECK),
        (Action.BET, Action.CALL),
        (Action.CHECK, Action.BET, Action.CALL),
    }:
        raise RuntimeError(f"Invalid terminal history: {_history_label(history)!r}.")

    showdown_value = 1 if history == (Action.CHECK, Action.CHECK) else 2
    winner = 0 if state.player_cards[0] > state.player_cards[1] else 1
    if winner == 0:
        return (showdown_value, -showdown_value)
    return (-showdown_value, showdown_value)
