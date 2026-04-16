"""Leduc Poker environment for simulation and learning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import random


class Rank(IntEnum):
    """Card ranks in Leduc poker."""

    J = 0
    Q = 1
    K = 2


class Card(IntEnum):
    """Distinct cards in a 6-card Leduc deck."""

    J1 = 0
    J2 = 1
    Q1 = 2
    Q2 = 3
    K1 = 4
    K2 = 5


class Action(str, Enum):
    """Actions available in limit Leduc poker."""

    CHECK = "check"
    BET = "bet"
    CALL = "call"
    RAISE = "raise"
    FOLD = "fold"


ANTE = 1
BET_SIZES = (1, 2)  # preflop, flop
MAX_RAISES_PER_ROUND = 1

ALL_CARDS = tuple(Card)
_CARD_RANK = {
    Card.J1: Rank.J,
    Card.J2: Rank.J,
    Card.Q1: Rank.Q,
    Card.Q2: Rank.Q,
    Card.K1: Rank.K,
    Card.K2: Rank.K,
}
_ACTION_TOKEN = {
    Action.CHECK: "k",
    Action.BET: "b",
    Action.CALL: "c",
    Action.RAISE: "r",
    Action.FOLD: "f",
}


@dataclass(frozen=True)
class LeducPokerState:
    """Immutable state for one Leduc poker hand."""

    player_cards: tuple[Card, Card]
    board_card: Card | None
    deck: tuple[Card, ...]
    round_index: int
    histories: tuple[tuple[Action, ...], tuple[Action, ...]]
    current_player: int
    to_call: int
    raises_in_round: int
    contributions: tuple[int, int]
    pot: int
    terminal: bool
    folded_player: int | None


class LeducPokerGame:
    """Stateful wrapper around pure Leduc transitions."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._state: LeducPokerState | None = None

    @property
    def state(self) -> LeducPokerState:
        if self._state is None:
            raise RuntimeError("Game has not been reset. Call reset() first.")
        return self._state

    def reset(
        self,
        *,
        seed: int | None = None,
        cards: tuple[Card, Card] | None = None,
        board_card: Card | None = None,
    ) -> LeducPokerState:
        """Start a new hand and return initial state."""
        if seed is not None:
            self._rng.seed(seed)
        self._state = initial_state(self._rng, cards=cards, board_card=board_card)
        return self._state

    def legal_actions(self) -> tuple[Action, ...]:
        return legal_actions_for_state(self.state)

    def step(self, action: Action | str) -> tuple[LeducPokerState, tuple[int, int], bool]:
        state, rewards, done = step_state(self.state, action)
        self._state = state
        return state, rewards, done

    def history_label(self) -> str:
        return history_label(self.state)


def card_rank(card: Card) -> Rank:
    return _CARD_RANK[card]


def rank_symbol(rank: Rank) -> str:
    return {Rank.J: "J", Rank.Q: "Q", Rank.K: "K"}[rank]


def card_label(card: Card) -> str:
    return f"{rank_symbol(card_rank(card))}{1 if card in (Card.J1, Card.Q1, Card.K1) else 2}"


def initial_state(
    rng: random.Random | None = None,
    *,
    cards: tuple[Card, Card] | None = None,
    board_card: Card | None = None,
) -> LeducPokerState:
    """Create a valid initial hand state."""
    if cards is None:
        if rng is None:
            raise ValueError("rng is required when cards are not provided.")
        shuffled = list(ALL_CARDS)
        rng.shuffle(shuffled)
        dealt = (shuffled[0], shuffled[1])
        remainder = shuffled[2:]
    else:
        dealt = cards
        if dealt[0] == dealt[1]:
            raise ValueError("Player cards must be distinct.")
        remainder = [card for card in ALL_CARDS if card not in dealt]

    if board_card is not None:
        if board_card in dealt:
            raise ValueError("Board card cannot match a player's exact card.")
        if board_card not in remainder:
            raise ValueError("Board card is not available in the deck.")
        remainder = [card for card in remainder if card != board_card]
        deck = (board_card, *remainder)
    else:
        deck = tuple(remainder)

    return LeducPokerState(
        player_cards=dealt,
        board_card=None,
        deck=tuple(deck),
        round_index=0,
        histories=((), ()),
        current_player=0,
        to_call=0,
        raises_in_round=0,
        contributions=(ANTE, ANTE),
        pot=2 * ANTE,
        terminal=False,
        folded_player=None,
    )


def all_deals() -> list[tuple[Card, Card, Card]]:
    """Enumerate all distinct (player0, player1, board) deals."""
    deals: list[tuple[Card, Card, Card]] = []
    for card0 in ALL_CARDS:
        for card1 in ALL_CARDS:
            if card1 == card0:
                continue
            for board in ALL_CARDS:
                if board in (card0, card1):
                    continue
                deals.append((card0, card1, board))
    return deals


def legal_actions_for_state(state: LeducPokerState) -> tuple[Action, ...]:
    """Return legal actions for the acting player."""
    if state.terminal:
        return ()
    if state.to_call == 0:
        return (Action.CHECK, Action.BET)
    if state.raises_in_round < MAX_RAISES_PER_ROUND:
        return (Action.CALL, Action.RAISE, Action.FOLD)
    return (Action.CALL, Action.FOLD)


def step_state(
    state: LeducPokerState, action: Action | str
) -> tuple[LeducPokerState, tuple[int, int], bool]:
    """Pure transition function for one action."""
    if state.terminal:
        raise ValueError("Cannot act in terminal state.")

    parsed_action = _parse_action(action)
    legal = legal_actions_for_state(state)
    if parsed_action not in legal:
        raise ValueError(
            f"Illegal action {parsed_action.value!r} for history "
            f"{history_label(state)!r}. Legal: {[a.value for a in legal]}"
        )

    player = state.current_player
    contributions = [state.contributions[0], state.contributions[1]]
    pot = state.pot
    to_call = state.to_call
    raises_in_round = state.raises_in_round
    round_histories = [state.histories[0], state.histories[1]]
    round_actions = round_histories[state.round_index] + (parsed_action,)
    round_histories[state.round_index] = round_actions

    if parsed_action == Action.BET:
        amount = BET_SIZES[state.round_index]
        contributions[player] += amount
        pot += amount
        to_call = amount
    elif parsed_action == Action.CALL:
        amount = to_call
        contributions[player] += amount
        pot += amount
        to_call = 0
    elif parsed_action == Action.RAISE:
        raise_size = BET_SIZES[state.round_index]
        amount = to_call + raise_size
        contributions[player] += amount
        pot += amount
        to_call = raise_size
        raises_in_round += 1
    elif parsed_action == Action.CHECK:
        pass
    elif parsed_action == Action.FOLD:
        terminal_state = LeducPokerState(
            player_cards=state.player_cards,
            board_card=state.board_card,
            deck=state.deck,
            round_index=state.round_index,
            histories=(tuple(round_histories[0]), tuple(round_histories[1])),
            current_player=-1,
            to_call=to_call,
            raises_in_round=raises_in_round,
            contributions=(contributions[0], contributions[1]),
            pot=pot,
            terminal=True,
            folded_player=player,
        )
        rewards = terminal_rewards(terminal_state)
        return terminal_state, rewards, True

    round_closed = _is_round_closed(to_call, round_actions)
    if round_closed:
        if state.round_index == 0:
            if not state.deck:
                raise RuntimeError("Deck exhausted before board reveal.")
            revealed = state.deck[0]
            next_state = LeducPokerState(
                player_cards=state.player_cards,
                board_card=revealed,
                deck=state.deck[1:],
                round_index=1,
                histories=(tuple(round_histories[0]), tuple(round_histories[1])),
                current_player=0,
                to_call=0,
                raises_in_round=0,
                contributions=(contributions[0], contributions[1]),
                pot=pot,
                terminal=False,
                folded_player=None,
            )
            return next_state, (0, 0), False

        terminal_state = LeducPokerState(
            player_cards=state.player_cards,
            board_card=state.board_card,
            deck=state.deck,
            round_index=state.round_index,
            histories=(tuple(round_histories[0]), tuple(round_histories[1])),
            current_player=-1,
            to_call=0,
            raises_in_round=raises_in_round,
            contributions=(contributions[0], contributions[1]),
            pot=pot,
            terminal=True,
            folded_player=None,
        )
        rewards = terminal_rewards(terminal_state)
        return terminal_state, rewards, True

    next_state = LeducPokerState(
        player_cards=state.player_cards,
        board_card=state.board_card,
        deck=state.deck,
        round_index=state.round_index,
        histories=(tuple(round_histories[0]), tuple(round_histories[1])),
        current_player=1 - player,
        to_call=to_call,
        raises_in_round=raises_in_round,
        contributions=(contributions[0], contributions[1]),
        pot=pot,
        terminal=False,
        folded_player=None,
    )
    return next_state, (0, 0), False


def history_label(state: LeducPokerState) -> str:
    """Compact per-round public action history label, e.g. 'kb|rc'."""
    left = "".join(_ACTION_TOKEN[action] for action in state.histories[0])
    right = "".join(_ACTION_TOKEN[action] for action in state.histories[1])
    return f"{left}|{right}"


def terminal_rewards(state: LeducPokerState) -> tuple[int, int]:
    """Net chip rewards from player 0 / player 1 perspective."""
    if not state.terminal:
        return (0, 0)

    c0, c1 = state.contributions
    if state.folded_player is not None:
        winner = 1 - state.folded_player
        if winner == 0:
            return (state.pot - c0, -c1)
        return (-c0, state.pot - c1)

    winner = _showdown_winner(state)
    if winner is None:
        # In Leduc showdown ties split the pot.
        share = state.pot // 2
        return (share - c0, share - c1)
    if winner == 0:
        return (state.pot - c0, -c1)
    return (-c0, state.pot - c1)


def _parse_action(action: Action | str) -> Action:
    if isinstance(action, Action):
        return action
    try:
        return Action(action)
    except ValueError as exc:
        raise ValueError(f"Unknown action {action!r}.") from exc


def _is_round_closed(to_call: int, round_actions: tuple[Action, ...]) -> bool:
    if to_call != 0:
        return False
    if not round_actions:
        return False
    if round_actions[-1] == Action.CALL:
        return True
    return len(round_actions) >= 2 and round_actions[-2:] == (Action.CHECK, Action.CHECK)


def _showdown_winner(state: LeducPokerState) -> int | None:
    if state.board_card is None:
        raise RuntimeError("Board card must be revealed at showdown.")

    board_rank = card_rank(state.board_card)
    p0_rank = card_rank(state.player_cards[0])
    p1_rank = card_rank(state.player_cards[1])

    p0_score = (1 if p0_rank == board_rank else 0, int(p0_rank))
    p1_score = (1 if p1_rank == board_rank else 0, int(p1_rank))

    if p0_score > p1_score:
        return 0
    if p1_score > p0_score:
        return 1
    return None
