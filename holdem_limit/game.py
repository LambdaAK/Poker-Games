"""Heads-up limit Texas Hold'em environment and hand evaluator."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import random


class Action(str, Enum):
    """Limit Hold'em actions."""

    CHECK = "check"
    BET = "bet"
    CALL = "call"
    RAISE = "raise"
    FOLD = "fold"


RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"  # clubs, diamonds, hearts, spades
SMALL_BLIND = 1
BIG_BLIND = 2
BET_SIZES = (2, 2, 4, 4)  # preflop, flop, turn, river
MAX_RAISES_PER_ROUND = 3
POSTFLOP_FIRST_PLAYER = 1

_ACTION_TOKEN = {
    Action.CHECK: "k",
    Action.BET: "b",
    Action.CALL: "c",
    Action.RAISE: "r",
    Action.FOLD: "f",
}


@dataclass(frozen=True, order=True)
class Card:
    """A standard playing card."""

    rank: int  # 2..14
    suit: int  # 0..3


@dataclass(frozen=True)
class HoldemLimitState:
    """Immutable state for one heads-up limit Hold'em hand."""

    player_hands: tuple[tuple[Card, Card], tuple[Card, Card]]
    board: tuple[Card, ...]
    deck: tuple[Card, ...]
    round_index: int  # 0: preflop, 1: flop, 2: turn, 3: river
    histories: tuple[tuple[Action, ...], tuple[Action, ...], tuple[Action, ...], tuple[Action, ...]]
    current_player: int
    to_call: int
    raises_in_round: int
    contributions: tuple[int, int]
    pot: int
    terminal: bool
    folded_player: int | None


class HoldemLimitGame:
    """Stateful wrapper around pure Hold'em transitions."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._state: HoldemLimitState | None = None

    @property
    def state(self) -> HoldemLimitState:
        if self._state is None:
            raise RuntimeError("Game has not been reset. Call reset() first.")
        return self._state

    def reset(
        self,
        *,
        seed: int | None = None,
        hands: tuple[tuple[Card, Card], tuple[Card, Card]] | None = None,
        board_cards: tuple[Card, Card, Card, Card, Card] | None = None,
    ) -> HoldemLimitState:
        """Start a new hand."""
        if seed is not None:
            self._rng.seed(seed)
        self._state = initial_state(self._rng, hands=hands, board_cards=board_cards)
        return self._state

    def legal_actions(self) -> tuple[Action, ...]:
        return legal_actions_for_state(self.state)

    def step(self, action: Action | str) -> tuple[HoldemLimitState, tuple[int, int], bool]:
        state, rewards, done = step_state(self.state, action)
        self._state = state
        return state, rewards, done

    def history_label(self) -> str:
        return history_label(self.state)


def rank_char(rank: int) -> str:
    if rank < 2 or rank > 14:
        raise ValueError(f"rank out of range: {rank}")
    return RANK_CHARS[rank - 2]


def suit_char(suit: int) -> str:
    if suit < 0 or suit >= len(SUIT_CHARS):
        raise ValueError(f"suit out of range: {suit}")
    return SUIT_CHARS[suit]


def card_label(card: Card) -> str:
    return f"{rank_char(card.rank)}{suit_char(card.suit)}"


def parse_card(label: str) -> Card:
    """Parse short label like 'As' or 'Td'."""
    if len(label) != 2:
        raise ValueError(f"Invalid card label {label!r}.")
    rank_symbol, suit_symbol = label[0].upper(), label[1].lower()
    if rank_symbol not in RANK_CHARS:
        raise ValueError(f"Unknown rank in {label!r}.")
    if suit_symbol not in SUIT_CHARS:
        raise ValueError(f"Unknown suit in {label!r}.")
    rank = RANK_CHARS.index(rank_symbol) + 2
    suit = SUIT_CHARS.index(suit_symbol)
    return Card(rank=rank, suit=suit)


def standard_deck() -> tuple[Card, ...]:
    cards: list[Card] = []
    for suit in range(4):
        for rank in range(2, 15):
            cards.append(Card(rank=rank, suit=suit))
    return tuple(cards)


ALL_CARDS = standard_deck()


def initial_state(
    rng: random.Random | None = None,
    *,
    hands: tuple[tuple[Card, Card], tuple[Card, Card]] | None = None,
    board_cards: tuple[Card, Card, Card, Card, Card] | None = None,
) -> HoldemLimitState:
    """Create initial game state with blinds posted."""
    if hands is None:
        if rng is None:
            raise ValueError("rng is required when hands are not provided.")
        shuffled = list(ALL_CARDS)
        rng.shuffle(shuffled)
        p0 = (shuffled[0], shuffled[1])
        p1 = (shuffled[2], shuffled[3])
        hands = (p0, p1)
        remainder = shuffled[4:]
    else:
        _validate_hands(hands)
        used = {hands[0][0], hands[0][1], hands[1][0], hands[1][1]}
        remainder = [card for card in ALL_CARDS if card not in used]
        if rng is not None:
            rng.shuffle(remainder)

    if board_cards is not None:
        if len(board_cards) != 5:
            raise ValueError("board_cards must contain exactly 5 cards.")
        used = {hands[0][0], hands[0][1], hands[1][0], hands[1][1]}
        for card in board_cards:
            if card in used:
                raise ValueError("Board card overlaps with private cards.")
            used.add(card)
        remaining_set = [card for card in remainder if card not in board_cards]
        deck = tuple(board_cards) + tuple(remaining_set)
    else:
        deck = tuple(remainder)

    return HoldemLimitState(
        player_hands=hands,
        board=(),
        deck=deck,
        round_index=0,
        histories=((), (), (), ()),
        current_player=0,  # small blind acts first preflop
        to_call=BIG_BLIND - SMALL_BLIND,
        raises_in_round=0,
        contributions=(SMALL_BLIND, BIG_BLIND),
        pot=SMALL_BLIND + BIG_BLIND,
        terminal=False,
        folded_player=None,
    )


def legal_actions_for_state(state: HoldemLimitState) -> tuple[Action, ...]:
    """Return legal actions for acting player."""
    if state.terminal:
        return ()
    if state.to_call == 0:
        return (Action.CHECK, Action.BET)
    if state.raises_in_round < MAX_RAISES_PER_ROUND:
        return (Action.CALL, Action.RAISE, Action.FOLD)
    return (Action.CALL, Action.FOLD)


def step_state(
    state: HoldemLimitState, action: Action | str
) -> tuple[HoldemLimitState, tuple[int, int], bool]:
    """Pure transition for one action."""
    if state.terminal:
        raise ValueError("Cannot act in terminal state.")

    parsed = _parse_action(action)
    legal = legal_actions_for_state(state)
    if parsed not in legal:
        raise ValueError(
            f"Illegal action {parsed.value!r} for history "
            f"{history_label(state)!r}. Legal: {[a.value for a in legal]}"
        )

    player = state.current_player
    contributions = [state.contributions[0], state.contributions[1]]
    pot = state.pot
    to_call = state.to_call
    raises = state.raises_in_round
    round_histories = [state.histories[0], state.histories[1], state.histories[2], state.histories[3]]
    round_actions = round_histories[state.round_index] + (parsed,)
    round_histories[state.round_index] = round_actions

    was_blind_completion_call = (
        state.round_index == 0
        and state.histories[0] == ()
        and state.to_call == BIG_BLIND - SMALL_BLIND
        and state.raises_in_round == 0
        and parsed == Action.CALL
    )

    if parsed == Action.BET:
        amount = BET_SIZES[state.round_index]
        contributions[player] += amount
        pot += amount
        to_call = amount
    elif parsed == Action.CALL:
        amount = to_call
        contributions[player] += amount
        pot += amount
        to_call = 0
    elif parsed == Action.RAISE:
        raise_size = BET_SIZES[state.round_index]
        amount = to_call + raise_size
        contributions[player] += amount
        pot += amount
        to_call = raise_size
        raises += 1
    elif parsed == Action.CHECK:
        pass
    elif parsed == Action.FOLD:
        terminal = HoldemLimitState(
            player_hands=state.player_hands,
            board=state.board,
            deck=state.deck,
            round_index=state.round_index,
            histories=(
                tuple(round_histories[0]),
                tuple(round_histories[1]),
                tuple(round_histories[2]),
                tuple(round_histories[3]),
            ),
            current_player=-1,
            to_call=to_call,
            raises_in_round=raises,
            contributions=(contributions[0], contributions[1]),
            pot=pot,
            terminal=True,
            folded_player=player,
        )
        rewards = terminal_rewards(terminal)
        return terminal, rewards, True

    round_closed = _is_round_closed(to_call, round_actions)
    if was_blind_completion_call:
        round_closed = False
    if state.round_index == 0 and round_actions == (Action.CALL, Action.CHECK):
        round_closed = True
    if round_closed:
        if state.round_index < 3:
            next_round = state.round_index + 1
            next_board, next_deck = _deal_board_for_round(next_round, state.board, state.deck)
            next_state = HoldemLimitState(
                player_hands=state.player_hands,
                board=next_board,
                deck=next_deck,
                round_index=next_round,
                histories=(
                    tuple(round_histories[0]),
                    tuple(round_histories[1]),
                    tuple(round_histories[2]),
                    tuple(round_histories[3]),
                ),
                current_player=POSTFLOP_FIRST_PLAYER,
                to_call=0,
                raises_in_round=0,
                contributions=(contributions[0], contributions[1]),
                pot=pot,
                terminal=False,
                folded_player=None,
            )
            return next_state, (0, 0), False

        terminal = HoldemLimitState(
            player_hands=state.player_hands,
            board=state.board,
            deck=state.deck,
            round_index=state.round_index,
            histories=(
                tuple(round_histories[0]),
                tuple(round_histories[1]),
                tuple(round_histories[2]),
                tuple(round_histories[3]),
            ),
            current_player=-1,
            to_call=0,
            raises_in_round=raises,
            contributions=(contributions[0], contributions[1]),
            pot=pot,
            terminal=True,
            folded_player=None,
        )
        rewards = terminal_rewards(terminal)
        return terminal, rewards, True

    next_state = HoldemLimitState(
        player_hands=state.player_hands,
        board=state.board,
        deck=state.deck,
        round_index=state.round_index,
        histories=(
            tuple(round_histories[0]),
            tuple(round_histories[1]),
            tuple(round_histories[2]),
            tuple(round_histories[3]),
        ),
        current_player=1 - player,
        to_call=to_call,
        raises_in_round=raises,
        contributions=(contributions[0], contributions[1]),
        pot=pot,
        terminal=False,
        folded_player=None,
    )
    return next_state, (0, 0), False


def history_label(state: HoldemLimitState) -> str:
    """Compact per-round history, e.g. 'brc|kk|b|c'."""
    parts: list[str] = []
    for actions in state.histories:
        if not actions:
            parts.append("-")
            continue
        parts.append("".join(_ACTION_TOKEN[action] for action in actions))
    return "|".join(parts)


def terminal_rewards(state: HoldemLimitState) -> tuple[int, int]:
    """Net chip rewards from player 0 and player 1 perspective."""
    if not state.terminal:
        return (0, 0)

    c0, c1 = state.contributions
    if state.folded_player is not None:
        winner = 1 - state.folded_player
        if winner == 0:
            return (state.pot - c0, -c1)
        return (-c0, state.pot - c1)

    winner = showdown_winner(state.player_hands, state.board)
    if winner is None:
        share = state.pot // 2
        return (share - c0, share - c1)
    if winner == 0:
        return (state.pot - c0, -c1)
    return (-c0, state.pot - c1)


def showdown_winner(
    hands: tuple[tuple[Card, Card], tuple[Card, Card]], board: tuple[Card, ...]
) -> int | None:
    """Return 0/1 winner, or None for tie."""
    if len(board) != 5:
        raise ValueError("Showdown requires 5 board cards.")
    p0_rank = evaluate_seven(hands[0] + board)
    p1_rank = evaluate_seven(hands[1] + board)
    if p0_rank > p1_rank:
        return 0
    if p1_rank > p0_rank:
        return 1
    return None


def evaluate_seven(cards: tuple[Card, ...] | list[Card]) -> tuple[int, tuple[int, ...]]:
    """Return best rank for 7 cards."""
    if len(cards) != 7:
        raise ValueError("evaluate_seven requires 7 cards.")
    best: tuple[int, tuple[int, ...]] | None = None
    for combo in combinations(cards, 5):
        rank = evaluate_five(combo)
        if best is None or rank > best:
            best = rank
    if best is None:
        raise RuntimeError("No 5-card combinations were generated.")
    return best


def evaluate_five(cards: tuple[Card, ...] | list[Card]) -> tuple[int, tuple[int, ...]]:
    """Evaluate 5-card hand; higher tuple is stronger."""
    if len(cards) != 5:
        raise ValueError("evaluate_five requires 5 cards.")

    ranks = sorted((card.rank for card in cards), reverse=True)
    suits = [card.suit for card in cards]
    rank_counts = Counter(ranks)
    counts_desc = sorted(
        ((count, rank) for rank, count in rank_counts.items()),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)
    is_straight = straight_high is not None

    # 8: straight flush
    if is_straight and is_flush:
        return (8, (straight_high,))

    # 7: four of a kind
    if counts_desc[0][0] == 4:
        quad_rank = counts_desc[0][1]
        kicker = counts_desc[1][1]
        return (7, (quad_rank, kicker))

    # 6: full house
    if counts_desc[0][0] == 3 and counts_desc[1][0] == 2:
        trips_rank = counts_desc[0][1]
        pair_rank = counts_desc[1][1]
        return (6, (trips_rank, pair_rank))

    # 5: flush
    if is_flush:
        return (5, tuple(sorted(ranks, reverse=True)))

    # 4: straight
    if is_straight:
        return (4, (straight_high,))

    # 3: trips
    if counts_desc[0][0] == 3:
        trips_rank = counts_desc[0][1]
        kickers = sorted((rank for rank, count in rank_counts.items() if count == 1), reverse=True)
        return (3, (trips_rank, kickers[0], kickers[1]))

    # 2: two pair
    if counts_desc[0][0] == 2 and counts_desc[1][0] == 2:
        high_pair = max(counts_desc[0][1], counts_desc[1][1])
        low_pair = min(counts_desc[0][1], counts_desc[1][1])
        kicker = counts_desc[2][1]
        return (2, (high_pair, low_pair, kicker))

    # 1: one pair
    if counts_desc[0][0] == 2:
        pair_rank = counts_desc[0][1]
        kickers = sorted((rank for rank, count in rank_counts.items() if count == 1), reverse=True)
        return (1, (pair_rank, kickers[0], kickers[1], kickers[2]))

    # 0: high card
    return (0, tuple(sorted(ranks, reverse=True)))


def _straight_high(ranks: list[int]) -> int | None:
    unique = sorted(set(ranks))
    if 14 in unique:
        unique.insert(0, 1)  # wheel support
    best: int | None = None
    run = 1
    for i in range(1, len(unique)):
        if unique[i] == unique[i - 1] + 1:
            run += 1
            if run >= 5:
                best = unique[i]
        elif unique[i] != unique[i - 1]:
            run = 1
    return best


def _parse_action(action: Action | str) -> Action:
    if isinstance(action, Action):
        return action
    try:
        return Action(action)
    except ValueError as exc:
        raise ValueError(f"Unknown action {action!r}.") from exc


def _validate_hands(hands: tuple[tuple[Card, Card], tuple[Card, Card]]) -> None:
    if len(hands) != 2 or len(hands[0]) != 2 or len(hands[1]) != 2:
        raise ValueError("hands must be ((card, card), (card, card)).")
    cards = [hands[0][0], hands[0][1], hands[1][0], hands[1][1]]
    if len(set(cards)) != 4:
        raise ValueError("Private cards must be distinct.")


def _is_round_closed(to_call: int, round_actions: tuple[Action, ...]) -> bool:
    if to_call != 0:
        return False
    if not round_actions:
        return False
    if round_actions[-1] == Action.CALL:
        return True
    return len(round_actions) >= 2 and round_actions[-2:] == (Action.CHECK, Action.CHECK)


def _deal_board_for_round(
    round_index: int, board: tuple[Card, ...], deck: tuple[Card, ...]
) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
    if round_index == 1:
        if len(deck) < 3:
            raise RuntimeError("Deck exhausted before flop.")
        return board + (deck[0], deck[1], deck[2]), deck[3:]
    if round_index in (2, 3):
        if len(deck) < 1:
            raise RuntimeError("Deck exhausted before board card.")
        return board + (deck[0],), deck[1:]
    raise RuntimeError(f"Unknown round index {round_index}.")
