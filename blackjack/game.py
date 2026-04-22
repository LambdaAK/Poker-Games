"""Blackjack environment and hand evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random


class Action(str, Enum):
    """Player actions in blackjack."""

    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"


@dataclass(frozen=True, order=True)
class Card:
    """A standard playing card."""

    rank: int  # 2..14 where 11/12/13 are J/Q/K and 14 is Ace
    suit: int  # 0..3


RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
DEALER_HITS_SOFT_17 = False


def standard_deck() -> tuple[Card, ...]:
    """Return a standard 52-card deck."""
    cards: list[Card] = []
    for suit in range(4):
        for rank in range(2, 15):
            cards.append(Card(rank=rank, suit=suit))
    return tuple(cards)


ALL_CARDS = standard_deck()
_ACTION_TOKEN = {
    Action.HIT: "h",
    Action.STAND: "s",
    Action.DOUBLE: "d",
}


@dataclass(frozen=True)
class BlackjackState:
    """Immutable state for one blackjack hand."""

    player_hand: tuple[Card, ...]
    dealer_hand: tuple[Card, ...]
    deck: tuple[Card, ...]
    action_history: tuple[Action, ...]
    current_player: int
    can_double: bool
    stake: int
    terminal: bool
    rewards: tuple[float, float]


class BlackjackGame:
    """Stateful blackjack environment with an automatic dealer."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._state: BlackjackState | None = None

    @property
    def state(self) -> BlackjackState:
        """Return the current state."""
        if self._state is None:
            raise RuntimeError("Game has not been reset. Call reset() first.")
        return self._state

    def reset(
        self,
        *,
        seed: int | None = None,
        player_hand: tuple[Card, Card] | None = None,
        dealer_hand: tuple[Card, Card] | None = None,
        deck: tuple[Card, ...] | None = None,
    ) -> BlackjackState:
        """Start a new hand."""
        if seed is not None:
            self._rng.seed(seed)
        self._state = initial_state(
            self._rng,
            player_hand=player_hand,
            dealer_hand=dealer_hand,
            deck=deck,
        )
        return self._state

    def legal_actions(self) -> tuple[Action, ...]:
        """Return legal actions for the current player."""
        return legal_actions_for_state(self.state)

    def step(self, action: Action | str) -> tuple[BlackjackState, tuple[float, float], bool]:
        """Apply one player action."""
        state = self.state
        if state.terminal:
            raise ValueError("Cannot act in a terminal state.")

        parsed = _parse_action(action)
        legal = legal_actions_for_state(state)
        if parsed not in legal:
            raise ValueError(
                f"Illegal action {parsed.value!r} for history "
                f"{history_label(state)!r}. Legal: {[a.value for a in legal]}"
            )

        player_hand = list(state.player_hand)
        dealer_hand = list(state.dealer_hand)
        deck = list(state.deck)
        history = state.action_history + (parsed,)
        stake = state.stake

        if parsed == Action.HIT:
            player_hand.append(_draw_card(deck))
            total, _ = hand_value(player_hand)
            if total > 21:
                next_state = _terminal_state(
                    player_hand=tuple(player_hand),
                    dealer_hand=tuple(dealer_hand),
                    deck=tuple(deck),
                    history=history,
                    stake=stake,
                )
                self._state = next_state
                return next_state, next_state.rewards, True
            if total == 21:
                dealer_hand, deck = _play_dealer(dealer_hand, deck)
                next_state = _terminal_state(
                    player_hand=tuple(player_hand),
                    dealer_hand=tuple(dealer_hand),
                    deck=tuple(deck),
                    history=history,
                    stake=stake,
                )
                self._state = next_state
                return next_state, next_state.rewards, True
            next_state = BlackjackState(
                player_hand=tuple(player_hand),
                dealer_hand=tuple(dealer_hand),
                deck=tuple(deck),
                action_history=history,
                current_player=0,
                can_double=False,
                stake=stake,
                terminal=False,
                rewards=(0.0, 0.0),
            )
            self._state = next_state
            return next_state, (0.0, 0.0), False

        if parsed == Action.DOUBLE:
            stake *= 2
            player_hand.append(_draw_card(deck))
            total, _ = hand_value(player_hand)
            if total > 21:
                next_state = _terminal_state(
                    player_hand=tuple(player_hand),
                    dealer_hand=tuple(dealer_hand),
                    deck=tuple(deck),
                    history=history,
                    stake=stake,
                )
                self._state = next_state
                return next_state, next_state.rewards, True
            dealer_hand, deck = _play_dealer(dealer_hand, deck)
            next_state = _terminal_state(
                player_hand=tuple(player_hand),
                dealer_hand=tuple(dealer_hand),
                deck=tuple(deck),
                history=history,
                stake=stake,
            )
            self._state = next_state
            return next_state, next_state.rewards, True

        dealer_hand, deck = _play_dealer(dealer_hand, deck)
        next_state = _terminal_state(
            player_hand=tuple(player_hand),
            dealer_hand=tuple(dealer_hand),
            deck=tuple(deck),
            history=history,
            stake=stake,
        )
        self._state = next_state
        return next_state, next_state.rewards, True

    def history_label(self) -> str:
        """Compact action history label."""
        return history_label(self.state)


def card_label(card: Card) -> str:
    """Return a short text label like 'As' or 'Td'."""
    return f"{rank_symbol(card.rank)}{SUIT_CHARS[card.suit]}"


def rank_symbol(rank: int) -> str:
    """Return the printable symbol for a rank."""
    if rank < 2 or rank > 14:
        raise ValueError(f"rank out of range: {rank}")
    return RANK_CHARS[rank - 2]


def parse_card(label: str) -> Card:
    """Parse a short card label like 'As' or 'Td'."""
    text = label.strip()
    if len(text) == 3 and text.startswith("10"):
        rank_text = "T"
        suit_text = text[2].lower()
    elif len(text) == 2:
        rank_text = text[0].upper()
        suit_text = text[1].lower()
    else:
        raise ValueError(f"Invalid card label {label!r}.")

    if rank_text not in RANK_CHARS:
        raise ValueError(f"Unknown rank in {label!r}.")
    if suit_text not in SUIT_CHARS:
        raise ValueError(f"Unknown suit in {label!r}.")

    rank = RANK_CHARS.index(rank_text) + 2
    suit = SUIT_CHARS.index(suit_text)
    return Card(rank=rank, suit=suit)


def blackjack_value(rank: int) -> int:
    """Return the blackjack value of a rank."""
    if rank < 2 or rank > 14:
        raise ValueError(f"rank out of range: {rank}")
    if rank == 14:
        return 11
    return min(rank, 10)


def hand_value(hand: tuple[Card, ...] | list[Card]) -> tuple[int, bool]:
    """Return the best blackjack total and whether it is soft."""
    total = 0
    aces = 0
    for card in hand:
        if card.rank == 14:
            aces += 1
            total += 1
        else:
            total += min(card.rank, 10)

    soft = False
    while aces > 0 and total + 10 <= 21:
        total += 10
        aces -= 1
        soft = True

    return total, soft


def is_blackjack(hand: tuple[Card, ...] | list[Card]) -> bool:
    """Return True if the hand is a natural blackjack."""
    total, _ = hand_value(hand)
    return len(hand) == 2 and total == 21


def legal_actions_for_state(state: BlackjackState) -> tuple[Action, ...]:
    """Return legal actions for the player."""
    if state.terminal:
        return ()

    total, _ = hand_value(state.player_hand)
    if total > 21:
        return ()
    if total == 21:
        return (Action.STAND,)

    legal = [Action.HIT, Action.STAND]
    if state.can_double and len(state.player_hand) == 2:
        legal.append(Action.DOUBLE)
    return tuple(legal)


def history_label(state: BlackjackState) -> str:
    """Compact player-action history label."""
    return "".join(_ACTION_TOKEN[action] for action in state.action_history)


def initial_state(
    rng: random.Random | None = None,
    *,
    player_hand: tuple[Card, Card] | None = None,
    dealer_hand: tuple[Card, Card] | None = None,
    deck: tuple[Card, ...] | None = None,
) -> BlackjackState:
    """Create a valid initial blackjack state."""
    if (player_hand is None) != (dealer_hand is None):
        raise ValueError("Provide both player_hand and dealer_hand, or neither.")

    if player_hand is None:
        if rng is None:
            raise ValueError("rng is required when hands are not provided.")
        shuffled = list(ALL_CARDS)
        rng.shuffle(shuffled)
        player_hand = (shuffled[0], shuffled[1])
        dealer_hand = (shuffled[2], shuffled[3])
        remainder = shuffled[4:]
    else:
        _validate_hand(player_hand)
        _validate_hand(dealer_hand)
        used = {player_hand[0], player_hand[1], dealer_hand[0], dealer_hand[1]}
        if len(used) != 4:
            raise ValueError("Player and dealer cards must be distinct.")
        remainder = [card for card in ALL_CARDS if card not in used]
        if deck is not None:
            _validate_deck(deck, used)
            remainder = list(deck)
        elif rng is not None:
            rng.shuffle(remainder)

    state = BlackjackState(
        player_hand=player_hand,
        dealer_hand=dealer_hand,
        deck=tuple(remainder),
        action_history=(),
        current_player=0,
        can_double=True,
        stake=1,
        terminal=False,
        rewards=(0.0, 0.0),
    )
    if is_blackjack(player_hand) or is_blackjack(dealer_hand):
        state = _terminal_state(
            player_hand=player_hand,
            dealer_hand=dealer_hand,
            deck=tuple(remainder),
            history=(),
            stake=1,
        )
    return state


def _validate_hand(hand: tuple[Card, Card]) -> None:
    if len(hand) != 2:
        raise ValueError("Each initial hand must contain exactly two cards.")
    if hand[0] == hand[1]:
        raise ValueError("A hand cannot contain duplicate cards.")


def _validate_deck(deck: tuple[Card, ...], used: set[Card]) -> None:
    seen: set[Card] = set()
    for card in deck:
        if card in used:
            raise ValueError("Deck overlaps with initial hands.")
        if card in seen:
            raise ValueError("Deck contains duplicate cards.")
        seen.add(card)


def _parse_action(action: Action | str) -> Action:
    if isinstance(action, Action):
        return action
    try:
        return Action(action)
    except ValueError as exc:
        raise ValueError(f"Unknown action {action!r}.") from exc


def _draw_card(deck: list[Card]) -> Card:
    if not deck:
        raise RuntimeError("Deck exhausted.")
    return deck.pop(0)


def _play_dealer(
    dealer_hand: list[Card], deck: list[Card]
) -> tuple[list[Card], list[Card]]:
    while True:
        total, soft = hand_value(dealer_hand)
        if total > 21:
            return dealer_hand, deck
        if total < 17:
            dealer_hand.append(_draw_card(deck))
            continue
        if total == 17 and soft and DEALER_HITS_SOFT_17:
            dealer_hand.append(_draw_card(deck))
            continue
        return dealer_hand, deck


def _terminal_state(
    *,
    player_hand: tuple[Card, ...],
    dealer_hand: tuple[Card, ...],
    deck: tuple[Card, ...],
    history: tuple[Action, ...],
    stake: int,
) -> BlackjackState:
    rewards = terminal_rewards(player_hand=player_hand, dealer_hand=dealer_hand, stake=stake)
    return BlackjackState(
        player_hand=player_hand,
        dealer_hand=dealer_hand,
        deck=deck,
        action_history=history,
        current_player=-1,
        can_double=False,
        stake=stake,
        terminal=True,
        rewards=rewards,
    )


def terminal_rewards(
    *,
    player_hand: tuple[Card, ...],
    dealer_hand: tuple[Card, ...],
    stake: int,
) -> tuple[float, float]:
    """Return net chip rewards from the player's perspective."""
    player_total, _ = hand_value(player_hand)
    dealer_total, _ = hand_value(dealer_hand)
    player_blackjack = is_blackjack(player_hand)
    dealer_blackjack = is_blackjack(dealer_hand)

    if player_blackjack and dealer_blackjack:
        return (0.0, 0.0)
    if player_blackjack:
        payout = 1.5 * stake
        return (payout, -payout)
    if dealer_blackjack:
        payout = 1.5 * stake
        return (-payout, payout)
    if player_total > 21:
        return (-float(stake), float(stake))
    if dealer_total > 21:
        return (float(stake), -float(stake))
    if player_total > dealer_total:
        return (float(stake), -float(stake))
    if dealer_total > player_total:
        return (-float(stake), float(stake))
    return (0.0, 0.0)
