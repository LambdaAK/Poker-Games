"""Information-state abstraction for limit Hold'em."""

from __future__ import annotations

from collections import Counter

from .game import Card, HoldemLimitState, evaluate_five, history_label


def info_state_key(state: HoldemLimitState, player: int) -> str:
    """Bucketed information state for learning algorithms."""
    hole = state.player_hands[player]
    board = state.board
    private = _private_bucket(hole)
    post = _postflop_bucket(hole, board)
    hist = history_label(state)
    return (
        f"r{state.round_index}|{private}|{post}|"
        f"tc{state.to_call}|ra{state.raises_in_round}|h{hist}"
    )


def _private_bucket(hole: tuple[Card, Card]) -> str:
    c1, c2 = hole
    high = max(c1.rank, c2.rank)
    low = min(c1.rank, c2.rank)
    suited = int(c1.suit == c2.suit)
    gap = abs(c1.rank - c2.rank)

    if c1.rank == c2.rank:
        if high >= 11:
            return "pf_pair_hi"
        if high >= 7:
            return "pf_pair_mid"
        return "pf_pair_lo"
    if high >= 13 and low >= 10:
        return f"pf_broadway_s{suited}"
    if suited and gap <= 1:
        return "pf_suited_conn"
    if suited and gap <= 3:
        return "pf_suited_gap"
    if gap <= 1:
        return "pf_off_conn"
    if high >= 12:
        return "pf_high"
    return "pf_trash"


def _postflop_bucket(hole: tuple[Card, Card], board: tuple[Card, ...]) -> str:
    if not board:
        return "pre"
    known = hole + board

    if len(board) == 5:
        score = evaluate_five(_best_five_from_seven(known))
        return f"rv_cat{score[0]}"

    rank_counts = Counter(card.rank for card in known)
    suit_counts = Counter(card.suit for card in known)
    max_mult = max(rank_counts.values())
    pair_count = sum(1 for count in rank_counts.values() if count == 2)
    flush_draw = max(suit_counts.values()) >= 4
    straight_draw = _long_straight_draw(known)
    overcard = int(max(hole[0].rank, hole[1].rank) >= 11)

    if max_mult >= 3:
        made = "m3"
    elif pair_count >= 2:
        made = "m2p"
    elif pair_count == 1:
        made = "m1p"
    else:
        made = "m0"

    draw = "d2" if (flush_draw and straight_draw) else "d1" if (flush_draw or straight_draw) else "d0"
    return f"{len(board)}c_{made}_{draw}_o{overcard}"


def _best_five_from_seven(cards: tuple[Card, ...]) -> tuple[Card, Card, Card, Card, Card]:
    # Minimal utility to avoid re-implementing ranking categories in the abstraction.
    best = None
    best_score = None
    from itertools import combinations

    for combo in combinations(cards, 5):
        score = evaluate_five(combo)
        if best_score is None or score > best_score:
            best_score = score
            best = combo
    if best is None:
        raise RuntimeError("No five-card combo generated.")
    return best  # type: ignore[return-value]


def _long_straight_draw(cards: tuple[Card, ...]) -> bool:
    ranks = {card.rank for card in cards}
    if 14 in ranks:
        ranks.add(1)
    ordered = sorted(ranks)
    if not ordered:
        return False
    run = 1
    for idx in range(1, len(ordered)):
        if ordered[idx] == ordered[idx - 1] + 1:
            run += 1
            if run >= 4:
                return True
        elif ordered[idx] != ordered[idx - 1]:
            run = 1
    return False
