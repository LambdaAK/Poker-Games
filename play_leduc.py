"""Play Leduc poker from the terminal (human vs random/CFR/RL bot)."""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Callable

from leduc_poker.cfr import AverageStrategyPolicy, info_state_key as cfr_info_state_key
from leduc_poker.game import Action, LeducPokerGame, card_label, card_rank, rank_symbol
from leduc_poker.rl import TabularSoftmaxPolicy, info_state_key as rl_info_state_key


class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def paint(self, text: str, *codes: str) -> str:
        if not self.enabled or not codes:
            return text
        return "".join(codes) + text + self.RESET


ACTION_ALIASES = {
    "k": Action.CHECK,
    "check": Action.CHECK,
    "b": Action.BET,
    "bet": Action.BET,
    "c": Action.CALL,
    "call": Action.CALL,
    "r": Action.RAISE,
    "raise": Action.RAISE,
    "f": Action.FOLD,
    "fold": Action.FOLD,
}
ACTION_KEYS = {
    Action.CHECK: "k",
    Action.BET: "b",
    Action.CALL: "c",
    Action.RAISE: "r",
    Action.FOLD: "f",
}
QUIT_WORDS = {"q", "quit", "exit"}


def rank_of_card_text(card) -> str:
    return rank_symbol(card_rank(card))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rl-policy", type=str, default=None, help="Path to RL policy JSON")
    group.add_argument("--cfr-policy", type=str, default=None, help="Path to CFR policy JSON")
    parser.add_argument(
        "--bot-greedy",
        action="store_true",
        help="Use greedy selection for loaded bot policy",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--no-clear", action="store_true", help="Disable screen clearing")
    parser.add_argument(
        "--bot-delay",
        type=float,
        default=0.35,
        help="Bot action delay in seconds (default: 0.35)",
    )
    return parser.parse_args()


def choose_bot_action(
    *,
    state,
    legal: tuple[Action, ...],
    rng: random.Random,
    rl_policy: TabularSoftmaxPolicy | None,
    cfr_policy: AverageStrategyPolicy | None,
    rl_key_fn: Callable,
    cfr_key_fn: Callable,
    greedy: bool,
) -> Action:
    if rl_policy is None and cfr_policy is None:
        return rng.choice(list(legal))
    if rl_policy is not None:
        key = rl_key_fn(state, 1)
        if greedy:
            return rl_policy.greedy_action(key, legal)
        return rl_policy.sample_action(key, legal, rng)
    key = cfr_key_fn(state, 1)
    if greedy:
        return cfr_policy.greedy_action(key, legal)
    return cfr_policy.sample_action(key, legal, rng)


def should_use_color(no_color: bool) -> bool:
    return (not no_color) and sys.stdout.isatty()


def clear_screen(enabled: bool) -> None:
    if enabled:
        print("\033[2J\033[H", end="")


def format_score(score: int) -> str:
    return f"{score:+d}"


def round_name(round_index: int) -> str:
    return "Preflop" if round_index == 0 else "Flop"


def format_history(label: str) -> str:
    preflop, flop = label.split("|", 1)
    pre_text = preflop if preflop else "-"
    flop_text = flop if flop else "-"
    return f"pre:{pre_text}  flop:{flop_text}"


def legal_action_line(legal: tuple[Action, ...], style: Style) -> str:
    parts: list[str] = []
    for action in legal:
        key = ACTION_KEYS[action]
        token = f"[{key}] {action.value}"
        if action in (Action.BET, Action.RAISE):
            parts.append(style.paint(token, Style.YELLOW, Style.BOLD))
        elif action == Action.FOLD:
            parts.append(style.paint(token, Style.RED))
        elif action == Action.CALL:
            parts.append(style.paint(token, Style.GREEN))
        else:
            parts.append(style.paint(token, Style.CYAN))
    parts.append(style.paint("[q] quit", Style.DIM))
    return "  ".join(parts)


def render_table(
    *,
    style: Style,
    clear_enabled: bool,
    matchup: str,
    hand_number: int,
    bankroll: int,
    your_card: str,
    board_text: str,
    round_text: str,
    pot: int,
    to_call: int,
    history_text: str,
    legal: tuple[Action, ...],
    your_turn: bool,
    status_line: str,
    error_line: str | None,
) -> None:
    clear_screen(clear_enabled)
    title = style.paint("Leduc Poker", Style.BOLD, Style.CYAN)
    score = style.paint(format_score(bankroll), Style.BOLD, Style.GREEN if bankroll >= 0 else Style.RED)
    print(f"{title}  ({matchup})")
    print(f"Hand {hand_number}    Score {score}")
    print(style.paint("-" * 58, Style.DIM))
    print(f"Your card: {style.paint(your_card, Style.BOLD, Style.YELLOW)}")
    print(
        f"Board: {style.paint(board_text, Style.BOLD)}    "
        f"Round: {style.paint(round_text, Style.BOLD)}"
    )
    print(f"Pot: {style.paint(str(pot), Style.BOLD)}    To call: {style.paint(str(to_call), Style.BOLD)}")
    print(f"History: {style.paint(history_text, Style.DIM)}")
    print(style.paint("-" * 58, Style.DIM))
    if status_line:
        print(status_line)
    if error_line:
        print(style.paint(error_line, Style.RED))
    if your_turn:
        print(legal_action_line(legal, style))
        print(style.paint("Action >", Style.BOLD), end=" ")
    else:
        print(style.paint("Bot is acting...", Style.MAGENTA))


def read_action_with_ui(
    *,
    style: Style,
    render_fn: Callable[[str | None], None],
    legal: tuple[Action, ...],
) -> Action:
    error: str | None = None
    while True:
        render_fn(error)
        raw = input().strip().lower()
        if raw in QUIT_WORDS:
            raise KeyboardInterrupt
        action = ACTION_ALIASES.get(raw)
        if action in legal:
            return action
        error = f"Invalid action. Legal: {', '.join(action.value for action in legal)}"


def show_hand_result(
    *,
    style: Style,
    clear_enabled: bool,
    matchup: str,
    hand_number: int,
    bankroll: int,
    rewards: tuple[int, int],
    your_card: str,
    bot_card: str,
    board_text: str,
    history_text: str,
) -> bool:
    clear_screen(clear_enabled)
    title = style.paint("Hand Complete", Style.BOLD, Style.CYAN)
    print(f"{title}  ({matchup})")
    print(style.paint("-" * 58, Style.DIM))
    print(f"Hand {hand_number}   Score {style.paint(format_score(bankroll), Style.BOLD)}")
    print(f"Your card: {style.paint(your_card, Style.YELLOW, Style.BOLD)}")
    print(f"Bot card:  {style.paint(bot_card, Style.MAGENTA, Style.BOLD)}")
    print(f"Board:     {style.paint(board_text, Style.BOLD)}")
    print(f"History:   {style.paint(history_text, Style.DIM)}")
    print(style.paint("-" * 58, Style.DIM))

    if rewards[0] > 0:
        result = style.paint(f"You win {rewards[0]} chips", Style.GREEN, Style.BOLD)
    elif rewards[0] < 0:
        result = style.paint(f"You lose {-rewards[0]} chips", Style.RED, Style.BOLD)
    else:
        result = style.paint("Split pot", Style.CYAN, Style.BOLD)
    print(result)
    print(style.paint("Press Enter for next hand, or type q to quit.", Style.DIM))
    raw = input().strip().lower()
    return raw not in QUIT_WORDS


def play_hand(
    *,
    game: LeducPokerGame,
    rng: random.Random,
    rl_policy: TabularSoftmaxPolicy | None,
    cfr_policy: AverageStrategyPolicy | None,
    greedy: bool,
    style: Style,
    clear_enabled: bool,
    matchup: str,
    hand_number: int,
    bankroll: int,
    bot_delay: float,
) -> tuple[int, bool]:
    state = game.reset()
    your_card = card_label(state.player_cards[0])
    last_status = style.paint("New hand.", Style.CYAN)

    while True:
        legal = game.legal_actions()
        board_text = "?" if state.board_card is None else rank_of_card_text(state.board_card)
        hist = format_history(game.history_label())

        def render(error_line: str | None) -> None:
            render_table(
                style=style,
                clear_enabled=clear_enabled,
                matchup=matchup,
                hand_number=hand_number,
                bankroll=bankroll,
                your_card=your_card,
                board_text=board_text,
                round_text=round_name(state.round_index),
                pot=state.pot,
                to_call=state.to_call,
                history_text=hist,
                legal=legal,
                your_turn=(state.current_player == 0),
                status_line=last_status,
                error_line=error_line,
            )

        if state.current_player == 0:
            action = read_action_with_ui(style=style, render_fn=render, legal=legal)
            last_status = style.paint(f"You chose {action.value}.", Style.CYAN)
        else:
            render(None)
            action = choose_bot_action(
                state=state,
                legal=legal,
                rng=rng,
                rl_policy=rl_policy,
                cfr_policy=cfr_policy,
                rl_key_fn=rl_info_state_key,
                cfr_key_fn=cfr_info_state_key,
                greedy=greedy,
            )
            last_status = style.paint(f"Bot chose {action.value}.", Style.MAGENTA)
            if bot_delay > 0:
                time.sleep(bot_delay)

        state, rewards, done = game.step(action)
        if done:
            bankroll_after = bankroll + rewards[0]
            board_final = "?" if state.board_card is None else rank_of_card_text(state.board_card)
            continue_playing = show_hand_result(
                style=style,
                clear_enabled=clear_enabled,
                matchup=matchup,
                hand_number=hand_number,
                bankroll=bankroll_after,
                rewards=rewards,
                your_card=card_label(state.player_cards[0]),
                bot_card=card_label(state.player_cards[1]),
                board_text=board_final,
                history_text=format_history(game.history_label()),
            )
            return rewards[0], continue_playing


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    clear_enabled = (not args.no_clear) and sys.stdout.isatty()
    style = Style(enabled=should_use_color(args.no_color))
    bot_delay = max(0.0, args.bot_delay)
    if not sys.stdout.isatty():
        bot_delay = 0.0

    game = LeducPokerGame(rng=rng)
    rl_policy = TabularSoftmaxPolicy.load(args.rl_policy) if args.rl_policy else None
    cfr_policy = AverageStrategyPolicy.load(args.cfr_policy) if args.cfr_policy else None

    if rl_policy is not None:
        matchup = "vs RL bot"
    elif cfr_policy is not None:
        matchup = "vs CFR bot"
    else:
        matchup = "vs random bot"

    bankroll = 0
    hands = 0
    try:
        while True:
            reward, continue_playing = play_hand(
                game=game,
                rng=rng,
                rl_policy=rl_policy,
                cfr_policy=cfr_policy,
                greedy=args.bot_greedy,
                style=style,
                clear_enabled=clear_enabled,
                matchup=matchup,
                hand_number=hands + 1,
                bankroll=bankroll,
                bot_delay=bot_delay,
            )
            bankroll += reward
            hands += 1
            if not continue_playing:
                break
    except KeyboardInterrupt:
        pass

    clear_screen(clear_enabled)
    title = style.paint("Session ended.", Style.BOLD, Style.CYAN)
    score = style.paint(
        format_score(bankroll),
        Style.BOLD,
        Style.GREEN if bankroll >= 0 else Style.RED,
    )
    print(title)
    print(f"Final score: {score} over {hands} hand(s)")


if __name__ == "__main__":
    main()
