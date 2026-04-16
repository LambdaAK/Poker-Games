"""Play heads-up limit Texas Hold'em in the terminal."""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Callable

from holdem_limit.abstraction import info_state_key
from holdem_limit.cfr import AverageStrategyPolicy
from holdem_limit.game import Action, HoldemLimitGame, card_label
from holdem_limit.rl import TabularSoftmaxPolicy


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rl-policy", type=str, default=None, help="Path to RL policy JSON")
    group.add_argument("--cfr-policy", type=str, default=None, help="Path to CFR policy JSON")
    parser.add_argument("--bot-greedy", action="store_true", help="Use greedy bot policy")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--no-clear", action="store_true", help="Disable screen clearing")
    parser.add_argument("--bot-delay", type=float, default=0.35, help="Bot delay seconds")
    return parser.parse_args()


def clear_screen(enabled: bool) -> None:
    if enabled:
        print("\033[2J\033[H", end="")


def street_name(round_index: int) -> str:
    return ("Preflop", "Flop", "Turn", "River")[round_index]


def render_board(board) -> str:
    slots = [card_label(card) for card in board]
    while len(slots) < 5:
        slots.append("--")
    return " ".join(slots)


def legal_line(legal: tuple[Action, ...], style: Style) -> str:
    pieces: list[str] = []
    for action in legal:
        label = f"[{ACTION_KEYS[action]}] {action.value}"
        if action in (Action.BET, Action.RAISE):
            pieces.append(style.paint(label, Style.YELLOW, Style.BOLD))
        elif action == Action.FOLD:
            pieces.append(style.paint(label, Style.RED))
        elif action == Action.CALL:
            pieces.append(style.paint(label, Style.GREEN))
        else:
            pieces.append(style.paint(label, Style.CYAN))
    pieces.append(style.paint("[q] quit", Style.DIM))
    return "  ".join(pieces)


def select_bot_action(
    *,
    state,
    legal: tuple[Action, ...],
    rng: random.Random,
    rl_policy: TabularSoftmaxPolicy | None,
    cfr_policy: AverageStrategyPolicy | None,
    greedy: bool,
) -> Action:
    if rl_policy is None and cfr_policy is None:
        return rng.choice(list(legal))
    key = info_state_key(state, 1)
    if rl_policy is not None:
        if greedy:
            return rl_policy.greedy_action(key, legal)
        return rl_policy.sample_action(key, legal, rng)
    if greedy:
        return cfr_policy.greedy_action(key, legal)
    return cfr_policy.sample_action(key, legal, rng)


def play_hand(
    *,
    game: HoldemLimitGame,
    rng: random.Random,
    rl_policy: TabularSoftmaxPolicy | None,
    cfr_policy: AverageStrategyPolicy | None,
    greedy: bool,
    style: Style,
    clear_enabled: bool,
    matchup: str,
    hand_no: int,
    bankroll: int,
    bot_delay: float,
) -> tuple[int, bool]:
    state = game.reset()
    hero_cards = f"{card_label(state.player_hands[0][0])} {card_label(state.player_hands[0][1])}"
    status = style.paint("New hand.", Style.CYAN)

    while True:
        legal = game.legal_actions()

        def draw(error: str | None = None) -> None:
            clear_screen(clear_enabled)
            score_color = Style.GREEN if bankroll >= 0 else Style.RED
            print(style.paint("Limit Hold'em", Style.BOLD, Style.CYAN), f"({matchup})")
            print(f"Hand {hand_no}    Score {style.paint(f'{bankroll:+d}', Style.BOLD, score_color)}")
            print(style.paint("-" * 64, Style.DIM))
            print(f"Your cards: {style.paint(hero_cards, Style.BOLD, Style.YELLOW)}")
            print(f"Board:      {style.paint(render_board(state.board), Style.BOLD)}")
            print(
                f"Street: {style.paint(street_name(state.round_index), Style.BOLD)}    "
                f"Pot: {style.paint(str(state.pot), Style.BOLD)}    "
                f"To call: {style.paint(str(state.to_call), Style.BOLD)}"
            )
            print(style.paint(f"History: {game.history_label()}", Style.DIM))
            print(style.paint("-" * 64, Style.DIM))
            print(status)
            if error:
                print(style.paint(error, Style.RED))
            if state.current_player == 0:
                print(legal_line(legal, style))
                print(style.paint("Action >", Style.BOLD), end=" ")
            else:
                print(style.paint("Bot is acting...", Style.MAGENTA))

        if state.current_player == 0:
            error = None
            while True:
                draw(error)
                raw = input().strip().lower()
                if raw in QUIT_WORDS:
                    raise KeyboardInterrupt
                action = ACTION_ALIASES.get(raw)
                if action in legal:
                    break
                error = f"Invalid action. Legal: {', '.join(a.value for a in legal)}"
            status = style.paint(f"You chose {action.value}.", Style.CYAN)
        else:
            draw()
            action = select_bot_action(
                state=state,
                legal=legal,
                rng=rng,
                rl_policy=rl_policy,
                cfr_policy=cfr_policy,
                greedy=greedy,
            )
            status = style.paint(f"Bot chose {action.value}.", Style.MAGENTA)
            if bot_delay > 0:
                time.sleep(bot_delay)

        state, rewards, done = game.step(action)
        if done:
            bankroll_after = bankroll + rewards[0]
            clear_screen(clear_enabled)
            print(style.paint("Hand Complete", Style.BOLD, Style.CYAN), f"({matchup})")
            print(style.paint("-" * 64, Style.DIM))
            print(f"Your cards: {style.paint(hero_cards, Style.BOLD, Style.YELLOW)}")
            bot_cards = f"{card_label(state.player_hands[1][0])} {card_label(state.player_hands[1][1])}"
            print(f"Bot cards:  {style.paint(bot_cards, Style.BOLD, Style.MAGENTA)}")
            print(f"Board:      {style.paint(render_board(state.board), Style.BOLD)}")
            print(style.paint(f"History: {game.history_label()}", Style.DIM))
            print(style.paint("-" * 64, Style.DIM))
            if rewards[0] > 0:
                print(style.paint(f"You win {rewards[0]} chips", Style.BOLD, Style.GREEN))
            elif rewards[0] < 0:
                print(style.paint(f"You lose {-rewards[0]} chips", Style.BOLD, Style.RED))
            else:
                print(style.paint("Split pot", Style.BOLD, Style.CYAN))
            print(f"Session score: {style.paint(f'{bankroll_after:+d}', Style.BOLD)}")
            print(style.paint("Press Enter for next hand, or q to quit.", Style.DIM))
            raw = input().strip().lower()
            return rewards[0], raw not in QUIT_WORDS


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    style = Style(enabled=(not args.no_color and sys.stdout.isatty()))
    clear_enabled = (not args.no_clear) and sys.stdout.isatty()
    bot_delay = max(0.0, args.bot_delay)
    if not sys.stdout.isatty():
        bot_delay = 0.0

    game = HoldemLimitGame(rng=rng)
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
            reward, cont = play_hand(
                game=game,
                rng=rng,
                rl_policy=rl_policy,
                cfr_policy=cfr_policy,
                greedy=args.bot_greedy,
                style=style,
                clear_enabled=clear_enabled,
                matchup=matchup,
                hand_no=hands + 1,
                bankroll=bankroll,
                bot_delay=bot_delay,
            )
            bankroll += reward
            hands += 1
            if not cont:
                break
    except KeyboardInterrupt:
        pass

    clear_screen(clear_enabled)
    print(style.paint("Session ended.", Style.BOLD, Style.CYAN))
    score_color = Style.GREEN if bankroll >= 0 else Style.RED
    print(f"Final score: {style.paint(f'{bankroll:+d}', Style.BOLD, score_color)} over {hands} hand(s)")


if __name__ == "__main__":
    main()
