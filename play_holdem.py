"""Play heads-up limit Texas Hold'em in the terminal."""

from __future__ import annotations

import argparse
import random
import sys
import time

from holdem_limit.abstraction import info_state_key
from holdem_limit.cfr import AverageStrategyPolicy
from holdem_limit.game import Action, BET_SIZES, HoldemLimitGame, card_label
from holdem_limit.nfsp import NfspAveragePolicy
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
ACTION_ORDER = (Action.CHECK, Action.BET, Action.CALL, Action.RAISE, Action.FOLD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rl-policy", type=str, default=None, help="Path to RL policy JSON")
    group.add_argument("--cfr-policy", type=str, default=None, help="Path to CFR policy JSON")
    group.add_argument("--nfsp-policy", type=str, default=None, help="Path to NFSP policy JSON")
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


def action_cost(state, action: Action) -> int:
    if action == Action.CALL:
        return state.to_call
    if action == Action.BET:
        return BET_SIZES[state.round_index]
    if action == Action.RAISE:
        return state.to_call + BET_SIZES[state.round_index]
    return 0


def action_label(state, action: Action) -> str:
    if action == Action.CHECK:
        return "Check"
    if action == Action.FOLD:
        return "Fold"
    cost = action_cost(state, action)
    if action == Action.CALL:
        return f"Call +{cost}"
    if action == Action.BET:
        return f"Bet +{cost}"
    if action == Action.RAISE:
        return f"Raise +{cost}"
    return action.value.title()


def legal_lines(state, legal: tuple[Action, ...], style: Style) -> list[str]:
    lines: list[str] = []
    legal_in_order = [action for action in ACTION_ORDER if action in legal]
    for idx, action in enumerate(legal_in_order, start=1):
        token = f"[{idx}] {action_label(state, action)} ({ACTION_KEYS[action]})"
        if action in (Action.BET, Action.RAISE):
            lines.append(style.paint(token, Style.YELLOW, Style.BOLD))
        elif action == Action.FOLD:
            lines.append(style.paint(token, Style.RED))
        elif action == Action.CALL:
            lines.append(style.paint(token, Style.GREEN))
        else:
            lines.append(style.paint(token, Style.CYAN))
    lines.append(style.paint("[q] quit", Style.DIM))
    return lines


def resolve_action_input(raw: str, legal: tuple[Action, ...]) -> Action | None:
    text = raw.strip().lower()
    if not text:
        return None
    if text in QUIT_WORDS:
        raise KeyboardInterrupt

    legal_in_order = [action for action in ACTION_ORDER if action in legal]
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= len(legal_in_order):
            return legal_in_order[idx - 1]

    mapped = ACTION_ALIASES.get(text)
    if mapped in legal:
        return mapped
    return None


def describe_to_call(to_call: int) -> str:
    if to_call <= 0:
        return "No bet to call."
    if to_call == 1:
        return "You need 1 chip to call."
    return f"You need {to_call} chips to call."


def format_action_log(events: list[str], max_lines: int = 6) -> list[str]:
    if not events:
        return ["No actions yet."]
    if len(events) <= max_lines:
        return events
    return ["..."] + events[-(max_lines - 1) :]


def select_bot_action(
    *,
    state,
    legal: tuple[Action, ...],
    rng: random.Random,
    rl_policy: TabularSoftmaxPolicy | None,
    cfr_policy: AverageStrategyPolicy | None,
    nfsp_policy: NfspAveragePolicy | None,
    greedy: bool,
) -> Action:
    if rl_policy is None and cfr_policy is None and nfsp_policy is None:
        return rng.choice(list(legal))
    key = info_state_key(state, 1)
    if rl_policy is not None:
        if greedy:
            return rl_policy.greedy_action(key, legal)
        return rl_policy.sample_action(key, legal, rng)
    if nfsp_policy is not None:
        if greedy:
            return nfsp_policy.greedy_action(key, legal)
        return nfsp_policy.sample_action(key, legal, rng)
    if greedy:
        return cfr_policy.greedy_action(key, legal)
    return cfr_policy.sample_action(key, legal, rng)


def play_hand(
    *,
    game: HoldemLimitGame,
    rng: random.Random,
    rl_policy: TabularSoftmaxPolicy | None,
    cfr_policy: AverageStrategyPolicy | None,
    nfsp_policy: NfspAveragePolicy | None,
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
    status = style.paint("New hand. You are small blind.", Style.CYAN)
    action_events: list[str] = ["Blinds posted: You 1, Bot 2."]

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
            print(style.paint(describe_to_call(state.to_call), Style.DIM))
            print(style.paint("-" * 64, Style.DIM))
            print(style.paint("Recent actions:", Style.BOLD))
            for event in format_action_log(action_events):
                print(style.paint(f"- {event}", Style.DIM))
            print(style.paint("-" * 64, Style.DIM))
            print(status)
            if error:
                print(style.paint(error, Style.RED))
            if state.current_player == 0:
                for line in legal_lines(state, legal, style):
                    print(line)
                print(style.paint("Action >", Style.BOLD), end=" ")
            else:
                print(style.paint("Bot is acting...", Style.MAGENTA))

        if state.current_player == 0:
            error = None
            while True:
                draw(error)
                raw = input()
                resolved = resolve_action_input(raw, legal)
                if resolved is not None:
                    action = resolved
                    break
                legal_names = ", ".join(action.value for action in legal)
                error = f"Invalid action. Choose number/key for: {legal_names}"
            status = style.paint(f"You: {action_label(state, action)}", Style.CYAN)
            action_events.append(f"{street_name(state.round_index)}: You {action.value}")
        else:
            draw()
            action = select_bot_action(
                state=state,
                legal=legal,
                rng=rng,
                rl_policy=rl_policy,
                cfr_policy=cfr_policy,
                nfsp_policy=nfsp_policy,
                greedy=greedy,
            )
            status = style.paint(f"Bot: {action_label(state, action)}", Style.MAGENTA)
            action_events.append(f"{street_name(state.round_index)}: Bot {action.value}")
            if bot_delay > 0:
                time.sleep(bot_delay)

        previous_round = state.round_index
        previous_board_len = len(state.board)
        state, rewards, done = game.step(action)
        if not done and state.round_index != previous_round:
            if len(state.board) == 3 and previous_board_len < 3:
                flop = " ".join(card_label(card) for card in state.board[:3])
                action_events.append(f"Flop dealt: {flop}")
            elif len(state.board) == 4 and previous_board_len < 4:
                action_events.append(f"Turn dealt: {card_label(state.board[3])}")
            elif len(state.board) == 5 and previous_board_len < 5:
                action_events.append(f"River dealt: {card_label(state.board[4])}")
        if done:
            bankroll_after = bankroll + rewards[0]
            clear_screen(clear_enabled)
            print(style.paint("Hand Complete", Style.BOLD, Style.CYAN), f"({matchup})")
            print(style.paint("-" * 64, Style.DIM))
            print(f"Your cards: {style.paint(hero_cards, Style.BOLD, Style.YELLOW)}")
            bot_cards = f"{card_label(state.player_hands[1][0])} {card_label(state.player_hands[1][1])}"
            print(f"Bot cards:  {style.paint(bot_cards, Style.BOLD, Style.MAGENTA)}")
            print(f"Board:      {style.paint(render_board(state.board), Style.BOLD)}")
            print(style.paint("Action summary:", Style.BOLD))
            for event in format_action_log(action_events, max_lines=10):
                print(style.paint(f"- {event}", Style.DIM))
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
    nfsp_policy = NfspAveragePolicy.load(args.nfsp_policy) if args.nfsp_policy else None

    if rl_policy is not None:
        matchup = "vs RL bot"
    elif nfsp_policy is not None:
        matchup = "vs NFSP bot"
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
                nfsp_policy=nfsp_policy,
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
