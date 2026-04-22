"""Play blackjack in the terminal or auto-run a policy."""

from __future__ import annotations

import argparse
import random
import sys
import time

from blackjack import (
    Action,
    BasicStrategyPolicy,
    BlackjackGame,
    RandomPolicy,
    TabularActionValuePolicy,
    card_label,
    hand_value,
    history_label,
)


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
    "h": Action.HIT,
    "hit": Action.HIT,
    "s": Action.STAND,
    "stand": Action.STAND,
    "d": Action.DOUBLE,
    "double": Action.DOUBLE,
}
ACTION_KEYS = {
    Action.HIT: "h",
    Action.STAND: "s",
    Action.DOUBLE: "d",
}
ACTION_ORDER = (Action.HIT, Action.STAND, Action.DOUBLE)
QUIT_WORDS = {"q", "quit", "exit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-play the hand using the selected algorithm",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=("random", "basic", "mc", "q"),
        default="basic",
        help="Policy to use in auto mode",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to a trained MC/Q policy JSON file",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--no-clear", action="store_true", help="Disable screen clearing")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Auto-play delay in seconds",
    )
    return parser.parse_args()


def clear_screen(enabled: bool) -> None:
    if enabled:
        print("\033[2J\033[H", end="")


def hand_summary(hand) -> str:
    total, soft = hand_value(hand)
    softness = "soft" if soft else "hard"
    return f"{' '.join(card_label(card) for card in hand)}  ({total} {softness})"


def dealer_visible_summary(hand) -> str:
    return f"{card_label(hand[0])} ??"


def action_label(action: Action) -> str:
    return action.value


def legal_action_line(legal: tuple[Action, ...], style: Style) -> str:
    parts: list[str] = []
    for idx, action in enumerate(action_order(legal), start=1):
        token = f"[{idx}] {action.value} ({ACTION_KEYS[action]})"
        if action == Action.HIT:
            parts.append(style.paint(token, Style.YELLOW, Style.BOLD))
        elif action == Action.STAND:
            parts.append(style.paint(token, Style.GREEN, Style.BOLD))
        else:
            parts.append(style.paint(token, Style.CYAN, Style.BOLD))
    parts.append(style.paint("[q] quit", Style.DIM))
    return "  ".join(parts)


def action_order(legal: tuple[Action, ...]) -> tuple[Action, ...]:
    return tuple(action for action in ACTION_ORDER if action in legal)


def read_action(legal: tuple[Action, ...]) -> Action:
    legal_in_order = action_order(legal)
    legal_names = ", ".join(action.value for action in legal_in_order)
    while True:
        raw = input(f"Your action ({legal_names}): ").strip().lower()
        if raw in QUIT_WORDS:
            raise KeyboardInterrupt
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(legal_in_order):
                return legal_in_order[idx - 1]
        action = ACTION_ALIASES.get(raw)
        if action in legal_in_order:
            return action
        print(f"Invalid action. Allowed: {legal_names}")


def build_policy(args: argparse.Namespace):
    if args.algorithm == "random":
        return RandomPolicy()
    if args.algorithm == "basic":
        return BasicStrategyPolicy()

    path = args.policy or f"models/blackjack_{args.algorithm}_policy.json"
    return TabularActionValuePolicy.load(path)


def describe_policy(policy) -> str:
    if isinstance(policy, RandomPolicy):
        return "random policy"
    if isinstance(policy, BasicStrategyPolicy):
        return "basic strategy"
    return "trained tabular policy"


def choose_action(policy, state, legal: tuple[Action, ...], rng: random.Random) -> Action:
    if isinstance(policy, TabularActionValuePolicy):
        return policy.choose_action(state, legal, rng, greedy=True)
    return policy.choose_action(state, legal, rng, greedy=True)


def render_state(
    *,
    style: Style,
    clear_enabled: bool,
    hand_no: int,
    bankroll: float,
    state,
    legal: tuple[Action, ...],
    auto: bool,
    status_line: str,
    error_line: str | None,
) -> None:
    clear_screen(clear_enabled)
    title = style.paint("Blackjack", Style.BOLD, Style.CYAN)
    score = style.paint(f"{bankroll:+.1f}", Style.BOLD, Style.GREEN if bankroll >= 0 else Style.RED)
    print(f"{title}  Hand {hand_no}  Score {score}")
    print(style.paint("-" * 64, Style.DIM))
    print(f"Your hand:   {style.paint(hand_summary(state.player_hand), Style.BOLD, Style.YELLOW)}")
    if state.terminal:
        print(f"Dealer hand: {style.paint(hand_summary(state.dealer_hand), Style.BOLD, Style.MAGENTA)}")
    else:
        print(f"Dealer upcard: {style.paint(dealer_visible_summary(state.dealer_hand), Style.BOLD)}")
    print(f"History:     {style.paint(history_label(state) or '-', Style.DIM)}")
    print(
        f"Stake:       {style.paint(str(state.stake), Style.BOLD)}    "
        f"Can double: {style.paint('yes' if state.can_double else 'no', Style.BOLD)}"
    )
    print(style.paint("-" * 64, Style.DIM))
    if status_line:
        print(status_line)
    if error_line:
        print(style.paint(error_line, Style.RED))
    if not state.terminal:
        if auto:
            print(style.paint("Auto policy is acting...", Style.MAGENTA))
        else:
            print(legal_action_line(legal, style))
            print(style.paint("Action >", Style.BOLD), end=" ")


def show_result(
    *,
    style: Style,
    clear_enabled: bool,
    hand_no: int,
    bankroll: float,
    state,
    reward: float,
    auto: bool,
) -> bool:
    clear_screen(clear_enabled)
    title = style.paint("Hand Complete", Style.BOLD, Style.CYAN)
    print(f"{title}  Hand {hand_no}  Score {style.paint(f'{bankroll:+.1f}', Style.BOLD)}")
    print(style.paint("-" * 64, Style.DIM))
    print(f"Your hand:   {style.paint(hand_summary(state.player_hand), Style.BOLD, Style.YELLOW)}")
    print(f"Dealer hand: {style.paint(hand_summary(state.dealer_hand), Style.BOLD, Style.MAGENTA)}")
    print(style.paint("-" * 64, Style.DIM))
    if reward > 0:
        result = style.paint(f"You win {reward:.1f} chips", Style.GREEN, Style.BOLD)
    elif reward < 0:
        result = style.paint(f"You lose {-reward:.1f} chips", Style.RED, Style.BOLD)
    else:
        result = style.paint("Push", Style.CYAN, Style.BOLD)
    print(result)
    if auto:
        print(style.paint("Press Enter for the next hand, or q to quit.", Style.DIM))
    else:
        print(style.paint("Press Enter for the next hand, or q to quit.", Style.DIM))
    raw = input().strip().lower()
    return raw not in QUIT_WORDS


def play_hand(
    *,
    game: BlackjackGame,
    rng: random.Random,
    policy,
    auto: bool,
    style: Style,
    clear_enabled: bool,
    hand_no: int,
    bankroll: float,
    delay: float,
) -> tuple[float, bool]:
    state = game.reset()
    status_line = style.paint(f"Policy: {describe_policy(policy)}" if auto else "Manual play", Style.CYAN)

    if state.terminal:
        reward = state.rewards[0]
        bankroll_after = bankroll + reward
        return reward, show_result(
            style=style,
            clear_enabled=clear_enabled,
            hand_no=hand_no,
            bankroll=bankroll_after,
            state=state,
            reward=reward,
            auto=auto,
        )

    while True:
        legal = game.legal_actions()
        error: str | None = None
        while True:
            render_state(
                style=style,
                clear_enabled=clear_enabled,
                hand_no=hand_no,
                bankroll=bankroll,
                state=state,
                legal=legal,
                auto=auto,
                status_line=status_line,
                error_line=error,
            )
            if auto:
                action = choose_action(policy, state, legal, rng)
                print(action_label(action))
                if delay > 0 and sys.stdout.isatty():
                    time.sleep(delay)
                break
            raw = input().strip().lower()
            if raw in QUIT_WORDS:
                raise KeyboardInterrupt
            if raw.isdigit():
                idx = int(raw)
                legal_in_order = action_order(legal)
                if 1 <= idx <= len(legal_in_order):
                    action = legal_in_order[idx - 1]
                    break
            action = ACTION_ALIASES.get(raw)
            if action in legal:
                break
            error = f"Invalid action. Allowed: {', '.join(a.value for a in action_order(legal))}"

        status_line = style.paint(f"Action: {action.value}", Style.MAGENTA if auto else Style.CYAN)
        state, rewards, done = game.step(action)
        if done:
            bankroll_after = bankroll + rewards[0]
            return rewards[0], show_result(
                style=style,
                clear_enabled=clear_enabled,
                hand_no=hand_no,
                bankroll=bankroll_after,
                state=state,
                reward=rewards[0],
                auto=auto,
            )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    style = Style(enabled=(not args.no_color and sys.stdout.isatty()))
    clear_enabled = (not args.no_clear) and sys.stdout.isatty()
    delay = max(0.0, args.delay)
    if not sys.stdout.isatty():
        delay = 0.0

    game = BlackjackGame(rng=rng)
    policy = build_policy(args) if args.auto else BasicStrategyPolicy()
    bankroll = 0.0
    hands = 0

    print(
        "Blackjack ("
        + ("auto-play" if args.auto else "you play")
        + ")"
    )
    if args.auto:
        print(f"Using {describe_policy(policy)}")
    print("Actions: hit/h, stand/s, double/d. Type q to quit.")

    try:
        while True:
            reward, keep_playing = play_hand(
                game=game,
                rng=rng,
                policy=policy,
                auto=args.auto,
                style=style,
                clear_enabled=clear_enabled,
                hand_no=hands + 1,
                bankroll=bankroll,
                delay=delay,
            )
            bankroll += reward
            hands += 1
            print(f"Score after {hands} hand(s): {bankroll:+.1f} chips")
            if not keep_playing:
                break
    except KeyboardInterrupt:
        print("\nSession ended.")
        print(f"Final score: {bankroll:+.1f} chips over {hands} hand(s)")


if __name__ == "__main__":
    main()
