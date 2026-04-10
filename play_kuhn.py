"""Play Kuhn Poker from the terminal (human vs random or trained bot)."""

from __future__ import annotations

import argparse
import random

from kuhn_poker import Action, Card, KuhnPokerGame, TabularSoftmaxPolicy
from kuhn_poker.rl import info_state_key


ACTION_ALIASES = {
    "c": Action.CHECK,
    "check": Action.CHECK,
    "b": Action.BET,
    "bet": Action.BET,
    "call": Action.CALL,
    "ca": Action.CALL,
    "f": Action.FOLD,
    "fold": Action.FOLD,
}


def card_name(card: Card) -> str:
    return {Card.J: "J", Card.Q: "Q", Card.K: "K"}[card]


def action_name(action: Action) -> str:
    return action.value


def read_action(legal: tuple[Action, ...]) -> Action:
    legal_names = ", ".join(a.value for a in legal)
    while True:
        raw = input(f"Your action ({legal_names}): ").strip().lower()
        if raw in ("quit", "q", "exit"):
            raise KeyboardInterrupt
        action = ACTION_ALIASES.get(raw)
        if action in legal:
            return action
        print(f"Invalid action. Allowed: {legal_names}")


def play_hand(
    game: KuhnPokerGame,
    rng: random.Random,
    bot_policy: TabularSoftmaxPolicy | None,
    bot_greedy: bool,
) -> int:
    # Human is player 0, bot is player 1.
    state = game.reset()
    print("\n--- New Hand ---")
    print(f"Your card: {card_name(state.player_cards[0])}")

    while True:
        legal = game.legal_actions()
        if state.current_player == 0:
            action = read_action(legal)
            actor = "You"
        else:
            if bot_policy is None:
                action = rng.choice(list(legal))
            else:
                info_state = info_state_key(state.player_cards[1], game.history_label())
                if bot_greedy:
                    action = bot_policy.greedy_action(info_state, legal)
                else:
                    action = bot_policy.sample_action(info_state, legal, rng)
            actor = "Bot"
            print(f"Bot action: {action_name(action)}")

        state, rewards, done = game.step(action)
        if done:
            print(f"Bot card: {card_name(state.player_cards[1])}")
            print(f"History: {game.history_label()}")
            if rewards[0] > 0:
                print(f"You win {rewards[0]} chip(s).")
            elif rewards[0] < 0:
                print(f"You lose {-rewards[0]} chip(s).")
            else:
                print("Hand is a draw.")
            return rewards[0]

        if actor == "You":
            print("Bot to act...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Optional JSON path for a trained bot policy",
    )
    parser.add_argument(
        "--bot-greedy",
        action="store_true",
        help="Use greedy action selection for policy bot (default is stochastic)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random()
    if args.seed is not None:
        rng.seed(args.seed)
    game = KuhnPokerGame(rng=rng)
    bot_policy = TabularSoftmaxPolicy.load(args.policy) if args.policy else None
    bankroll = 0
    hands = 0

    print(
        "Kuhn Poker (you vs "
        + ("trained bot)" if bot_policy is not None else "random bot)")
    )
    print("Actions: check/c, bet/b, call/ca, fold/f. Type q to quit.")
    if args.policy:
        mode = "greedy" if args.bot_greedy else "stochastic"
        print(f"Loaded bot policy: {args.policy} ({mode})")

    try:
        while True:
            bankroll += play_hand(
                game, rng, bot_policy=bot_policy, bot_greedy=args.bot_greedy
            )
            hands += 1
            print(f"Score after {hands} hand(s): {bankroll:+d} chip(s)")
    except KeyboardInterrupt:
        print("\nSession ended.")
        print(f"Final score: {bankroll:+d} chip(s) over {hands} hand(s)")


if __name__ == "__main__":
    main()
