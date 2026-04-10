"""Play Kuhn Poker from the terminal (human vs random bot)."""

from __future__ import annotations

import random

from kuhn_poker import Action, Card, KuhnPokerGame


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


def play_hand(game: KuhnPokerGame, rng: random.Random) -> int:
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
            action = rng.choice(list(legal))
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


def main() -> None:
    rng = random.Random()
    game = KuhnPokerGame(rng=rng)
    bankroll = 0
    hands = 0

    print("Kuhn Poker (you vs random bot)")
    print("Actions: check/c, bet/b, call/ca, fold/f. Type q to quit.")

    try:
        while True:
            hands += 1
            bankroll += play_hand(game, rng)
            print(f"Score after {hands} hand(s): {bankroll:+d} chip(s)")
    except KeyboardInterrupt:
        print("\nSession ended.")
        print(f"Final score: {bankroll:+d} chip(s) over {hands} hand(s)")


if __name__ == "__main__":
    main()
