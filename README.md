# Poker Games

First implemented game: a minimal Kuhn Poker environment in Python.

## Quickstart

```bash
python3 -m unittest discover -s tests -v
```

## Play in Terminal

```bash
python3 play_kuhn.py
```

You are player 0. Actions:
- `check` or `c`
- `bet` or `b`
- `call` or `ca`
- `fold` or `f`
- `q` to quit

## Train an RL Agent (REINFORCE)

```bash
python3 train_reinforce_kuhn.py --episodes 150000 --save models/kuhn_reinforce_policy.json
```

Then play against the trained policy:

```bash
python3 play_kuhn.py --policy models/kuhn_reinforce_policy.json
```

## Example

```python
from kuhn_poker import Action, Card, KuhnPokerGame

game = KuhnPokerGame()
state = game.reset(cards=(Card.K, Card.Q))
print(state, game.legal_actions())

state, rewards, done = game.step(Action.BET)
print(state, rewards, done)

state, rewards, done = game.step(Action.CALL)
print(state, rewards, done)  # terminal; rewards are net chip gains
```

## Environment API

- `KuhnPokerGame.reset(seed=None, cards=None) -> KuhnPokerState`
- `KuhnPokerGame.legal_actions() -> tuple[Action, ...]`
- `KuhnPokerGame.step(action) -> (KuhnPokerState, rewards, done)`
- `KuhnPokerGame.history_label() -> str` (compact format like `cbf`, `bc`)

`rewards` is always from player 0 and player 1 perspective as net chips for the hand.
