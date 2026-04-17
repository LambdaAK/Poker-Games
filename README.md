# Poker Games

Implemented games:
- Kuhn Poker
- Leduc Poker (two rounds, board card, fixed-limit betting with one raise per round)
- Heads-up Limit Texas Hold'em

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

Play Leduc:

```bash
python3 play_leduc.py
```

Leduc actions:
- `check` or `k`
- `bet` or `b`
- `call` or `c`
- `raise` or `r`
- `fold` or `f`
- `q` to quit

Play Limit Hold'em:

```bash
python3 play_holdem.py
```

Hold'em actions:
- `check` or `k`
- `bet` or `b`
- `call` or `c`
- `raise` or `r`
- `fold` or `f`
- `q` to quit

`play_holdem.py` also supports numbered choices (`1`, `2`, `3`, ...) matching the on-screen action list.

## Train an RL Agent (REINFORCE)

```bash
python3 train_reinforce_kuhn.py --episodes 150000 --save models/kuhn_reinforce_policy.json
```

Then play against the trained policy:

```bash
python3 play_kuhn.py --policy models/kuhn_reinforce_policy.json
```

## Leduc Baselines

Train CFR (strong strategy baseline):

```bash
python3 train_cfr_leduc.py --iterations 4000 --save models/leduc_cfr_policy.json
```

Train REINFORCE (RL baseline):

```bash
python3 train_reinforce_leduc.py --episodes 400000 --save models/leduc_reinforce_policy.json
```

Play against trained bots:

```bash
python3 play_leduc.py --cfr-policy models/leduc_cfr_policy.json
python3 play_leduc.py --rl-policy models/leduc_reinforce_policy.json
```

## Hold'em Baselines

Train CFR baseline:

```bash
python3 train_cfr_holdem.py --iterations 5000 --save models/holdem_limit_cfr_policy.json
```

Train REINFORCE baseline:

```bash
python3 train_reinforce_holdem.py --episodes 250000 --save models/holdem_limit_reinforce_policy.json
```

Train NFSP-style advanced RL baseline:

```bash
python3 train_nfsp_holdem.py --episodes 350000 --save models/holdem_limit_nfsp_policy.json
```

Play against trained Hold'em bots:

```bash
python3 play_holdem.py --cfr-policy models/holdem_limit_cfr_policy.json
python3 play_holdem.py --rl-policy models/holdem_limit_reinforce_policy.json
python3 play_holdem.py --nfsp-policy models/holdem_limit_nfsp_policy.json
```

If your terminal does not render ANSI nicely:

```bash
python3 play_holdem.py --no-color --no-clear
```

If your terminal does not render ANSI nicely:

```bash
python3 play_leduc.py --no-color --no-clear
```

## Benchmark Harness

Run round-robin evaluation across games and agent types (random / CFR / RL / NFSP for Hold'em):

```bash
python3 evaluate_agents.py --games kuhn,leduc,holdem --seeds 0,1,2,3,4 --episodes 5000
```

The script prints:
- chips/hand
- 95% confidence interval over seeds
- bb/100
- Elo ratings within each game

It also writes a JSON report to `results/benchmark_<timestamp>.json`.

If your policies are in non-default locations, pass explicit paths:

```bash
python3 evaluate_agents.py \
  --kuhn-rl-policy /tmp/kuhn_eval_rl.json \
  --leduc-rl-policy /tmp/leduc_eval_rl.json \
  --leduc-cfr-policy /tmp/leduc_eval_cfr.json \
  --holdem-rl-policy /tmp/holdem_eval_rl.json \
  --holdem-cfr-policy /tmp/holdem_eval_cfr.json \
  --holdem-nfsp-policy /tmp/holdem_eval_nfsp.json
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
