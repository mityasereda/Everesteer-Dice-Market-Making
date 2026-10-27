# Dice Market Making: findings, status and ideas

Working notes for the Everesteer dice market making challenge. Last updated 27 Oct 2026. The challenge closes
**1 Nov 2026, 00:00 London time**. Nothing has been uploaded to the leaderboard yet.

## 1. Status

| Item | State |
|---|---|
| `dice_market/agent.py` (v1, adaptive reader) | Written, passes `--check`, 0 errors and 0 timeouts over a full 200-round match |
| Upload to the leaderboard | **Not done.** Needs the owner's go-ahead. 20 uploads per day, 30 s apart |
| Replay dataset | Downloaded locally (771 MB, not in git). 56,160 matches |
| Real-field strength of v1 | **Unknown.** Local tests only cover the house agents, which are much weaker than real candidates |

## 2. The game in brief (full rules: `dice_market/AGENTS.md`)

- 12 dice are rolled each round, the value is their sum (mean 42, sd 5.9). Seat `i` sees dice `i` and `i+6`. With 5 seats nobody sees dice 5 and 11.
- 4 quoting steps per round. Every seat sends a bid and an ask for 1 unit, at most 4 apart. Quotes rest and can be hit at their own price at the next step (a stale quote). A forced quote is 40 / 44.
- A match is 200 rounds, 5 seats, zero-sum. Elo is based on pairwise profit comparisons (10 games per match), so rank at the table matters, not the size of the profit.
- Pass gate: beat "the boss" in 2 of 3 matches with no errors or slow calls (0.05 s per call).

Corrections to what I first assumed:

- **A position of +5 can never be reached.** A seat has one bid slot, and its old bid can only be hit if the previous step's new bid did not fill. That caps buys at 4 per round (3 old-bid hits plus one new-bid fill at the last step). Confirmed by 300,000 random aggressive rounds (max 4) and a scripted round that reaches +4. The limit of 5 never binds.
- After step 0 every seat re-quotes every step, so cross trades are always at the midpoint. The only trades at a stale price are takes.
- `round_no` is documented as 0..99 in the interface section, but a match is 200 rounds.

## 3. The data

`replays.jsonl.gz` holds every leaderboard match, 23 Oct to 27 Oct 2026 (56,160 matches, 130 agent names; about 238k candidate seats, 42k house seats). It includes **"house pro"**, which the starter pack does not. Download with `dmapi.py dataset`. Files used for analysis live in `dice_market/analysis/`. The summary used by the charts is `docs/data/profile.json`.

Caveats: quote-level statistics come from every 8th match; agent names are account handles, not versions, so one name can mix several uploaded versions.

## 4. What the top agents do

![Who wins](figures/1_who_wins.png)

The best candidates win about 2 of 3 pairwise games, on only +0.4 to +1.05 per round. These margins are small, so a 200-round match is noisy and ranking needs many matches.

![Step 0 hiding](figures/2_step0_hiding.png)

**Nearly everyone hides at step 0** by sending exactly the forced quote 40 / 44. Exceptions: zeus (56%), MerchTrader (51%), spivak (70%). The house reader shows its dice at step 0 in 83% of rounds and loses badly.

![Estimation error](figures/3_estimation_error.png)

The information flow is: step 0 shows nothing, step 1 is where the field first shows its dice, step 2 quotes converge to a consensus, step 3 is the best estimate. By step 3 the best agents' quote mid is within about 3.0 to 3.1 RMSE of the true value. The floor with all ten visible dice known is 2.42. The quotes of the whole field shrink toward 42, so strong rounds are underestimated.

![Where the edge is](figures/4_where_the_edge_is.png)

Profit comes from **taking stale quotes at steps 2 and 3**. The house reader, hider and fair value agents lose about 1.8 per round at each of those steps because the field picks them off. Step 0 and step 1 carry almost no edge.

![Final-step spread](figures/5_final_step_spread.png)

Final-step spreads vary a lot. Docsdadi, zudvarnoki and billhwang quote about 1 wide. yurko_mud (highest win rate) quotes 2.0 and Joshua_Karat 2.95. **The correlation between a tight final spread and win rate across 38 candidates is only -0.19**, so tightening is not clearly the driver. Treat it as something to test.

## 5. How the decode works (and what did not)

All numbers are RMSE of an estimate of the value, from my dice plus the others' quotes.

| Approach | Step 2 | Step 3 |
|---|---|---|
| Own two dice only | 5.40 | 5.40 |
| Pooled linear decode over everyone's mids (offline OLS on 14,000 matches) | 4.33 | 3.28 |
| Other feature sets (sorted mids, abs values) | 4.29 | 3.26 |
| Weighting by each agent's identity (oracle) | n/a | 3.19 |
| Per-agent structural decode of their dice from their step-1 mid (oracle) | 4.47 | n/a |
| **Online per-match ridge regression (agent v1), rounds 30+** | **3.54** | **3.01** |
| Best agents' own quotes | 3.3 to 3.5 | 2.95 to 3.1 |

Conclusions:

- **What matters is adapting within the match.** A fixed pooled decode stalls at about 3.26. Online learning from the revealed true values reaches about 3.0, level with the best agents.
- Clever feature design, identity weighting and a structural dice decode did not help. My early guess that agent quality differences explained the gap was wrong.
- A pretrained starting prior only helps the first 30 rounds. Step 3 error in those rounds: 4.11 (old reader prior) to 3.65 (pooled prior, lambda 100).

## 6. Agent v1 (`dice_market/agent.py`)

- Estimates the value at each step as a linear function of my dice and the other seats' mids at earlier steps. Weights start at `PRIOR` (the pooled fit) and are refined every round by recursive ridge regression (`lam=100`), written in plain Python.
- Quotes estimate +/- half width (2, 2, 1.5, 1.5 by step) and shifts against inventory (0.4 per unit).
- Hides at step 0 (quotes 40 / 44). In the house tournament this raised the win rate against the house hider from 34% to 92%.

Local results (`python3 tourney.py agent.py`, 60 matches per mix, standard error about 0.07):

| Opponents | P&L / round | Win rate |
|---|---|---|
| fair_value x2, wide, random | +5.3 | 100% |
| house reader x2, house hider, house fair value | +3.1 | 99% |
| house reader, hider, fair_value, house wide | +3.8 | 96% |
| house reader x2, hider x2 | +3.9 | 100% |
| 5 copies of itself | about 0 | about 50% |

These are against weak opponents. Real candidates earn about +0.4 to +1.05 per round against each other, so expect much lower numbers on the real board.

**Pitfall found:** calling `numpy.linalg.solve` in `end_round` took 100 to 560 ms per call, far over the 0.05 s limit (200 timeouts in one match, which means forfeit). `--check` did not catch it because it only plays 50 rounds. Keep the learner in plain Python and always run a full 200-round match before uploading.

## 7. Ideas and next steps (not yet done)

1. **Upload v1** to get a real Elo against the real field. This is the only honest test. (Needs the owner's approval.)
2. **Take stale quotes deliberately** at steps 2 and 3 (raise the bid to a rival's resting ask when the estimate says it is cheap). The top agents earn +0.3 to +0.5 per round this way.
3. **Test spreads by step.** Evidence for a tight final spread is weak (r = -0.19), so test it, do not assume it.
4. **Step 1 behaviour.** This is where the field first shows its dice.
5. Use more than mids: bid and ask separately, rivals' position skew, and who traded with whom.
6. **Fingerprint opponents** in the first rounds and use an archetype-specific decode. Some seats do reveal at step 0 (zeus, MerchTrader).
7. **Build replay-based opponents** for local simulation, since the house agents are far too weak. Fit imitation models of the top agents from the dataset.
8. Train a policy in `dicesim.py` against those stronger opponents (the example learned agent only reads its own dice and does not beat the boss).
9. Late refinement: randomise slightly if rivals profile my quotes from the public replays.

## 8. How to reproduce

```
cd dice_market
python3 dicemarket.py --check agent.py
python3 dicemarket.py agent.py house/house_reader.py house/house_hider.py examples/fair_value.py examples/wide_quoter.py
python3 tourney.py agent.py                 # local tournaments against the house agents
python3 analysis/passive.py agent.py 150    # estimation error on recorded real matches
python3 analysis/profile.py data/replays.jsonl.gz data/profile.json 8   # per-agent statistics
python3 analysis/charts.py ../docs/data/profile.json ../docs/figures    # the figures above
```

The API token is stored in the macOS Keychain (service name `dice_token`), never in this repo. Load it with
`export DICE_TOKEN=$(security find-generic-password -a "$USER" -s dice_token -w)` before running `dmapi.py`.
`dice_market/data/` is git-ignored because of its size.
