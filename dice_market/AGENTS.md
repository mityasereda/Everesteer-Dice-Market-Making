## Dice Market Making

Write a trading agent that makes markets on dice. Your agent plays against other candidates' agents. We interview the top 20 on the leaderboard. The role is full time, 5+ days a week in person, in Budapest or London. AI tools are allowed.

### Start here
1. **Download the starter pack.** It has the game engine, a training simulator, a template, example agents, house agents to practise against, and `AGENTS.md` (a guide you can give to your AI coding tool).
2. **Build your agent.** Copy `agent_template.py` to `agent.py`. Check it:
   - Mac: `python3 dicemarket.py --check agent.py`
   - Windows: `py dicemarket.py --check agent.py`
3. **Upload it on Play.** It plays 3 matches against the boss. Beat the boss in 2 of 3 and your agent joins the leaderboard.
4. **Answer the four questions.** They open when the challenge closes, on 1 November 2026 at 00:00 London time. We do not score them.

### The game in one minute
- **12 dice.** Each round, all 12 are rolled. The value is their sum (on average 42).
- **You see 2 dice.** Each seat sees two different dice. With 5 seats, 2 dice are seen by nobody.
- **4 quoting steps per round.** Every step you must send a bid and an ask for 1 unit, at most 4 apart. Prices are rounded to 0.5. If you send nothing, or something invalid, you are given 40 / 44.
- **Quotes rest.** Your quote stays in the book until your next one replaces it, and it is still live while the others send theirs. A new order that meets a resting quote trades at the resting price. So a quote that has gone stale can be picked off. Two new orders that meet trade at the midpoint.
- **Nobody else trades.** There is no noise trader. Every point you win, another agent loses.
- **Settle every round.** Positions are valued at the round's value and reset to zero. The limit is 5 units long or short within a round.
- **After each round** you learn the value and all 12 dice.
- **A match** is 200 rounds with 5 seats. Your result is your profit.

### Ranking
The leaderboard ranks agents by Elo. Every agent starts at 1200. In each match, you win against a seat if your profit is higher. House agents sit on the leaderboard for comparison, but they do not count for the interview.

### Your code
- One file, `agent.py`, with a class `Agent` and four methods: `new_round`, `quote`, `end_round`, plus `__init__`.
- Python standard library and numpy only. No files, network, threads or subprocesses. The upload scan refuses `os`, `sys`, `getattr` and similar names.
- Each call must return in 0.05 seconds.
- Optional: train a model and upload a `.zip` with `agent.py` and `weights.npz` (20 MB maximum).
- Store up to 10 versions. Your newest version that beats the boss is active.
- If you upload an 11th version, the server deletes one old version. It deletes the oldest failed version first, then the oldest passing version. It never deletes the active version.
- To delete a version yourself, use Delete on Play or `POST /api/v1/agents/delete`.

### Fair play
Use one account. Do not share code or coordinate with other candidates. We disqualify everyone involved.

---

## Reference for AI coding tools

### Local commands
Open a terminal in the unzipped `dice_market` folder. Use Python 3.9 or newer. numpy is optional for the engine. You need it for weights and for `dicesim.py`: `python3 -m pip install numpy` (Mac) or `py -m pip install numpy` (Windows).

Mac or Linux:

    cp agent_template.py agent.py
    python3 dicemarket.py --check agent.py
    python3 dicemarket.py agent.py examples/fair_value.py examples/wide_quoter.py house/house_reader.py house/house_hider.py

Windows (PowerShell or Command Prompt). If `py` is not found, use `python`:

    copy agent_template.py agent.py
    py dicemarket.py --check agent.py
    py dicemarket.py agent.py examples\fair_value.py examples\wide_quoter.py house\house_reader.py house\house_hider.py

Options for a local match: `--rounds N` (default 100), `--seed S` (default 0). A local match has 2 to 6 agents. The site plays 5.

`--check` plays 50 rounds against the example agents. It prints your profit per round, trades, errors and slow answers. Fix every error and slow answer before you upload.

### Files in the pack
| File | Use |
|---|---|
| `AGENTS.md` | This guide. |
| `dicemarket.py` | The game engine and the local match runner. The site uses the same engine. It is the game. |
| `dicesim.py` | A fast simulator for training. For its use, see the module docstring. |
| `agent_template.py` | A simple agent with the four methods. Copy it to `agent.py`. |
| `dmapi.py` | The API command line: upload, versions, leaderboard, matches, replays and the dataset. See API. |
| `replays.py` | Reads the replay dataset into the full match shape. See The replay dataset. |
| `examples/fair_value.py` | Quotes +-2 around the expected value from its own 2 dice. |
| `examples/wide_quoter.py` | Ignores its dice. Quotes 40 / 44 every step. |
| `examples/random_quoter.py` | A random quote near its own expected value. |
| `examples/learned/agent.py` | Quotes with trained weights from `WEIGHTS`. |
| `examples/learned/train.py` | Trains the weights for `agent.py` in `dicesim.py`. |
| `examples/learned/weights.npz` | Trained weights for `agent.py`. The folder has the same shape as an upload. |
| `house/house_fair_value.py` | House agent: quotes around its own estimate of the value. |
| `house/house_wide_market_maker.py` | House agent: quotes a wide market around the mean. |
| `house/house_reader.py` | House agent: reads the quotes of the other seats and quotes tight. |
| `house/house_hider.py` | House agent: shows little at step 0, then reads the other seats. |

The pack does not have house pro or the boss.

### Rules in detail
- The 12 dice are numbered 0 to 11. Each die is 1 to 6. All 12 are rolled fresh every round.
- The value is the sum of the 12 dice: 12 to 72, mean 42, standard deviation about 5.9.
- Seat `i` sees die `i` and die `i + 6`. With 5 seats, no seat sees dice 5 and 11.
- A round has 4 steps (0 to 3). In each step, every seat sends one quote: a bid and an ask, 1 unit each.
- A valid quote has `12 <= bid < ask <= 72` and `ask - bid <= 4`. Prices are rounded to 0.5 before the checks.
- A missing, invalid or slow quote, an error, or a stopped agent gives the forced quote `(40, 44)`.
- There is one book. It is empty at the start of each round. A seat has at most one bid and one ask in it.
- There is no noise trader. The game is zero-sum: the round profits of all seats add to 0.
- The limit is 5 units long or short within a round.
- At the end of a round, every position settles at the value. Round profit = cash from trades + position x value. Positions then reset to 0.
- After each round, every agent learns the value, all 12 dice, the profit of each seat and the full history.
- A match is 200 rounds. Your match result is the sum of your round profits.

### The order in each step
The engine collects the quotes of all seats, then matches them in this order:

1. **Take.** The seats go in a random order. A seat's new bid takes the best resting ask of another seat at or below it. The trade is at the resting price. A new ask takes the best resting bid at or above it in the same way. "Resting" means posted in an earlier step. Your old quote is still live here, before your new one replaces it. So another seat can pick off your stale quote.
2. **Replace.** Your new quote replaces your old one. The sides of your new quote that did not trade start to rest.
3. **Cross.** If resting orders now cross, they trade. The price is the price of the older order. If both are from this step, the price is the midpoint. This repeats until nothing crosses.

Each fill is 1 unit. Ties at the same price break at random. If a fill would take a seat past 5 long or short, it does not occur, and that resting side is removed.

### The interface
```python
WEIGHTS = None  # dict[str, numpy.ndarray] if weights.npz sits next to agent.py, else None

class Agent:
    def __init__(self, seat: int, n_seats: int) -> None:
        # Once per match. seat is 0 .. n_seats - 1.
        ...

    def new_round(self, round_no: int, my_dice: dict[int, int]) -> None:
        # Start of each round. round_no is 0 .. 99. my_dice is {seat: value, seat + 6: value}.
        ...

    def quote(self, step: int, history: list[dict]) -> tuple[float, float]:
        # Each of the 4 steps. step is 0 .. 3. Return (bid, ask).
        ...

    def end_round(self, result: dict) -> None:
        # End of each round.
        ...
```

`history` lists the earlier steps of this round (empty at step 0). Each step is:

```python
{
    "sent":   {seat: (bid, ask)},             # dict[int, tuple[float, float]]: the quote each seat sent, after the forced-quote rule
    "quotes": {seat: (bid, ask)},             # the resting book after the step; a filled side is None; a seat with no side left is not in it
    "trades": [{"buyer": int, "seller": int, "price": float, "kind": "take" | "cross"}],
}
```

- `kind` is `"take"` when a new order picked off a resting quote at its price. It is `"cross"` when two resting orders crossed (part 3 of the order above).
- `result` is `{"value": int, "dice": list[int] (all 12, by index), "pnl": list[float] (round profit of each seat), "history": list[dict] (all 4 steps)}`.
- The engine gives each call a copy. Changes you make to `history` or `result` have no effect.
- You can return a list `[bid, ask]` or numpy numbers. The engine converts them to floats.

### Limits and the sandbox
- Each call must return in 0.05 seconds. The engine ignores a slower answer and uses the forced quote. It counts a slow answer.
- Use only the Python standard library and numpy.
- Allowed: `math`, `random`, `statistics`, `collections`, `itertools`, `functools`, `numpy`, and other pure computation.
- Not allowed: files, the network, threads, subprocesses, and reading the internal state of the game while it runs (for example the dice of other seats).
- The upload scan refuses these names in your code: `inspect`, `gc`, `_getframe`, `modules`, `open`, `socket`, `subprocess`, `threading`, `multiprocessing`, `system`, `__import__`, `importlib`, `ctypes`, `builtins`, `globals`, `f_back`, `f_locals`, `exec`, `eval`.
- On the site, each agent runs in its own sealed process. It has no network and cannot see other agents. `print()` output is discarded.

### Upload
- Upload `agent.py`, or one `.zip` with `agent.py` and `weights.npz` at the top level.
- `agent.py` can have 64 KB. A `.zip` can have 20 MB.
- You can upload 20 times each day. Wait 30 seconds between uploads. A failed attempt counts.
- Give each upload a name, for example `tight spread v2`. You can store up to 10 versions.

### Weights and training
1. Save arrays with `numpy.savez("weights.npz", name=array, ...)`. Do not save Python objects: the engine loads with `allow_pickle=False`.
2. Put `agent.py` and `weights.npz` at the top level of one .zip file. Upload the .zip on Play.
3. The engine loads the file into the module global `WEIGHTS` after your module runs. `WEIGHTS` is a dict of arrays. It is `None` if there is no file.

To use weights in a local match, keep `weights.npz` next to `agent.py`.

`dicesim.py` runs many tables at once in numpy, for training. For its API, see the module docstring.

The simulator gives you the tools. A strong opponent is yours to build:
- Each table has its own opponents. Give an array of names, one row for each table.
- The built-in opponents are the simple examples (fair value, wide, random) and the house sparring agents (house fair value, house wide market maker, house reader, house hider).
- To use an agent file as an opponent, give its path. A file agent runs much slower than a numpy opponent.
- To freeze your earlier weights as an opponent, call `env.add_opponent(name, policy_fn)` before `reset()`. Then use that name.
- For self-play, give `learner` as a tuple of seats. Then send one action for each of these seats.

`examples/learned/train.py` trains the small policy of `examples/learned/agent.py` against the simple examples and writes `weights.npz` next to itself. Run:

    python3 examples/learned/train.py

The example reads only its own dice and its position. It does not read the other agents' quotes, so it does not beat the boss. Reading the table, and training against stronger opponents, is your work. To try it, run `python3 dicemarket.py --check examples/learned/agent.py`. To upload it, put `agent.py` and `weights.npz` from that folder in one .zip.

### The replay dataset
Every match played on the leaderboard is in `replays.jsonl.gz` (Downloads on the Docs page). One line per match, stored in a compact form: prices as integer half-point ticks, no profit or position (they follow from the trades). `replays.py` in the pack reads it into the full shape: `labels` (the agents' names), `kinds` (`candidate` or `house`), `seats` (result per seat) and `rounds`. Each round has `value`, `dice` (all 12), `pnl` per seat, `positions` per seat and `steps`, each with `sent`, `quotes` (the resting book) and `trades`, exactly as the live `history`. Read it with:

    import replays
    matches = list(replays.load("replays.jsonl.gz"))

Use it to learn how the other agents quote. The file grows while the challenge runs. To get only the new matches, use `dmapi.py dataset --since TIME` (see API).

### API
Scripts can use the site through an HTTP API. The command line `dmapi.py` (in the pack) calls it for you.

**The token.**
1. On Play, under API token, click Create token.
2. Copy the token. It starts with `dm_`. We show it once. The server keeps only a hash of it.
3. Send it in each request: `Authorization: Bearer <token>`.

Rotate token makes a new token and stops the old one. Revoke stops the token. Do not share your token: it can upload as you.

**Base URL:** `https://dice.everesteer.ai/api/v1/`. All answers are JSON, except `dataset`. An error is `{"error": "<message>"}`.

| Status | Meaning |
|---|---|
| 400 | The request is not valid. The message says why. |
| 401 | The token is missing, wrong or revoked. |
| 403 | The challenge is closed (uploads and changes only). |
| 404 | No such endpoint or replay. Replays are kept for the 30 newest matches. |
| 429 | Too many requests or uploads. Wait the seconds in the `Retry-After` header or the message. |

**Limits.** 60 requests each minute for each token. The upload limits are the same as on Play: 20 uploads each day, 30 seconds between uploads, 64 KB for `agent.py`, 20 MB for a `.zip`.

**`GET /api/v1/me`**: you and your versions.

```json
{"handle": "ann", "active": "a1b2c3", "uploads_left_today": 18,
 "versions": [{"id": "a1b2c3", "name": "tight spread v2", "uploaded_at": "2026-10-23T09:00:00+00:00",
               "passed": true, "elo": 1234, "matches": 12}]}
```

`elo` is `null` until the version plays a leaderboard match.

**`POST /api/v1/agents`**: upload a version. It plays the boss, as on Play.
- Body: the bytes of `agent.py` with `Content-Type: text/plain`, or a `.zip` with `Content-Type: application/zip`.
- Optional header `X-Agent-Name`: the version name, URL-encoded, 30 characters maximum.
- The answer comes after the 3 matches against the boss (5 to 20 seconds).

```json
{"gate": {"passed": true, "reason": "...", "scan_hits": [],
          "matches": [{"you": 1.25, "boss": 0.4, "forfeit": null}]}}
```

If `scan_hits` is not empty, the code scan refused the file. A passing version becomes active.

**`POST /api/v1/agents/active`**: make a stored, passing version active. Body: `{"id": "a1b2c3"}`. The answer is your profile, as on Play.

**`POST /api/v1/agents/delete`**: delete a stored version that is not active. Body: `{"id": "a1b2c3"}`. The answer is your profile. The version leaves the leaderboard. Its replays stay.

**`GET /api/v1/leaderboard`**: the rows of the leaderboard page, by Elo.

```json
{"rows": [{"rank": 1, "label": "ann", "agent": "tight spread v2", "kind": "you", "elo": 1290, "matches": 40,
           "pnl_per_round": 0.84, "se": 0.12, "provisional": false}]}
```

`kind` is `you`, `candidate` or `house`.

**`GET /api/v1/matches`**: the 30 newest matches. Add `?mine=1` to keep only matches with your agents.

```json
{"matches": [{"id": "9f2e4a1b0c", "at": "2026-10-23T09:10:00+00:00",
              "seats": [{"label": "ann", "kind": "you", "pnl_per_round": 0.84, "forfeit": null}]}]}
```

**`GET /api/v1/replays/{id}`**: one of those matches in full: `id`, `at`, `table` (seat, label, kind), `seed`, `seats` (result per seat) and `rounds` (as in the dataset).

**`GET /api/v1/dataset`**: `replays.jsonl.gz`, every match played on the leaderboard (gzip, one JSON line per match). Add `?since=2026-10-23T12:00:00Z` to get only the matches played after that time. Keep the `at` of the last match you have and use it as `since` next time.

**The command line.** Python 3.9 or newer, standard library only.

Mac or Linux:

    export DICE_TOKEN=dm_...
    python3 dmapi.py me
    python3 dmapi.py upload agent.py --name "tight spread v2"
    python3 dmapi.py upload agent.zip --name "learned v1"
    python3 dmapi.py activate a1b2c3
    python3 dmapi.py leaderboard
    python3 dmapi.py matches --mine
    python3 dmapi.py replay 9f2e4a1b0c -o replay.json
    python3 dmapi.py dataset -o replays.jsonl.gz
    python3 dmapi.py dataset -o new.jsonl.gz --since 2026-10-23T12:00:00Z

Windows: use `set DICE_TOKEN=dm_...` (Command Prompt) or `$env:DICE_TOKEN="dm_..."` (PowerShell), and `py` for `python3`.

Options: `--token` instead of `DICE_TOKEN`; `--url` or `DICE_URL` for another server; `--json` prints the raw answer.

curl:

    curl -H "Authorization: Bearer $DICE_TOKEN" https://dice.everesteer.ai/api/v1/me
    curl -H "Authorization: Bearer $DICE_TOKEN" -H "Content-Type: text/plain" -H "X-Agent-Name: tight%20spread%20v2" \
      --data-binary @agent.py https://dice.everesteer.ai/api/v1/agents
    curl -H "Authorization: Bearer $DICE_TOKEN" -o replays.jsonl.gz https://dice.everesteer.ai/api/v1/dataset

### Forfeit
An agent forfeits a match if one of these occurs:
- It gives no answer in 1 second.
- It crashes.
- It has 50 errors or slow answers in one match.

A forfeited agent plays no more rounds. The engine posts the forced quote for it. It loses 10 for each remaining round. The other seats keep their results.

### The boss and versions
- Your agent plays 3 matches against the boss and three example agents. Each attempt uses new seeds.
- To pass: beat the boss's profit in at least 2 of the 3 matches, with no errors, no slow answers and no forfeit.
- Your newest passing version becomes active. If an upload fails, your last passing version stays active.
- To switch, make an older passing version active on Play.
- Each version has its own Elo. A new version starts at 1200.

### The leaderboard and Elo
- A match has 5 seats. It counts as 10 games, one for each pair of seats. Win 1, draw 0.5, loss 0. You win against a seat if your match profit is higher.
- Expected score against one opponent: `E = 1 / (1 + 10^((R_opp - R) / 400))`.
- After a match: `R' = R + K / 4 * sum(S - E)`. The sum covers your 4 opponents.
- K is 40 for your first 10 matches. After that, K is 20.
- Elo shows "?" until 10 matches.
- The leaderboard picks your opponents from the 6 agents nearest your Elo.
- House agents sit on the leaderboard for comparison. They do not count for the interview.
- Replays are public. Agent code is private.

### The four questions
They open when the challenge closes, on 1 November 2026 at 00:00 London time. Give a number and one reason for each. We do not score them. We talk about them with you later.
- Two thirds: a whole number from 0 to 100. The best number is closest to two thirds of the average of all candidates.
- Guess the average: any number from 0 to 100. The best guess is closest to the average of all Two thirds numbers.
- Lowest unique number: a whole number from 1 to 100. The best number is the lowest one that no other candidate picks.
- Traveller's dilemma: a whole number from 2 to 100. You play every other candidate. Same number: both get it. Different numbers: both get the lower number. The lower picker gets +2. The higher picker gets -2.

You can also upload one page of notes as a PDF (2 MB maximum). We do not score the notes.
