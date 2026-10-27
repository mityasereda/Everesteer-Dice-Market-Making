"""dicesim: a fast vectorised simulator of the dice market game (game v3), for training agents.

The rules are the rules of dicemarket.py. N independent tables run at once in numpy. On each table, the LEARNER
seats (default: seat 0) take your actions. The other seats are the opponents below.

Use:
    env = DiceMarketEnv(n_tables, n_seats=5, rounds=200, seed=0, learner=0,
                        opponents=("fair_value", "wide", "random", "house_reader"))
    env.add_opponent("old_me", policy_fn)   # optional, before reset()
    obs = env.reset()
    obs, reward, done, info = env.step(action)

learner: one seat, or a tuple of seats (self-play: one policy in several seats).
opponents: one name per opponent seat for all tables, or an (n_tables, n_opponent_seats) array of names (per table).
    "fair_value", "wide", "random"   examples/fair_value.py, wide_quoter.py, random_quoter.py.
    "reader"                         +-2 around its dice plus what the others' last sent mids say about their dice.
    "house_fair_value", "house_wide" the house baselines: the same quotes as "fair_value" and "wide".
    "house_reader", "house_hider"    baselines/reader.py and hider.py, in numpy.
    a path to an agent .py file      that agent through the real interface, one copy per table. This is slow:
                                     about 25,000 table-steps a second, against 300,000 or more in numpy.
    a name given to add_opponent     your own policy, for example a frozen copy of your earlier weights.
add_opponent(name, policy_fn): policy_fn(obs) returns (n, 2) quotes. obs is the observation of one seat on n tables.
Action: (N, 2) bid, ask for one learner seat; (N, L, 2) for L learner seats. An invalid quote (outside 12 to 72,
    bid >= ask, wider than 4, NaN) becomes (40, 44), as in the engine. Prices are rounded to 0.5 first.
reward: the learner's P&L of the round on the last step of each round, else 0. (N,) for one seat, (N, L) for L.
info: {"pnl": (N, K) every seat's P&L of the round} on the last step of each round, else {}.
Observation: a dict for one learner seat, a list of dicts (one per seat) for several. env.obs(seat) gives any seat's.
    round, step, seat ints. The observation before each quote; step 0 is the start of a round.
    dice              (N, 2) the seat's dice (die seat and die seat + 6)
    book              (N, K, 2) resting (bid, ask) of every seat after the last step, NaN = empty side
    sent              (N, K, 2) the (clean) quote every seat sent at the last step, NaN at step 0
    sent_round        (N, 4, K, 2) every quote sent in this round so far, NaN for the steps to come
    fills             (N, K) units each seat bought minus units sold at the last step
    position          (N,) the seat's position in this round
    value, all_dice   (N,), (N, 12) of the round that just ended, NaN before the first round ends

Matching, each step (dicemarket.match_step):
    1. take: in random seat order, a new bid takes the best resting ask of another seat at or below it, at the
       resting price (a new ask likewise). The old quote of every seat is still live here.
    2. replace: the new quote replaces the old one; unfilled new sides rest.
    3. cross: resting orders that cross trade. Every seat sends a quote every step, so after the replace every
       resting order is new and a cross is always at the midpoint.
    A fill past +-5 is skipped and that resting side is removed. Ties go at random.
"""
from pathlib import Path

import numpy as np

from dicemarket import load_bot as load_agent

N_DICE, STEPS, LIMIT, MAX_WIDTH, TICK = 12, 4, 5, 4.0, 0.5
PRICE_RANGE = (12.0, 72.0)
DEFAULT_QUOTE = (40.0, 44.0)
ALIASES = {"house_fair_value": "fair_value", "house_wide": "wide"}  # the same quotes as the built-in ones


def clean_quotes(q):
    """(..., 2) quotes; each invalid one becomes DEFAULT_QUOTE (like dicemarket.clean_quote)."""
    q = np.round(np.asarray(q, float) / TICK) * TICK  # the engine's tick
    bid, ask = q[..., 0], q[..., 1]
    ok = (bid >= PRICE_RANGE[0]) & (bid < ask) & (ask <= PRICE_RANGE[1]) & (ask - bid <= MAX_WIDTH + 1e-9)
    return np.where(ok[..., None], q, DEFAULT_QUOTE)


def _room(p, buy):
    return p < LIMIT if buy else p > -LIMIT


def _best(cand, key, rng):
    """Per table: the seat with the lowest key among cand, ties at random. -1 = no candidate."""
    k = np.where(cand, key, np.inf)
    tie = cand & (k == k.min(1, keepdims=True))
    return np.where(tie.any(1), np.where(tie, rng.random(k.shape), -1.0).argmax(1), -1)


def match_step(book, quotes, pos, cash, rng, log=None):
    """One step on N tables. book (N, K, 2) resting (bid, ask), NaN = empty; quotes (N, K, 2) clean new quotes of
    every seat; pos, cash (N, K). All are changed in place: book becomes the book after the step.
    If log is a list, (tables, buyers, sellers, prices, kind) arrays are appended to it for every batch of fills."""
    n, k = pos.shape
    r, seats = np.arange(n), np.arange(k)
    left = quotes.astype(float)
    order = rng.random((n, k)).argsort(1)
    for j in range(k):  # 1) take
        s = order[:, j]
        for side in (0, 1):  # 0: my new bid against resting asks; 1: my new ask against resting bids
            sign = 1.0 if side == 0 else -1.0  # key: the best resting price is the lowest key
            key = sign * book[:, :, 1 - side]
            cand = (_room(pos[r, s], side == 0)[:, None] & (seats != s[:, None])
                    & (key <= sign * left[r, s, side][:, None]))  # NaN compares False
            while cand.any():
                t = _best(cand, key, rng)
                rows = np.flatnonzero(t >= 0)
                ts, ss = t[rows], s[rows]
                ok = _room(pos[rows, ts], side == 1)
                fr, ft, fs = rows[ok], ts[ok], ss[ok]
                price = book[fr, ft, 1 - side]
                pos[fr, fs] += sign
                pos[fr, ft] -= sign
                cash[fr, fs] -= sign * price
                cash[fr, ft] += sign * price
                book[rows, ts, 1 - side] = np.nan  # filled, or removed at the limit
                left[fr, fs, side] = np.nan
                if log is not None:
                    log.append((fr, fs if side == 0 else ft, ft if side == 0 else fs, price, "take"))
                cand[rows, ts] = False
                cand[fr] = False
    book[:] = left  # 2) replace
    while True:  # 3) cross
        bt = _best(~np.isnan(book[:, :, 0]), -book[:, :, 0], rng)
        at = _best(~np.isnan(book[:, :, 1]), book[:, :, 1], rng)
        rows = np.flatnonzero((bt >= 0) & (at >= 0))
        b, a = bt[rows], at[rows]
        hit = book[rows, b, 0] >= book[rows, a, 1]
        rows, b, a = rows[hit], b[hit], a[hit]
        if not len(rows):
            return
        no_b = ~_room(pos[rows, b], True)
        book[rows[no_b], b[no_b], 0] = np.nan
        no_a = ~no_b & ~_room(pos[rows, a], False)
        book[rows[no_a], a[no_a], 1] = np.nan
        ok = ~no_b & ~no_a
        fr, fb, fa = rows[ok], b[ok], a[ok]
        price = (book[fr, fb, 0] + book[fr, fa, 1]) / 2
        if log is not None:
            log.append((fr, fb, fa, price, "cross"))
        pos[fr, fb] += 1
        pos[fr, fa] -= 1
        cash[fr, fb] -= price
        cash[fr, fa] += price
        book[fr, fb, 0] = book[fr, fa, 1] = np.nan




def _builtin(name, own, step, sent_round, rng):
    """Mid and half width of a built-in opponent on every seat. own (N, K): each seat's two dice."""
    fair, unseen = own + 3.5 * (N_DICE - 2), 3.5 * (N_DICE - 2 * own.shape[1])
    if name == "fair_value":
        return fair, 2.0
    if name == "wide":
        return np.full_like(fair, 42.0), 2.0
    if name == "random":
        return fair + rng.uniform(-3, 3, fair.shape), rng.uniform(0.5, 2, fair.shape)
    if step == 0:  # the readers have nothing to read yet; the house hider shows nothing
        return (np.full_like(fair, 42.0) if name == "house_hider" else fair), 2.0
    # each other seat's mid is read as its two dice + 35: the last step for "reader", step 0 for the house agents
    x = np.clip(sent_round[:, step - 1 if name == "reader" else 0].mean(-1) - 35.0, 2, 12)
    return own + x.sum(1, keepdims=True) - x + unseen, 2.0 if name == "reader" else 1.0


# name -> quote(env) returning (mid, half width) for every seat, (N, K) arrays or numbers. An entry may also have
# reset(env) and end_round(env), called when a table uses it.
OPPONENTS = {name: (lambda env, name=name: _builtin(name, env.own, env.step_no, env.sent_round, env.rng))
             for name in ("fair_value", "wide", "random", "reader", "house_reader", "house_hider")}


class DiceMarketEnv:
    def __init__(self, n_tables, n_seats=5, rounds=200, seed=0, opponents=("fair_value", "wide", "random", "reader"),
                 learner=0):
        self.n, self.k, self.rounds = n_tables, n_seats, rounds
        self.learners = tuple(int(s) for s in np.atleast_1d(learner))
        self.others = [s for s in range(n_seats) if s not in self.learners]
        self.opponents, self.fns = opponents, {}
        self.rng = np.random.default_rng(seed)

    def add_opponent(self, name, policy_fn):
        """Register an opponent by name. policy_fn(obs) returns (n, 2) quotes for the n tables in obs."""
        self.fns[name] = policy_fn

    def reset(self):
        names = np.broadcast_to(np.asarray(self.opponents, dtype=object), (self.n, len(self.others)))
        names = np.vectorize(lambda x: ALIASES.get(x, x), otypes=[object])(names)
        self.kinds = sorted(set(names.ravel()))
        self.kind = np.full((self.n, self.k), -1)
        self.kind[:, self.others] = np.vectorize(self.kinds.index)(names)
        self.agents, classes = {}, {}  # (table, seat): a plain Python agent, run through the real interface
        for c, name in enumerate(self.kinds):
            if name in OPPONENTS or name in self.fns:
                continue
            path = Path(name)
            if not path.is_file():
                raise ValueError(f"unknown opponent {name!r}")
            for t, s in np.argwhere(self.kind == c).tolist():
                if path not in classes:
                    classes[path] = type(load_agent(path, s, self.k))
                self.agents[(t, s)] = classes[path](s, self.k)
        self.py_rows = np.array(sorted({t for t, _ in self.agents}), int)
        self._hooks("reset")
        self.round = 0
        self.value, self.all_dice = np.full(self.n, np.nan), np.full((self.n, N_DICE), np.nan)
        self._new_round()
        return self._learner_obs()

    def _hooks(self, method):
        for name in self.kinds:
            fn = getattr(OPPONENTS.get(name), method, None)
            if fn:
                fn(self)

    @staticmethod
    def _call(agent, method, *args):
        try:
            return getattr(agent, method)(*args)
        except Exception:
            return None

    def _new_round(self):
        n, k = self.n, self.k
        self.dice = self.rng.integers(1, 7, (n, N_DICE))
        seats = np.arange(k)
        self.own = (self.dice[:, seats] + self.dice[:, seats + 6]).astype(float)  # (N, K) each seat's two dice
        self.step_no = 0
        self.book, self.sent_round = np.full((n, k, 2), np.nan), np.full((n, STEPS, k, 2), np.nan)
        self.pos, self.cash, self.fills = np.zeros((n, k)), np.zeros((n, k)), np.zeros((n, k))
        self.hist = {t: [] for t in self.py_rows.tolist()}
        for (t, s), a in self.agents.items():
            self._call(a, "new_round", self.round, {s: int(self.dice[t, s]), s + 6: int(self.dice[t, s + 6])})

    def obs(self, seat, rows=None):
        """The observation of one seat, on all tables or on the tables in rows."""
        r = np.arange(self.n) if rows is None else rows
        last = self.sent_round[r, self.step_no - 1] if self.step_no else np.full((len(r), self.k, 2), np.nan)
        return {"round": self.round, "step": self.step_no, "seat": seat, "dice": self.dice[r][:, [seat, seat + 6]],
                "book": self.book[r], "sent": last, "sent_round": self.sent_round[r], "fills": self.fills[r],
                "position": self.pos[r, seat], "value": self.value[r], "all_dice": self.all_dice[r]}

    def _learner_obs(self):
        obs = [self.obs(s) for s in self.learners]
        return obs[0] if len(obs) == 1 else obs

    def _opponent_quotes(self):
        q = np.full((self.n, self.k, 2), np.nan)
        for c, name in enumerate(self.kinds):
            m = self.kind == c
            if name in OPPONENTS:
                mid, half = OPPONENTS[name](self)
                q[m] = np.stack([mid - half, mid + half], -1)[m]
            elif name in self.fns:
                for s in np.flatnonzero(m.any(0)):
                    rows = np.flatnonzero(m[:, s])
                    q[rows, s] = self.fns[name](self.obs(s, rows))
        for (t, s), a in self.agents.items():
            out = self._call(a, "quote", self.step_no, self.hist[t])  # shortcut: no deep copy of the history
            try:
                q[t, s] = float(out[0]), float(out[1])
            except Exception:
                pass  # NaN: the forced quote
        return q

    def _record(self, q, log):
        """Append this step to the history of every table with a Python agent (the engine's format)."""
        trades = {t: [] for t in self.hist}
        for rows, b, s, p, kind in log:
            keep = np.isin(rows, self.py_rows)
            for t, bi, si, pi in zip(rows[keep].tolist(), b[keep].tolist(), s[keep].tolist(), p[keep].tolist()):
                trades[t].append({"buyer": bi, "seller": si, "price": pi, "kind": kind})
        for t in self.hist:
            bk = self.book[t].tolist()
            self.hist[t].append({"sent": {s: tuple(x) for s, x in enumerate(q[t].tolist())},
                                 "quotes": {s: tuple(None if x != x else x for x in bk[s])
                                            for s in range(self.k) if bk[s][0] == bk[s][0] or bk[s][1] == bk[s][1]},
                                 "trades": trades[t]})

    def step(self, action):
        """action: (N, 2) for one learner seat, (N, L, 2) for L learner seats."""
        q = self._opponent_quotes()
        q[:, self.learners] = np.asarray(action, float).reshape(self.n, len(self.learners), 2)
        q = clean_quotes(q)
        p0, log = self.pos.copy(), ([] if self.agents else None)
        match_step(self.book, q, self.pos, self.cash, self.rng, log)
        self.fills, self.sent_round[:, self.step_no] = self.pos - p0, q
        if self.agents:
            self._record(q, log)
        self.step_no += 1
        reward, info, done = np.zeros((self.n, len(self.learners))), {}, False
        if self.step_no == STEPS:  # settle at the value
            self.value, self.all_dice = self.dice.sum(1).astype(float), self.dice.astype(float)
            pnl = self.cash + self.pos * self.value[:, None]
            reward, info = pnl[:, self.learners], {"pnl": pnl}
            self._hooks("end_round")
            for (t, s), a in self.agents.items():
                self._call(a, "end_round", {"value": int(self.value[t]), "dice": self.dice[t].tolist(),
                                            "pnl": pnl[t].tolist(), "history": self.hist[t]})
            self.round += 1
            done = self.round == self.rounds
            if not done:
                self._new_round()
        return self._learner_obs(), (reward[:, 0] if len(self.learners) == 1 else reward), done, info
