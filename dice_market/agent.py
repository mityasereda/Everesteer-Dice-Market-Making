"""Dice market agent v8: an adaptive reader that quotes the field's three-level step 0, quotes wide early and takes stale quotes.

Idea: the value is the sum of 12 dice and I see 2. The other seats' quotes carry information about the dice I
cannot see. At every step I estimate the value as a linear function of my own dice and the mids the other seats
sent in earlier steps. The weights start at PRIOR (a fit on real leaderboard matches) and are refined during the
match by ridge regression: after every round I learn the true value, so I can check how well each seat's quotes
predicted it and re-weight them. The ridge strength (lam) is how many rounds of belief the prior is worth.

Step 0 (v8): the quote has three levels like the agents ranked 15 to 40: dice sum below 7 -> mid 42 - P, above 7 -> mid
42 + P, exactly 7 -> 40 / 44, width 4, with P = 6 (a far quote wins priority in the engine's cross phase; the default
40 / 44 lost 1.2 per round at steps 0 to 1 on 28 Oct). v7 hid at step 0 (hide0=True); v6's smooth lean (step0_levels=None)
revealed everything and lost at steps 1 to 3. At steps 1 to 3 the quote is the
estimate -/+ a half width, shifted against my position so inventory is worked off. Optional take logic: if a rival's
resting quote is stale against my estimate by more than a margin, my quote is moved to meet it, so I trade at its price.
Pure Python, about 0.1 ms per call.
"""
MEAN = 42.0
CLIP = 15.0                 # later mids can say more about the dice than step-0 mids (clipped at PARAMS["clip0"])
PARAMS = dict(
    half=(2.0, 2.0, 2.0, 1.0),   # half width of my quote, by step: wide at steps 1-2 like the field, tighter at step 3 (replay-tested)
    skew=0.25,                   # price shift per unit of position
    lam=100.0,                   # ridge strength towards the prior weights (v5 and v7 used 30; the v8 prior is a fit on today's field)
    mode="seat",                 # what the regression learns: "seat" = one weight per rival seat (v5), "shared" = one weight
                                 # per step on the rivals' summed mids, "both" = shared plus per-seat corrections
    lam_seat=300.0,              # ridge strength of the per-seat corrections in "both" mode (strong: they learn slowly)
    hide0=False,                 # True: step 0 quotes 40 / 44 and shows nothing (v7). False: quote informed (see step0_levels)
    step0_levels=6.0,            # informed step 0 with three levels: mid 42 -/+ this when my dice are below / above 7 (None: smooth lean)
    clip0=3.0,                   # a rival's step-0 mid is clipped at +-clip0 before use (the field's pushes run from 4 to 9, so
                                 # only its sign and rough size carry information; held-out accuracy was best at 3 to 4)
    step0_k=1.5,                 # smooth informed step 0 (step0_levels=None): centre = 42 + step0_k * (my dice - 7)
    step0_w=4.0,                 # informed step 0: full quote width
    take=(None, 0.5, 0.25, 0.25),    # by step: margin for taking a stale rival quote (None = off). If a rival's resting ask is at
                                     # least this far below my centre, my bid is raised to it (resting bid: ask lowered to it)
)
# Starting weights per step: (weight on my dice, weight on the sum of the other seats' mids at step 0, 1, ...). v8: fitted on
# 28 Oct matches, seat-rounds that quoted informed at step 0, step-0 mids clipped at +-3 (analysis/fit_prior.py ... informed 3).
# The v5 / v7 prior (fitted on 23-26 Oct, when most of the field hid) was
# {1: (1.0, [0.72]), 2: (0.75, [-0.16, 0.26]), 3: (0.26, [0.05, -0.03, 0.26])}.
PRIOR = {0: (1.0, []), 1: (1.0, [0.74]), 2: (0.30, [0.08, 0.21]), 3: (0.28, [0.18, -0.07, 0.26])}


def _dot(a, b):
    return sum(u * v for u, v in zip(a, b))


def _mid(q):
    return (float(q[0]) + float(q[1])) / 2.0


class Agent:
    def __init__(self, seat, n_seats):
        self.seat, self.n = seat, n_seats
        self.others = [s for s in range(n_seats) if s != seat]
        self.p = dict(PARAMS)
        self.own = 7.0
        self.P, self.b, self.w0, self.w = [], [], [], []   # P is the inverse of the ridge matrix, kept by RLS updates
        no = len(self.others)
        mode = self.p["mode"]
        for s in range(4):
            own_w, mid_w = PRIOR[s] if PRIOR is not None else (1.0, [1.0] if s else [])
            w0, lams = [0.0, own_w], [self.p["lam"]] * 2           # constant, my dice
            if mode in ("shared", "both"):
                w0 += [mid_w[t] for t in range(s)]                 # summed mids of the others, one feature per earlier step
                lams += [self.p["lam"]] * s
            if mode in ("seat", "both"):
                for t in range(s):
                    w0 += [mid_w[t] if mode == "seat" else 0.0] * no   # each rival's own mid at step t
                    lams += [self.p["lam"] if mode == "seat" else self.p["lam_seat"]] * no
            d = len(w0)
            self.w0.append(w0)
            self.w.append(list(w0))
            self.P.append([[1.0 / lams[i] if i == j else 0.0 for j in range(d)] for i in range(d)])
            self.b.append([0.0] * d)

    def new_round(self, round_no, my_dice):
        self.own = float(sum(my_dice.values()))

    def feats(self, step, history):
        x = [1.0, self.own - 7.0]
        mids = []
        for t in range(step):
            clip = self.p["clip0"] if t == 0 else CLIP
            sent = history[t]["sent"]
            mids.append([max(-clip, min(clip, _mid(sent[j]) - MEAN)) if sent.get(j) else 0.0 for j in self.others])
        mode = self.p["mode"]
        if mode in ("shared", "both"):
            x += [sum(m) for m in mids]
        if mode in ("seat", "both"):
            for m in mids:
                x += m
        return x

    def position(self, history):
        return sum((tr["buyer"] == self.seat) - (tr["seller"] == self.seat) for h in history for tr in h["trades"])

    def estimate(self, step, history):
        """My estimate of the value from my dice and the others' mids in the steps before this one."""
        return MEAN + _dot(self.w[step], self.feats(step, history))

    def quote(self, step, history):
        if step == 0:
            if self.p["hide0"]:
                return (MEAN - 2.0, MEAN + 2.0)
            w = self.p["step0_w"]
            P = self.p["step0_levels"]
            if P is not None:
                if self.own == 7.0:
                    return (MEAN - 2.0, MEAN + 2.0)
                bid = round((MEAN + (P if self.own > 7.0 else -P) - w / 2.0) * 2.0) / 2.0
                bid = min(max(bid, 12.0), 72.0 - w)
                return (bid, bid + w)
            bid = round((MEAN + self.p["step0_k"] * (self.own - 7.0) - w / 2.0) * 2.0) / 2.0
            bid = min(max(bid, 12.0), 72.0 - w)
            return (bid, bid + w)
        centre = self.estimate(step, history) - self.p["skew"] * self.position(history)
        h = self.p["half"][step]
        bid = round((centre - h) * 2.0) / 2.0                 # snap the bid to the 0.5 grid, then ask = bid + width,
        bid = min(max(bid, 12.0), 72.0 - 2.0 * h)             # so rounding can never narrow or invalidate the quote
        ask = bid + 2.0 * h
        m = self.p["take"][step]
        if m is not None and history:
            book = history[-1]["quotes"]                       # the rivals' resting quotes entering this step
            asks = [q[1] for j, q in book.items() if j != self.seat and q[1] is not None]
            bids = [q[0] for j, q in book.items() if j != self.seat and q[0] is not None]
            if asks and bid < min(asks) <= centre - m:         # a stale ask far below my value: bid up to it and buy at its price
                bid = min(asks)
            elif bids and ask > max(bids) >= centre + m:       # a stale bid far above my value: ask down to it and sell at its price
                ask = max(bids)
            if not (bid < ask <= bid + 4.0):                   # keep the quote valid whatever happened
                bid = round((centre - h) * 2.0) / 2.0
                bid = min(max(bid, 12.0), 72.0 - 2.0 * h)
                ask = bid + 2.0 * h
        return (bid, ask)

    def end_round(self, result):
        y = result["value"] - MEAN
        hist = result["history"]
        for s in range(4):
            x = self.feats(s, hist)
            P, b = self.P[s], self.b[s]
            px = [_dot(row, x) for row in P]               # Sherman-Morrison: P <- P - (Px)(Px)' / (1 + x'Px)
            den = 1.0 + _dot(x, px)
            for i in range(len(x)):
                row = P[i]
                for j in range(len(x)):
                    row[j] -= px[i] * px[j] / den
            r = y - _dot(self.w0[s], x)
            for i in range(len(x)):
                b[i] += x[i] * r
            self.w[s] = [a + c for a, c in zip(self.w0[s], (_dot(row, b) for row in P))]
