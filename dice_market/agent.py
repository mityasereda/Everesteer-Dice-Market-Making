"""Dice market agent v6: an adaptive reader.

Idea: the value is the sum of 12 dice and I see 2. The other seats' quotes carry information about the dice I
cannot see. At every step I estimate the value as a linear function of my own dice and the mids the other seats
sent in earlier steps. The weights start at PRIOR (a fit on real leaderboard matches) and are refined during the
match by ridge regression: after every round I learn the true value, so I can check how well each seat's quotes
predicted it and re-weight them. The ridge strength (lam) is how many rounds of belief the prior is worth.

Step 0 is quoted informed: centre 42 + 1.5 x (my dice - 7), width 4. The default 40 / 44 is crossed at the midpoint by
informed rivals, which is where v5 lost its money. At steps 1 to 3 the quote is the estimate -/+ a half width, shifted
against my position so inventory is worked off. Pure Python, about 0.1 ms per call.
"""
MEAN = 42.0
CLIP0, CLIP = 5.0, 15.0     # a step-0 mid says at most +-5 about a pair of dice; later mids can say more
PARAMS = dict(
    half=(2.0, 2.0, 1.0, 1.0),   # half width of my quote, by step (tuned on adaptive copies of real agents)
    skew=0.25,                   # price shift per unit of position
    lam=100.0,                   # ridge strength towards the prior weights (100 beat 30 on held-out real matches, all phases)
    mode="both",                 # what the regression learns: "seat" = one weight per rival seat (v5), "shared" = one weight
                                 # per step on the rivals' summed mids, "both" = shared plus per-seat corrections
    lam_seat=300.0,              # ridge strength of the per-seat corrections in "both" mode (strong: they learn slowly)
    hide0=False,                 # True: step 0 quotes 40 / 44 and shows nothing. False: quote informed (see step0_k)
    step0_k=1.5,                 # informed step 0: centre = 42 + step0_k * (my dice - 7); the leaders lean 1.5 to 2.2
    step0_w=4.0,                 # informed step 0: full quote width
)
# Starting weights per step: (weight on my dice, weight on each other seat's mid at step 0, 1, ...). Pooled least-squares
# fit on 2,200 real matches played on 27 Oct 2026 (analysis/fit_prior.py), when about half the field already quoted
# informed at step 0. (The first fit, on 23-26 Oct when almost everyone hid, was 0.72 / 0.75 / 0.26 ...)
# None gives the old "house reader" prior (every step-0 mid read as that seat's dice).
PRIOR = {0: (1.0, []), 1: (1.00, [0.57]), 2: (0.55, [0.05, 0.19]), 3: (0.30, [0.10, -0.06, 0.26])}


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
            clip = CLIP0 if t == 0 else CLIP
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
            bid = round((MEAN + self.p["step0_k"] * (self.own - 7.0) - w / 2.0) * 2.0) / 2.0
            bid = min(max(bid, 12.0), 72.0 - w)
            return (bid, bid + w)
        centre = self.estimate(step, history) - self.p["skew"] * self.position(history)
        h = self.p["half"][step]
        bid = round((centre - h) * 2.0) / 2.0                 # snap the bid to the 0.5 grid, then ask = bid + width,
        bid = min(max(bid, 12.0), 72.0 - 2.0 * h)             # so rounding can never narrow or invalidate the quote
        return (bid, bid + 2.0 * h)

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
