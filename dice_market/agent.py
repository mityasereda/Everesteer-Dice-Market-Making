"""Dice market agent v5: an adaptive reader.

Idea: the value is the sum of 12 dice and I see 2. The other seats' quotes carry information about the dice I
cannot see. At every step I estimate the value as a linear function of my own dice and the mids the other seats
sent in earlier steps. The weights start at PRIOR (a fit on real leaderboard matches) and are refined during the
match by ridge regression: after every round I learn the true value, so I can check how well each seat's quotes
predicted it and re-weight them. The ridge strength (lam) is how many rounds of belief the prior is worth.

Step 0 shows nothing (the default quote 40 / 44). At steps 1 to 3 the quote is the estimate -/+ a half width, shifted
against my position so inventory is worked off. Pure Python, about 0.1 ms per call.
"""
MEAN = 42.0
CLIP0, CLIP = 5.0, 15.0     # a step-0 mid says at most +-5 about a pair of dice; later mids can say more
PARAMS = dict(
    half=(2.0, 2.0, 1.0, 1.0),   # half width of my quote, by step (tuned on adaptive copies of real agents)
    skew=0.25,                   # price shift per unit of position
    lam=30.0,                    # ridge strength towards the prior weights, in rounds (30 beat 100 on rank in every mix)
    hide0=True,                  # step 0: quote 40 / 44 so my first quote shows nothing of my dice
)
# Starting weights per step: (weight on my dice, weight on each other seat's mid at step 0, 1, ...). They are the
# pooled least-squares fit on 14,000 real leaderboard matches. The field hides at step 0, so step-0 mids say little.
# None gives the old "house reader" prior (every step-0 mid read as that seat's dice).
PRIOR = {0: (1.0, []), 1: (1.0, [0.72]), 2: (0.75, [-0.16, 0.26]), 3: (0.26, [0.05, -0.03, 0.26])}


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
        for s in range(4):
            d = 2 + s * len(self.others)
            w0 = [0.0] * d
            own_w, mid_w = PRIOR[s] if PRIOR is not None else (1.0, [1.0] if s else [])
            w0[1] = own_w                                 # my own dice
            for t, wt in enumerate(mid_w):
                for k in range(len(self.others)):
                    w0[2 + t * len(self.others) + k] = wt  # the others' mids at step t
            self.w0.append(w0)
            self.w.append(list(w0))
            self.P.append([[1.0 / self.p["lam"] if i == j else 0.0 for j in range(d)] for i in range(d)])
            self.b.append([0.0] * d)

    def new_round(self, round_no, my_dice):
        self.own = float(sum(my_dice.values()))

    def feats(self, step, history):
        x = [1.0, self.own - 7.0]
        for t in range(step):
            clip = CLIP0 if t == 0 else CLIP
            sent = history[t]["sent"]
            for j in self.others:
                q = sent.get(j)
                x.append(max(-clip, min(clip, _mid(q) - MEAN)) if q else 0.0)
        return x

    def position(self, history):
        return sum((tr["buyer"] == self.seat) - (tr["seller"] == self.seat) for h in history for tr in h["trades"])

    def estimate(self, step, history):
        """My estimate of the value from my dice and the others' mids in the steps before this one."""
        return MEAN + _dot(self.w[step], self.feats(step, history))

    def quote(self, step, history):
        if step == 0 and self.p["hide0"]:
            return (MEAN - 2.0, MEAN + 2.0)
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
