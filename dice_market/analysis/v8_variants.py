N = None
VARIANTS = {
    "v8 (defaults)": {},
    "gain 1.1 all steps": dict(gain=(1.0, 1.1, 1.1, 1.1)),
    "gain 1.2 all steps": dict(gain=(1.0, 1.2, 1.2, 1.2)),
    "gain 1.4 all steps": dict(gain=(1.0, 1.4, 1.4, 1.4)),
    "gain 1.2 step 1 only": dict(gain=(1.0, 1.2, 1.0, 1.0)),
    "gain 1.2 step 2 only": dict(gain=(1.0, 1.0, 1.2, 1.0)),
    "gain 1.2 step 3 only": dict(gain=(1.0, 1.0, 1.0, 1.2)),
    "gain 0.9 all steps": dict(gain=(1.0, 0.9, 0.9, 0.9)),
    "boost 0.2 step 1": dict(boost=(0.0, 0.2, 0.0, 0.0)),
    "boost 0.5 step 1": dict(boost=(0.0, 0.5, 0.0, 0.0)),
    "boost 0.5 step 2": dict(boost=(0.0, 0.0, 0.5, 0.0)),
    "boost -0.5 step 2": dict(boost=(0.0, 0.0, -0.5, 0.0)),
    "boost 0.5 steps 1-2": dict(boost=(0.0, 0.5, 0.5, 0.0)),
}
