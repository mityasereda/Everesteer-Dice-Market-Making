"""Dice Market Making API client. Python 3.9+, standard library only, Mac and Windows.

Create a token on the Play page (API token). Then:

    export DICE_TOKEN=dm_...          (Windows: set DICE_TOKEN=dm_...)
    python3 dmapi.py me
    python3 dmapi.py upload agent.py --name "tight spread v2"   (or a .zip with agent.py and weights.npz)
    python3 dmapi.py activate ID
    python3 dmapi.py leaderboard
    python3 dmapi.py matches [--mine]
    python3 dmapi.py replay ID [-o replay.json]
    python3 dmapi.py dataset -o replays.jsonl.gz [--since 2026-10-23T12:00:00Z]

Options: --url (or DICE_URL, default https://dice.everesteer.ai), --token (or DICE_TOKEN), --json for the raw answer.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = "https://dice.everesteer.ai"


def call(args, method, path, body=None, headers=None, raw=False):
    """One API request. Returns parsed JSON, or bytes if raw. Exits with the server's message on an error."""
    h = {"Authorization": "Bearer " + args.token, "User-Agent": "dmapi/1"}
    h.update(headers or {})
    req = urllib.request.Request(args.url.rstrip("/") + "/api/v1/" + path, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error")
        except ValueError:
            msg = None
        sys.exit("Error {}: {}".format(e.code, msg or e.reason))
    except urllib.error.URLError as e:
        sys.exit("The server does not answer: {}".format(e.reason))
    return data if raw else json.loads(data)


def show(args, obj, text):
    print(json.dumps(obj, indent=1) if args.json else text)


def seats(m):
    return ", ".join("{} {:+.2f}".format(s["label"], s["pnl_per_round"]) for s in m["seats"])


def main(argv=None):
    p = argparse.ArgumentParser(description="Dice Market Making API client.")
    p.add_argument("--url", default=os.environ.get("DICE_URL", DEFAULT_URL))
    p.add_argument("--token", default=os.environ.get("DICE_TOKEN", ""))
    p.add_argument("--json", action="store_true", help="print the raw JSON answer")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("me", help="your handle, versions and uploads left today")
    u = sub.add_parser("upload", help="upload agent.py or a .zip with agent.py and weights.npz")
    u.add_argument("file")
    u.add_argument("--name", default="")
    sub.add_parser("activate", help="make a version active").add_argument("id")
    sub.add_parser("delete", help="delete a version that is not active").add_argument("id")
    sub.add_parser("leaderboard", help="the leaderboard")
    sub.add_parser("matches", help="recent matches").add_argument("--mine", action="store_true")
    r = sub.add_parser("replay", help="one replay as JSON")
    r.add_argument("id")
    r.add_argument("-o", "--out")
    d = sub.add_parser("dataset", help="download every match (gzip JSON lines)")
    d.add_argument("-o", "--out", default="replays.jsonl.gz")
    d.add_argument("--since", help="only matches after this ISO time, for example 2026-10-23T12:00:00Z")
    args = p.parse_args(argv)
    if not args.token:
        sys.exit("Set DICE_TOKEN or give --token. Create a token on the Play page.")

    if args.cmd == "me":
        m = call(args, "GET", "me")
        lines = ["{}  uploads left today: {}".format(m["handle"], m["uploads_left_today"])]
        for v in m["versions"]:
            lines.append("{} {:<6} {:<30} {:<7} Elo {:>5}  {} matches".format(
                "*" if v["id"] == m["active"] else " ", v["id"], v["name"], "passed" if v["passed"] else "failed",
                "-" if v["elo"] is None else v["elo"], v["matches"]))
        show(args, m, "\n".join(lines))
    elif args.cmd == "upload":
        with open(args.file, "rb") as f:
            body = f.read()
        zip_ = args.file.lower().endswith(".zip")
        print("Playing the boss: 3 matches. This takes 5 to 20 seconds.", file=sys.stderr)
        g = call(args, "POST", "agents", body, {"Content-Type": "application/zip" if zip_ else "text/plain",
                                                "X-Agent-Name": urllib.parse.quote(args.name)})["gate"]
        text = "The boss: pass" if g["passed"] else "The boss: not beaten"
        if g.get("scan_hits"):
            text = "The code scan refused the file. Remove: " + ", ".join(g["scan_hits"])
        elif g.get("reason"):
            text += ". " + g["reason"]
        for i, m in enumerate(g.get("matches", []), 1):
            text += "\n  match {}: you {:+.2f}, boss {:+.2f}{}".format(i, m["you"], m["boss"],
                                                                    "  forfeit: " + m["forfeit"] if m.get("forfeit") else "")
        show(args, {"gate": g}, text)
    elif args.cmd == "activate":
        m = call(args, "POST", "agents/active", json.dumps({"id": args.id}).encode(), {"Content-Type": "application/json"})
        show(args, m, "Active version: " + args.id)
    elif args.cmd == "delete":
        m = call(args, "POST", "agents/delete", json.dumps({"id": args.id}).encode(), {"Content-Type": "application/json"})
        show(args, m, "Deleted version: " + args.id)
    elif args.cmd == "leaderboard":
        rows = call(args, "GET", "leaderboard")["rows"]
        show(args, rows, "\n".join("{:>3} {:<24} {:<20} Elo {:>5}{} {:>4} matches  {}".format(
            r["rank"], r["label"], r["agent"], r["elo"], "?" if r["provisional"] else " ", r["matches"],
            "-" if r["pnl_per_round"] is None else "{:+.2f} per round".format(r["pnl_per_round"])) for r in rows))
    elif args.cmd == "matches":
        ms = call(args, "GET", "matches" + ("?mine=1" if args.mine else ""))["matches"]
        show(args, ms, "\n".join("{}  {}  {}".format(m["id"], m["at"][:19], seats(m)) for m in ms) or "No matches yet.")
    elif args.cmd == "replay":
        data = call(args, "GET", "replays/" + urllib.parse.quote(args.id), raw=True)
        if args.out:
            with open(args.out, "wb") as f:
                f.write(data)
            print("Saved {} ({} bytes).".format(args.out, len(data)))
        else:
            print(data.decode())
    elif args.cmd == "dataset":
        q = "?since=" + urllib.parse.quote(args.since) if args.since else ""
        data = call(args, "GET", "dataset" + q, raw=True)
        with open(args.out, "wb") as f:
            f.write(data)
        print("Saved {} ({} bytes).".format(args.out, len(data)))


if __name__ == "__main__":
    main()
