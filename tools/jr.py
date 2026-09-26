#!/usr/bin/env python3
"""Small journal editor for the executor.

jr.py section "<heading>" <file-with-new-body>   replace a "## <heading>" section of STATE.md
jr.py log "<line>"                                append "- HH:MM <line>" under today's date in LOG.md
jr.py updated                                     refresh the "Updated:" line of STATE.md
jr.py commit "<message>"                          fence check, commit and push the journal
"""
import datetime, pathlib, re, subprocess, sys

J = pathlib.Path(__file__).resolve().parents[1]
OWNER = pathlib.Path("/Users/aschuler/Documents/research/projects/pymars/.git/pymars-executor/lock/owner")
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def now():
    return datetime.datetime.now()


def section(heading, body_file):
    p = J / "STATE.md"
    s = p.read_text()
    body = pathlib.Path(body_file).read_text().strip("\n")
    pat = re.compile(r"(^## " + re.escape(heading) + r"\n\n)(.*?)(?=^## |\Z)", re.S | re.M)
    if not pat.search(s):
        s = s.rstrip("\n") + f"\n\n## {heading}\n\n{body}\n"
    else:
        s = pat.sub(lambda m: m.group(1) + body + "\n\n", s, count=1)
    p.write_text(s.rstrip("\n") + "\n")


def updated(session):
    p = J / "STATE.md"
    s = p.read_text()
    stamp = now().strftime("%Y-%m-%d %H:%M PDT")
    s = re.sub(r"^Updated: .*$", f"Updated: {stamp}, by executor session `{session}`.", s, count=1, flags=re.M)
    p.write_text(s)


def log(line):
    p = J / "LOG.md"
    s = p.read_text()
    day = now().strftime("%Y-%m-%d")
    entry = f"- {now().strftime('%H:%M')} {line}\n"
    marker = "\n## Core-hour ledger"
    head, sep, tail = s.partition(marker)
    if f"## {day}\n" not in head:
        head = head.rstrip("\n") + f"\n\n## {day}\n\n"
    head = head.rstrip("\n") + "\n" + entry
    p.write_text(head + sep + tail)


def commit(msg):
    owner = OWNER.read_text().strip()
    if owner != SESSION:
        sys.exit(f"FENCE: lock owner is {owner}, not {SESSION}; stop")
    subprocess.run(["git", "-C", str(J), "add", "-A"], check=True)
    r = subprocess.run(["git", "-C", str(J), "commit", "-q", "-m", msg, "-m", TRAILER])
    if r.returncode == 0:
        subprocess.run(["git", "-C", str(J), "push", "-q", "origin", "executor"], check=True)
        print("journal pushed")


SESSION = "local_a59469db-c3f1-480c-81c4-a4c7beae39f4"

if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "section":
        section(a[0], a[1])
    elif cmd == "log":
        log(a[0])
    elif cmd == "updated":
        updated(SESSION)
    elif cmd == "commit":
        updated(SESSION)
        commit(a[0])
    else:
        sys.exit(__doc__)
