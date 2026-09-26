#!/usr/bin/env python3
"""Create the board issues from issues.md in alejandroschuler/mars.

A block in issues.md starts with a line "=== <title>" and a line
"labels: <comma-separated labels>"; the rest is the body. Titles that
already exist as issues (open or closed) are skipped, so a second run is
safe. With --dry-run nothing is created. The numbers of all board issues
are written to numbers.json next to this file.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

REPO = "alejandroschuler/mars"
HERE = pathlib.Path(__file__).resolve().parent


def gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def main() -> None:
    dry = "--dry-run" in sys.argv
    blocks = (HERE / "issues.md").read_text().split("\n=== ")[1:]
    existing = json.loads(
        gh("issue", "list", "--repo", REPO, "--state", "all", "--limit", "500",
           "--json", "number,title")
    )
    numbers = {i["title"]: i["number"] for i in existing}
    for block in blocks:
        title, rest = block.split("\n", 1)
        labels_line, body = rest.split("\n", 1)
        if not labels_line.startswith("labels: "):
            sys.exit(f"bad block, no labels line: {title}")
        title = title.strip()
        labels = labels_line[len("labels: "):].strip()
        if title in numbers:
            print(f"exists #{numbers[title]}: {title}")
            continue
        if dry:
            print(f"would create [{labels}]: {title}")
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(body.strip() + "\n")
        url = gh("issue", "create", "--repo", REPO, "--title", title,
                 "--label", labels, "--body-file", f.name).strip()
        numbers[title] = int(url.rsplit("/", 1)[1])
        print(f"created {url}: {title}")
    if not dry:
        board = {t: n for t, n in numbers.items()
                 if t.startswith(("T", "later:", "needs-user:"))}
        (HERE / "numbers.json").write_text(json.dumps(board, indent=1) + "\n")


if __name__ == "__main__":
    main()
