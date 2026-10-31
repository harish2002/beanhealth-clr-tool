"""
Validation report from a gaze-exam database:

    python -m exam.validation --db exam.sqlite [--json validation.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exam.db import connect
from exam.validation.from_db import summary, validate


def main(argv: list = None) -> int:      # type: ignore[assignment]
    ap = argparse.ArgumentParser(prog="python -m exam.validation", description=__doc__)
    ap.add_argument("--db", required=True, help="SQLite database written by exam.db.store_exam")
    ap.add_argument("--json", help="also write the full report as JSON")
    args = ap.parse_args(argv)
    if not Path(args.db).exists():
        print(f"no database at {args.db}", file=sys.stderr)
        return 2
    report = validate(connect(args.db))
    print(summary(report))
    if args.json:
        Path(args.json).write_text(report.model_dump_json(indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
