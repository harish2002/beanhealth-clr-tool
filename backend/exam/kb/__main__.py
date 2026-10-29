"""
Knowledge base command line.

    python -m exam.kb check    validate the KB and print its version
    python -m exam.kb review   list every entry not yet clinician-approved
    python -m exam.kb stamp    re-stamp manifest.content_sha256 after an edit

Run from backend/.
"""

from __future__ import annotations

import re
import sys
from collections import Counter

from exam.kb.loader import DEFAULT_KB_DIR, MANIFEST, KBValidationError, content_sha256, load_kb, read_raw


def _check() -> int:
    try:
        kb = load_kb(DEFAULT_KB_DIR)
    except KBValidationError as e:
        print(e, file=sys.stderr)
        return 1
    counts = Counter(e.review_status for _, e in kb.entries())
    print(f"KB {kb.version}  sha256 {kb.sha256[:12]}  —  " +
          ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    return 0


def _review() -> int:
    kb = load_kb(DEFAULT_KB_DIR)
    for file_name, e in kb.entries():
        if e.review_status != "approved":
            unset = "  [value not set]" if getattr(e, "value", 0) is None else ""
            print(f"{e.review_status:24} {file_name:18} {e.id}{unset}")
    return 0


def _stamp() -> int:
    manifest_path = DEFAULT_KB_DIR / MANIFEST
    _, raw = read_raw(DEFAULT_KB_DIR)
    sha = content_sha256(raw)
    text = manifest_path.read_text(encoding="utf-8")
    new_text, n = re.subn(r"(?m)^content_sha256:.*$", f"content_sha256: {sha}", text)
    if n != 1:
        print(f"{MANIFEST}: expected one content_sha256 line", file=sys.stderr)
        return 1
    if new_text == text:
        print(f"Already stamped: {sha[:12]}")
        return 0
    manifest_path.write_text(new_text, encoding="utf-8")
    print(f"Stamped {sha[:12]}. Remember to bump kb_version in {MANIFEST}.")
    return _check()


if __name__ == "__main__":
    commands = {"check": _check, "review": _review, "stamp": _stamp}
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd not in commands:
        print(__doc__)
        sys.exit(2)
    sys.exit(commands[cmd]())
