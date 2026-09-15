#!/usr/bin/env python
"""Check every mechanically checkable claim in the docs against reality.

Documentation drifts silently. A number is right when written and wrong three
commits later, and nobody re-reads a README looking for arithmetic. An external
review of this repo found exactly that: the architecture judgements held up, and
the corpus counts, the test total and a claim about prompt caching did not.

So the checkable parts get checked — counts against the manifest and the test
suite, every file a document links to or names, every environment variable
against the Settings model, and every eval figure against the saved results.
Run it before publishing anything.

    python scripts/verify_docs.py
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

_failures: list[str] = []

# Standards present in the source but absent from the shipped index. The parser
# fix that recovers them is in place (tests/test_ingest.py); the index has not
# been rebuilt. Documented in README "Known limitations" and PART2 §6.
KNOWN_UNINDEXED = {42, 43, 44, 46, 47, 48}


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = f"{GREEN}OK {RESET}" if ok else f"{RED}BAD{RESET}"
    print(f"  {mark} {label:52} {detail}")
    if not ok:
        _failures.append(label)


def passing_tests() -> int:
    out = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout
    found = re.search(r"(\d+) passed", out)
    return int(found.group(1)) if found else 0


def check_counts(docs: str, manifest: dict) -> None:
    print("\n-- counts --")
    n = passing_tests()
    check("test count", f"{n} tests" in docs, str(n))
    chunks = str(manifest["chunk_count"])
    check("chunk count", chunks in docs.replace(",", ""), chunks)
    indexed = len(manifest["standards"])
    # Pins the phrasing that carries both numbers, so "48 standards" alone -- the
    # circular form that hid the gap -- no longer satisfies the check.
    check("standards coverage stated with its real denominator",
          f"{indexed} of the edition's 54" in docs, f"{indexed} of 54")
    check("corpus_version", manifest["corpus_version"] in docs, manifest["corpus_version"])
    check("index_snapshot", manifest["index_snapshot"] in docs, manifest["index_snapshot"])


def check_corpus_coverage(manifest: dict) -> None:
    """The index must cover every standard the source actually contains.

    This is the check that was missing when six standards (42, 43, 44, 46, 47,
    48) sat outside the index while the docs called the corpus complete. The
    denominator has to come from the source numbering rather than from the
    parser's own output, or the check just confirms the parser found what the
    parser found.
    """
    print("\n-- corpus coverage --")
    indexed = {int(n) for n in manifest["standards"]}
    expected = set(range(1, 55))  # 2017 English edition, none reserved
    missing = expected - indexed
    # Declared, not assumed. parse.py now recognises these headers, so a rebuilt
    # index closes the gap; until it is rebuilt they stay out, and anything
    # missing *beyond* this set is new breakage and fails.
    unexpected = sorted(missing - KNOWN_UNINDEXED)
    check("no standard missing beyond the disclosed gap", not unexpected,
          f"unexpected {unexpected}" if unexpected else "ok")
    check("indexed coverage", True,
          f"{len(indexed)}/{len(expected)} — absent: {sorted(missing) or 'none'}")
    total = sum(v["chunks"] for v in manifest["standards"].values())
    check("chunk counts sum to the manifest total",
          total == manifest["chunk_count"], str(total))


def check_references(pages: dict[pathlib.Path, str]) -> None:
    """Every file a document links to or names by path must exist.

    Links resolve against the page that contains them, so `../deploy/x` from
    docs/ and `docs/x` from the README both land in the right place.
    """
    print("\n-- file references --")
    found: dict[str, pathlib.Path] = {}
    for page, text in pages.items():
        base = (ROOT / page).parent
        targets = {
            target
            for target in re.findall(r"\[[`\w][^\]]*\]\(([^)]+)\)", text)
            if not target.startswith(("http", "#"))
        }
        named = set(
            re.findall(
                r"`((?:src|eval|scripts|tests|deploy|docs|corpus)/[\w./-]+?"
                r"\.(?:py|yaml|yml|json|md|sh|gz))`",
                text,
            )
        )
        # Markdown links are relative to the page; a path named in backticks is
        # always written from the repo root.
        for target, anchor in ((t, base) for t in targets):
            resolved = (anchor / target.split("#")[0]).resolve()
            try:
                label = str(resolved.relative_to(ROOT))
            except ValueError:
                label = str(resolved)
            found.setdefault(label, resolved)
        for target in named:
            found.setdefault(target, ROOT / target)
    for label, resolved in sorted(found.items()):
        check(f"exists: {label}", resolved.exists())


def check_settings(docs: str) -> None:
    """Every SCA_* variable a document mentions must be a real setting."""
    print("\n-- environment variables --")
    sys.path.insert(0, str(ROOT / "src"))
    from sharia_agent.config import Settings

    fields = set(Settings.model_fields)
    aliases = {f.alias for f in Settings.model_fields.values() if f.alias}
    for var in sorted(set(re.findall(r"\b(SCA_[A-Z_]+|OPENROUTER_API_KEY)\b", docs))):
        real = var in aliases or (var.startswith("SCA_") and var[4:].lower() in fields)
        check(var, real)


def check_eval_figures(part2: str) -> None:
    """Numbers quoted from eval runs must match the saved results."""
    print("\n-- eval figures --")
    runs = []
    # Sorted by the timestamp in the filename, not mtime: a fresh clone stamps
    # every file at checkout time, so mtime ordering is arbitrary there and
    # "the last two runs" would pick arbitrary ones.
    for path in sorted(glob.glob(str(ROOT / "eval/results/*.json"))):
        data = json.loads(pathlib.Path(path).read_text())
        if not data["config"].get("subset"):
            runs.append(data["summary"])
    if len(runs) < 2:
        print("  (fewer than two full runs on disk; nothing to cross-check)")
        return
    a, b = runs[-2], runs[-1]
    for label, value in (
        ("run 1 accuracy", a["accuracy"]),
        ("run 2 accuracy", b["accuracy"]),
    ):
        check(f"{label} quoted in PART2", f"{value}" in part2, str(value))
    check("both runs zero false-COMPLIANT",
          a["false_compliant"] == 0 and b["false_compliant"] == 0)
    for value in (a["recall"]["5"], b["recall"]["5"], a["mrr"], b["mrr"]):
        check(f"figure {value:.3f} quoted in PART2", f"{value:.3f}" in part2)


def main() -> int:
    os.chdir(ROOT)
    # Every prose page, so moving a claim between documents keeps it checked
    # rather than quietly removing it from scope.
    pages = {
        pathlib.Path(p): pathlib.Path(p).read_text()
        for p in ["README.md", *sorted(glob.glob("docs/*.md"))]
    }
    part2 = pages[pathlib.Path("docs/PART2-technical-decisions.md")]
    manifest = json.loads(pathlib.Path("corpus/manifest.json").read_text())
    everything = "\n".join(pages.values())

    check_counts(everything, manifest)
    check_corpus_coverage(manifest)
    check_references(pages)
    check_settings(everything)
    check_eval_figures(part2)

    if _failures:
        print(f"\n  {len(_failures)} claim(s) do not match reality:")
        for failure in _failures:
            print(f"    - {failure}")
        return 1
    print("\n  every checkable claim matches reality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
