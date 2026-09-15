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


def check_counts(readme: str, docs: str, manifest: dict) -> None:
    print("\n-- counts --")
    n = passing_tests()
    check("test count in README", f"{n} tests" in readme, str(n))
    chunks = str(manifest["chunk_count"])
    check("chunk count", chunks in readme.replace(",", ""), chunks)
    standards = len(manifest["standards"])
    check("standards count", f"{standards} standards" in docs, str(standards))
    check("corpus_version", manifest["corpus_version"] in docs, manifest["corpus_version"])
    check("index_snapshot", manifest["index_snapshot"] in docs, manifest["index_snapshot"])


def check_references(docs: str) -> None:
    """Every file a document links to or names by path must exist."""
    print("\n-- file references --")
    targets = {
        re.sub(r"^\./", "", target)
        for _, target in re.findall(r"\[`([^`]+)`\]\(([^)]+)\)", docs)
        if not target.startswith("http")
    }
    named = set(
        re.findall(
            r"`((?:src|eval|scripts|tests|deploy|docs|corpus)/[\w./-]+?"
            r"\.(?:py|yaml|yml|json|md))`",
            docs,
        )
    )
    for path in sorted(targets | named):
        check(f"exists: {path}", (ROOT / path).exists())


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
    for path in sorted(glob.glob(str(ROOT / "eval/results/*.json")), key=os.path.getmtime):
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
    readme = pathlib.Path("README.md").read_text()
    part2 = pathlib.Path("docs/PART2-technical-decisions.md").read_text()
    manifest = json.loads(pathlib.Path("corpus/manifest.json").read_text())

    check_counts(readme, readme + part2, manifest)
    check_references(readme + part2)
    check_settings(readme + part2)
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
