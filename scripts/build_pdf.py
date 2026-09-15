#!/usr/bin/env python
"""Render the architecture document to the PDF the submission asks for.

The brief caps it at 2-4 pages, so the page count is a requirement rather than
an aesthetic, and this prints it on every run.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from markdown_it import MarkdownIt
from weasyprint import CSS, HTML

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs/PART2-technical-decisions.md"
OUTPUT = ROOT / "docs/PART2-technical-decisions.pdf"
PAGE_LIMIT = 4

STYLE = """
@page {
  size: A4;
  margin: 15mm 18mm 14mm 18mm;
  @bottom-right {
    content: counter(page) " / " counter(pages);
    font: 7.5pt "DejaVu Sans"; color: #8a8a8a;
  }
  @bottom-left {
    content: "Shari'ah Compliance Agent — technical decisions";
    font: 7.5pt "DejaVu Sans"; color: #8a8a8a;
  }
}

html { font-size: 9.3pt; }
body {
  font-family: "DejaVu Serif", Georgia, serif;
  line-height: 1.32;
  color: #16181d;
  text-align: justify;
  hyphens: auto;
}

h1 {
  font: 700 17pt "DejaVu Sans", sans-serif;
  margin: 0 0 1mm; letter-spacing: -0.2pt; color: #0f1115;
}
h1 + p {                       /* the subtitle line under the title */
  margin: 0 0 5mm;
  font: 9.5pt "DejaVu Sans", sans-serif;
  color: #5d636e; text-align: left;
}
h2 {
  font: 700 11pt "DejaVu Sans", sans-serif;
  color: #0f1115;
  margin: 4.4mm 0 1.8mm;
  padding-top: 1.4mm;
  border-top: 0.6pt solid #d5d8dd;
  break-after: avoid;
}
h2:first-of-type { margin-top: 3mm; }

p { margin: 0 0 2mm; orphans: 2; widows: 2; }

strong { color: #000; }
em { color: #2b2f36; }

code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 0.86em;
  background: #f2f3f5;
  padding: 0.3pt 1.2pt;
  border-radius: 1.5pt;
}

blockquote {
  margin: 2.5mm 0 3mm;
  padding: 1.5mm 0 1.5mm 4mm;
  border-left: 1.6pt solid #2f6f4f;
  font-style: italic;
  color: #1b1d22;
  break-inside: avoid;
}
blockquote p { margin: 0; text-align: left; }

table {
  border-collapse: collapse;
  margin: 2.5mm 0 3.5mm;
  font-size: 0.93em;
  break-inside: avoid;
}
th, td {
  padding: 1.1mm 3.5mm 1.1mm 0;
  text-align: left;
  border-bottom: 0.5pt solid #e2e4e8;
}
thead th {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 0.92em;
  border-bottom: 0.8pt solid #9aa0aa;
  color: #3a3f48;
}
tbody tr:last-child td { border-bottom: none; }

hr { border: none; border-top: 0.5pt solid #d5d8dd; margin: 4mm 0 3mm; }
"""


def render(source: pathlib.Path, output: pathlib.Path) -> int:
    md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    body = md.render(source.read_text())
    doc = HTML(string=f"<article>{body}</article>").render(
        stylesheets=[CSS(string=STYLE)]
    )
    doc.write_pdf(output)
    return len(doc.pages)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=pathlib.Path, default=SOURCE)
    ap.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    ap.add_argument("--limit", type=int, default=PAGE_LIMIT)
    args = ap.parse_args()

    pages = render(args.source, args.output)
    size_kb = args.output.stat().st_size / 1024
    rel = args.output.relative_to(ROOT)
    print(f"  {rel}  {pages} pages, {size_kb:.0f} KB")

    if pages > args.limit:
        over = pages - args.limit
        print(f"  over the {args.limit}-page limit by {over}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
