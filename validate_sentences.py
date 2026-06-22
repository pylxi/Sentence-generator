"""
validate_sentences.py — CEFR sentence bank quality audit

Reads output/generated_sentences.csv, classifies each sentence's CEFR level
using Groq (llama-3.3-70b-versatile), compares against the target level, and
writes an HTML audit report to output/validation_report.html.

Mismatch severity:
  ✓  on target       — predicted level matches saved level
  ⚠  off by 1 band  — borderline, review recommended
  ✗  off by 2+ bands — likely wrong level, should replace

Usage:
    cd ~/Documents/Sentence-generator
    source venv/bin/activate
    python validate_sentences.py

Optional flags:
    --batch N    sentences per Groq call (default 10)
    --csv PATH   path to CSV (default output/generated_sentences.csv)
    --out PATH   output report path (default output/validation_report.html)
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_INDEX = {l: i for i, l in enumerate(LEVELS)}

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "output" / "generated_sentences.csv"
DEFAULT_OUT = BASE_DIR / "output" / "validation_report.html"


def level_distance(a: str, b: str) -> int:
    """How many CEFR bands apart are two levels?"""
    if a not in LEVEL_INDEX or b not in LEVEL_INDEX:
        return 99
    return abs(LEVEL_INDEX[a] - LEVEL_INDEX[b])


def severity(target: str, predicted: str) -> str:
    d = level_distance(target, predicted)
    if d == 0:
        return "ok"
    if d == 1:
        return "warn"
    return "fail"


def classify_batch(client: Groq, sentences: list[dict]) -> list[str]:
    """
    Send a batch of sentences to Groq and return predicted CEFR levels.
    Returns a list of level strings (same length as input).
    Falls back to "?" on parse error.
    """
    numbered = "\n".join(
        f"{i+1}. [{row['level']}] {row['sentence']}"
        for i, row in enumerate(sentences)
    )
    prompt = (
        "You are an expert EFL assessor. For each sentence below, classify the "
        "CEFR difficulty level of the sentence ITSELF (grammar, syntax, discourse "
        "complexity — not just vocabulary). Ignore the level shown in brackets; "
        "that is the claimed level, not the actual one.\n\n"
        "Reply with ONLY a numbered list, one level per line, e.g.:\n"
        "1. B1\n2. A2\n3. C1\n\n"
        f"Sentences:\n{numbered}"
    )

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=len(sentences) * 8 + 20,
        )
        raw = resp.choices[0].message.content.strip()
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.search(r"\b([ABC][12])\b", line, re.IGNORECASE)
            results.append(m.group(1).upper() if m else "?")
        # Pad / truncate to match batch size
        while len(results) < len(sentences):
            results.append("?")
        return results[: len(sentences)]
    except Exception as e:
        print(f"  Groq error: {e}", file=sys.stderr)
        return ["?"] * len(sentences)


def build_report(rows: list[dict]) -> str:
    """Generate an HTML audit report."""
    total = len(rows)
    ok = sum(1 for r in rows if r["sev"] == "ok")
    warn = sum(1 for r in rows if r["sev"] == "warn")
    fail = sum(1 for r in rows if r["sev"] == "fail")
    unknown = sum(1 for r in rows if r["sev"] == "unknown")

    # Group by level
    by_level: dict[str, list] = {l: [] for l in LEVELS}
    for r in rows:
        by_level.get(r["level"], []).append(r)

    sev_icon = {"ok": "✓", "warn": "⚠", "fail": "✗", "unknown": "?"}
    sev_color = {
        "ok": "#1d7a47",
        "warn": "#b07500",
        "fail": "#c03030",
        "unknown": "#888",
    }
    sev_bg = {
        "ok": "#edf7f1",
        "warn": "#fdf6e3",
        "fail": "#fdf0f0",
        "unknown": "#f5f5f5",
    }

    level_blocks = ""
    for level in LEVELS:
        level_rows = by_level[level]
        if not level_rows:
            continue
        rows_html = ""
        for r in level_rows:
            icon = sev_icon.get(r["sev"], "?")
            color = sev_color.get(r["sev"], "#888")
            bg = sev_bg.get(r["sev"], "#f5f5f5")
            predicted_display = r["predicted"] if r["predicted"] != "?" else "unknown"
            rows_html += f"""
            <tr style="background:{bg}">
              <td style="padding:9px 12px;width:32px;text-align:center;font-size:16px;color:{color};font-weight:700">{icon}</td>
              <td style="padding:9px 12px;font-size:13px;font-weight:700;white-space:nowrap;color:#555">{r['word']}</td>
              <td style="padding:9px 12px;font-size:13px;color:#3a3a44">{r['sentence']}</td>
              <td style="padding:9px 12px;font-size:12px;font-weight:700;white-space:nowrap;color:#888">{r['prompt_label']}</td>
              <td style="padding:9px 12px;text-align:center;font-size:13px;font-weight:800;color:{color}">{predicted_display}</td>
            </tr>"""

        level_ok = sum(1 for r in level_rows if r["sev"] == "ok")
        level_blocks += f"""
        <div style="margin-bottom:28px">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
            <span style="display:inline-flex;align-items:center;justify-content:center;width:44px;height:32px;border-radius:9px;background:rgba(239,107,74,0.12);color:#d9542f;font-size:15px;font-weight:900">{level}</span>
            <span style="font-size:15px;font-weight:700;color:#15151c">{len(level_rows)} sentences</span>
            <span style="font-size:12px;color:#888">{level_ok} on target</span>
          </div>
          <table style="width:100%;border-collapse:collapse;border:1px solid #e8e6e2;border-radius:12px;overflow:hidden">
            <thead>
              <tr style="background:#f8f7f5">
                <th style="padding:8px 12px;font-size:11px;font-weight:800;color:#888;letter-spacing:0.06em;text-align:center"></th>
                <th style="padding:8px 12px;font-size:11px;font-weight:800;color:#888;letter-spacing:0.06em;text-align:left">WORD</th>
                <th style="padding:8px 12px;font-size:11px;font-weight:800;color:#888;letter-spacing:0.06em;text-align:left">SENTENCE</th>
                <th style="padding:8px 12px;font-size:11px;font-weight:800;color:#888;letter-spacing:0.06em;text-align:left">PROMPT TYPE</th>
                <th style="padding:8px 12px;font-size:11px;font-weight:800;color:#888;letter-spacing:0.06em;text-align:center">PREDICTED</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentence Bank Audit — {datetime.now().strftime('%Y-%m-%d')}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#fbfbfa;color:#15151c;margin:0;padding:32px 24px}}
  .container{{max-width:960px;margin:0 auto}}
  .header{{margin-bottom:28px}}
  .summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:32px}}
  .tile{{background:#fff;border:1px solid #e8e6e2;border-radius:12px;padding:16px 18px}}
  .tile .val{{font-size:28px;font-weight:800;letter-spacing:-0.03em}}
  .tile .lbl{{font-size:12px;font-weight:600;color:#888;margin-top:3px}}
  .legend{{display:flex;gap:18px;margin-bottom:24px;flex-wrap:wrap}}
  .legend-item{{display:flex;align-items:center;gap:7px;font-size:13px;font-weight:600;color:#555}}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div style="font-size:11px;font-weight:800;letter-spacing:0.10em;color:#ef6b4a;text-transform:uppercase;margin-bottom:6px">SentenceStudio · Sentence Bank Audit</div>
    <h1 style="margin:0;font-size:26px;font-weight:900;letter-spacing:-0.03em">CEFR Level Validation Report</h1>
    <p style="margin:6px 0 0;font-size:13px;color:#888">Generated {datetime.now().strftime('%B %d, %Y at %H:%M')} · Model: llama-3.3-70b-versatile · {total} sentences</p>
  </div>

  <div class="summary">
    <div class="tile"><div class="val">{total}</div><div class="lbl">Total sentences</div></div>
    <div class="tile"><div class="val" style="color:#1d7a47">{ok}</div><div class="lbl">✓ On target</div></div>
    <div class="tile"><div class="val" style="color:#b07500">{warn}</div><div class="lbl">⚠ Off by 1 band</div></div>
    <div class="tile"><div class="val" style="color:#c03030">{fail}</div><div class="lbl">✗ Off by 2+ bands</div></div>
    {"" if not unknown else f'<div class="tile"><div class="val" style="color:#888">{unknown}</div><div class="lbl">? Unclassified</div></div>'}
  </div>

  <div class="legend">
    <div class="legend-item"><span style="color:#1d7a47;font-size:16px">✓</span> Predicted level matches target — keep</div>
    <div class="legend-item"><span style="color:#b07500;font-size:16px">⚠</span> Off by 1 band — review recommended</div>
    <div class="legend-item"><span style="color:#c03030;font-size:16px">✗</span> Off by 2+ bands — replace before study</div>
  </div>

  {level_blocks}
</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Validate CEFR sentence bank levels")
    parser.add_argument("--batch", type=int, default=10, help="Sentences per Groq call")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Input CSV path")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output HTML path")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key)

    # Load CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = [r for r in reader if r.get("status", "").lower() == "accepted"]

    if not all_rows:
        print("No accepted sentences found in CSV.")
        sys.exit(0)

    print(f"Validating {len(all_rows)} accepted sentences in batches of {args.batch}…\n")

    results = []
    for i in range(0, len(all_rows), args.batch):
        batch = all_rows[i : i + args.batch]
        end = min(i + args.batch, len(all_rows))
        print(f"  Batch {i+1}–{end} / {len(all_rows)}", end="", flush=True)
        predictions = classify_batch(client, batch)
        for row, pred in zip(batch, predictions):
            sev = severity(row["level"], pred) if pred != "?" else "unknown"
            results.append({**row, "predicted": pred, "sev": sev})
            icon = {"ok": "✓", "warn": "⚠", "fail": "✗", "unknown": "?"}.get(sev, "?")
            print(f" {icon}", end="", flush=True)
        print()
        if i + args.batch < len(all_rows):
            time.sleep(0.5)  # be kind to rate limits

    # Summary
    ok = sum(1 for r in results if r["sev"] == "ok")
    warn = sum(1 for r in results if r["sev"] == "warn")
    fail = sum(1 for r in results if r["sev"] == "fail")
    print(f"\nResults: ✓ {ok} on target  ⚠ {warn} off by 1  ✗ {fail} off by 2+")

    # Write report
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(build_report(results), encoding="utf-8")
    print(f"\nReport saved to: {out_path}")
    print("Open it in your browser to review flagged sentences.")


if __name__ == "__main__":
    main()
