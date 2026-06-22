"""
validate_sentences_cefrsp.py — CEFR-SP sentence bank quality audit
====================================================================
Uses the academically validated CEFR-SP model (Arase et al., EMNLP 2022,
84.5% macro-F1) to classify each sentence's difficulty level and flag
mismatches against the target level in generated_sentences.csv.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP (one-time, ~5 min + download time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Clone the CEFR-SP repository anywhere on your machine:
      git clone https://github.com/yukiar/CEFR-SP.git ~/CEFR-SP

2. Install dependencies (in your SentenceStudio venv):
      pip install torch transformers "pytorch-lightning==1.7.7" scikit-learn scipy seaborn

3. Download the pretrained model (1.2 GB) from Zenodo:
      https://zenodo.org/records/7234096
   Save the file as:  ~/CEFR-SP/level_estimator.ckpt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   cd ~/Documents/Sentence-generator
   source venv/bin/activate
   python validate_sentences_cefrsp.py \\
       --repo ~/CEFR-SP \\
       --model ~/CEFR-SP/level_estimator.ckpt

Optional:
   --csv   PATH    input CSV  (default: output/generated_sentences.csv)
   --out   PATH    output HTML (default: output/validation_report_cefrsp.html)
   --batch N       sentences per forward pass (default: 32)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Arase, Y., Uchida, S., & Kajiwara, T. (2022).
CEFR-Based Sentence Difficulty Annotation and Assessment.
Proceedings of EMNLP 2022, pp. 6206–6219.
https://aclanthology.org/2022.emnlp-main.416
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_INDEX = {l: i for i, l in enumerate(LEVELS)}

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "output" / "generated_sentences.csv"
DEFAULT_OUT = BASE_DIR / "output" / "validation_report_cefrsp.html"


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(repo_path: Path, checkpoint_path: Path):
    """
    Load the CEFR-SP contrastive model from checkpoint.

    We override with_loss_weight=False so the model does not try to read
    training corpus files (which are only needed during training, not inference).
    """
    sys.path.insert(0, str(repo_path / "src"))

    try:
        import torch
        from model import LevelEstimaterContrastive
    except ImportError as e:
        print(f"\nError: {e}")
        print("Make sure you have installed the dependencies:")
        print("  pip install torch transformers 'pytorch-lightning==1.7.7' scikit-learn scipy seaborn")
        sys.exit(1)

    if not checkpoint_path.exists():
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Download level_estimator.ckpt from https://zenodo.org/records/7234096")
        sys.exit(1)

    print(f"Loading CEFR-SP model from {checkpoint_path} …")
    print("(This may take 30–60 seconds on first load)")

    model = LevelEstimaterContrastive.load_from_checkpoint(
        str(checkpoint_path),
        # Required constructor args — not used during inference
        corpus_path="dummy",
        test_corpus_path="dummy",
        # Override to skip precompute_loss_weights() which reads corpus files
        with_loss_weight=False,
        map_location="cpu",
    )
    model.eval()
    print("Model loaded.\n")
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def classify_batch(model, sentences: list[str]) -> list[str]:
    """
    Run a batch of sentences through the CEFR-SP model.
    Returns a list of predicted CEFR level strings (A1–C2).

    The model tokenizer expects word-split input (is_split_into_words=True)
    which matches how it was trained on the SCoRE / Wiki-Auto corpus.
    """
    import torch

    word_lists = [s.split() for s in sentences]

    inputs = model.tokenizer(
        word_lists,
        return_tensors="pt",
        padding=True,
        is_split_into_words=True,
        return_offsets_mapping=True,
    )
    # offset_mapping is not used in forward() — remove to avoid BERT complaints
    inputs.pop("offset_mapping", None)

    with torch.no_grad():
        predictions = model(inputs)  # numpy array shape (N, 1), values 0–5

    return [LEVELS[int(p[0])] for p in predictions]


# ── Severity ─────────────────────────────────────────────────────────────────

def severity(target: str, predicted: str) -> str:
    if target not in LEVEL_INDEX or predicted not in LEVEL_INDEX:
        return "unknown"
    d = abs(LEVEL_INDEX[target] - LEVEL_INDEX[predicted])
    if d == 0:
        return "ok"
    if d == 1:
        return "warn"
    return "fail"


# ── HTML report ──────────────────────────────────────────────────────────────

def build_report(rows: list[dict]) -> str:
    total  = len(rows)
    ok      = sum(1 for r in rows if r["sev"] == "ok")
    warn    = sum(1 for r in rows if r["sev"] == "warn")
    fail    = sum(1 for r in rows if r["sev"] == "fail")
    unknown = sum(1 for r in rows if r["sev"] == "unknown")

    by_level: dict[str, list] = {l: [] for l in LEVELS}
    for r in rows:
        by_level.get(r["level"], []).append(r)

    sev_icon  = {"ok": "✓", "warn": "⚠", "fail": "✗", "unknown": "?"}
    sev_color = {"ok": "#1d7a47", "warn": "#b07500", "fail": "#c03030", "unknown": "#888"}
    sev_bg    = {"ok": "#edf7f1", "warn": "#fdf6e3", "fail": "#fdf0f0", "unknown": "#f5f5f5"}

    level_blocks = ""
    for level in LEVELS:
        level_rows = by_level[level]
        if not level_rows:
            continue

        rows_html = ""
        for r in level_rows:
            icon  = sev_icon.get(r["sev"], "?")
            color = sev_color.get(r["sev"], "#888")
            bg    = sev_bg.get(r["sev"], "#f5f5f5")
            pred  = r["predicted"] if r["predicted"] else "—"
            rows_html += f"""
            <tr style="background:{bg}">
              <td style="padding:9px 12px;width:32px;text-align:center;font-size:16px;color:{color};font-weight:700">{icon}</td>
              <td style="padding:9px 12px;font-size:13px;font-weight:700;white-space:nowrap;color:#555">{r['word']}</td>
              <td style="padding:9px 12px;font-size:13px;color:#3a3a44">{r['sentence']}</td>
              <td style="padding:9px 12px;font-size:12px;font-weight:700;white-space:nowrap;color:#888">{r['prompt_label']}</td>
              <td style="padding:9px 12px;text-align:center;font-size:13px;font-weight:800;color:{color}">{pred}</td>
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
                <th style="padding:8px 12px;font-size:11px;font-weight:800;color:#888;letter-spacing:0.06em;text-align:center">CEFR-SP</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>"""

    accuracy_pct = round(ok / total * 100) if total else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CEFR-SP Audit — {datetime.now().strftime('%Y-%m-%d')}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#fbfbfa;color:#15151c;margin:0;padding:32px 24px}}
  .container{{max-width:960px;margin:0 auto}}
  .summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:32px}}
  .tile{{background:#fff;border:1px solid #e8e6e2;border-radius:12px;padding:16px 18px}}
  .tile .val{{font-size:28px;font-weight:800;letter-spacing:-0.03em}}
  .tile .lbl{{font-size:12px;font-weight:600;color:#888;margin-top:3px}}
  .legend{{display:flex;gap:18px;margin-bottom:24px;flex-wrap:wrap}}
  .legend-item{{display:flex;align-items:center;gap:7px;font-size:13px;font-weight:600;color:#555}}
  .citation{{margin-top:36px;padding:16px 18px;background:#fff;border:1px solid #e8e6e2;border-radius:12px;font-size:12px;color:#888;line-height:1.7}}
</style>
</head>
<body>
<div class="container">
  <div style="margin-bottom:28px">
    <div style="font-size:11px;font-weight:800;letter-spacing:0.10em;color:#ef6b4a;text-transform:uppercase;margin-bottom:6px">SentenceStudio · Sentence Bank Audit</div>
    <h1 style="margin:0;font-size:26px;font-weight:900;letter-spacing:-0.03em">CEFR-SP Level Validation Report</h1>
    <p style="margin:6px 0 0;font-size:13px;color:#888">
      Generated {datetime.now().strftime('%B %d, %Y at %H:%M')} ·
      Model: Arase et al. EMNLP 2022 (bert-base-cased, macro-F1 84.5%) ·
      {total} sentences
    </p>
  </div>

  <div class="summary">
    <div class="tile"><div class="val">{total}</div><div class="lbl">Total sentences</div></div>
    <div class="tile"><div class="val" style="color:#1d7a47">{ok}</div><div class="lbl">✓ On target</div></div>
    <div class="tile"><div class="val" style="color:#b07500">{warn}</div><div class="lbl">⚠ Off by 1 band</div></div>
    <div class="tile"><div class="val" style="color:#c03030">{fail}</div><div class="lbl">✗ Off by 2+ bands</div></div>
    <div class="tile"><div class="val">{accuracy_pct}%</div><div class="lbl">On-target rate</div></div>
  </div>

  <div class="legend">
    <div class="legend-item"><span style="color:#1d7a47;font-size:16px">✓</span> CEFR-SP prediction matches target — keep</div>
    <div class="legend-item"><span style="color:#b07500;font-size:16px">⚠</span> Off by 1 band — review recommended</div>
    <div class="legend-item"><span style="color:#c03030;font-size:16px">✗</span> Off by 2+ bands — replace before study</div>
  </div>

  {level_blocks}

  <div class="citation">
    <strong style="color:#555">Model citation:</strong><br>
    Arase, Y., Uchida, S., &amp; Kajiwara, T. (2022). CEFR-Based Sentence Difficulty Annotation and Assessment.
    <em>Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)</em>, pp. 6206–6219.
    <a href="https://aclanthology.org/2022.emnlp-main.416" style="color:#ef6b4a">https://aclanthology.org/2022.emnlp-main.416</a><br>
    Pretrained model: <a href="https://doi.org/10.5281/zenodo.7234096" style="color:#ef6b4a">https://doi.org/10.5281/zenodo.7234096</a>
  </div>
</div>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate sentence bank with CEFR-SP model")
    parser.add_argument("--repo",  required=True, help="Path to cloned CEFR-SP repository")
    parser.add_argument("--model", required=True, help="Path to level_estimator.ckpt")
    parser.add_argument("--csv",   default=str(DEFAULT_CSV),  help="Input CSV")
    parser.add_argument("--out",   default=str(DEFAULT_OUT),  help="Output HTML report")
    parser.add_argument("--batch", type=int, default=32,      help="Sentences per forward pass")
    args = parser.parse_args()

    repo_path  = Path(args.repo).expanduser()
    ckpt_path  = Path(args.model).expanduser()
    csv_path   = Path(args.csv)
    out_path   = Path(args.out)

    # Load model
    model = load_model(repo_path, ckpt_path)

    # Load CSV
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = [r for r in reader if r.get("status", "").lower() == "accepted"]

    if not all_rows:
        print("No accepted sentences found.")
        sys.exit(0)

    print(f"Classifying {len(all_rows)} sentences (batch size {args.batch}) …\n")

    results = []
    for i in range(0, len(all_rows), args.batch):
        batch = all_rows[i : i + args.batch]
        end   = min(i + args.batch, len(all_rows))
        sentences = [r["sentence"] for r in batch]

        print(f"  [{i+1}–{end} / {len(all_rows)}]", end=" ", flush=True)
        t0 = time.time()
        predictions = classify_batch(model, sentences)
        elapsed = time.time() - t0

        for row, pred in zip(batch, predictions):
            sev = severity(row["level"], pred)
            results.append({**row, "predicted": pred, "sev": sev})
            print({"ok": "✓", "warn": "⚠", "fail": "✗"}.get(sev, "?"), end="", flush=True)

        print(f"  ({elapsed:.1f}s)")

    # Print summary
    ok   = sum(1 for r in results if r["sev"] == "ok")
    warn = sum(1 for r in results if r["sev"] == "warn")
    fail = sum(1 for r in results if r["sev"] == "fail")
    print(f"\n{'─'*50}")
    print(f"Results: ✓ {ok} on target  ⚠ {warn} off by 1  ✗ {fail} off by 2+")
    print(f"On-target rate: {round(ok/len(results)*100)}%  (CEFR-SP model baseline: 84.5%)")
    print(f"{'─'*50}")

    # Write report
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(build_report(results), encoding="utf-8")
    print(f"\nReport saved → {out_path}")
    print("Open in your browser to review flagged sentences.")


if __name__ == "__main__":
    main()
