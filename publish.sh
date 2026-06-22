#!/bin/bash
# ── Publish to Railway ──────────────────────────────────────────────────────
# Commits the latest sentence data and pushes to GitHub.
# Railway auto-redeploys on push — the phrase book updates within ~60 seconds.

cd "$(dirname "$0")"

echo "Checking for changes to publish..."

# Always stage the data files
git add output/generated_sentences.csv output/distractor_candidates.csv 2>/dev/null

# Check if there's anything to commit
if git diff --cached --quiet; then
  echo "No new sentences to publish — Railway is already up to date."
  exit 0
fi

# Count how many sentences are in the file
SENTENCE_COUNT=$(tail -n +2 output/generated_sentences.csv | grep -c ',accepted,' 2>/dev/null || echo "?")

git commit -m "data: publish sentence bank ($SENTENCE_COUNT accepted sentences)"
git push origin main

echo ""
echo "  ✦ Published! Railway will redeploy in ~60 seconds."
echo "  Check: https://sentence-generator-production.up.railway.app/books"
echo ""
