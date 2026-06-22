"""
distractor_logic.py — Distractor generation for multiple-choice vocabulary items.

Three-layer constraint system:
  Layer 1 - POS match (hard filter)
  Layer 2 - CEFR proximity (target band +/- 1)
  Layer 3 - Semantic type selection (semantic neighbor / form neighbor / unrelated)
  Layer 4 - Sentence-frame plausibility heuristic (lightweight, no API calls)
"""

import re
import random
from pathlib import Path
from functools import lru_cache

import pandas as pd

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_INDEX = {level: i for i, level in enumerate(LEVELS)}

BASE_DIR = Path(__file__).resolve().parent
LEVELS_DIR = BASE_DIR / "levels"

TARGET_DISTRACTOR_COUNT = 3
FORM_NEIGHBOR_MAX_DISTANCE = 3
TRIVIAL_INFLECTION_MAX_DISTANCE = 1


def _find_csv_for_level(level: str):
    level_dir = LEVELS_DIR / level
    if not level_dir.exists():
        return None
    words_csv = level_dir / "words.csv"
    if words_csv.exists():
        return words_csv
    csv_files = list(level_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    return None


@lru_cache(maxsize=None)
def _read_level_dataframe(level: str):
    csv_path = _find_csv_for_level(level)
    if csv_path is None:
        return None
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        return None
    df.columns = df.columns.str.strip()
    word_col = None
    for col in df.columns:
        if col.strip().lower() == "headword":
            word_col = col
            break
    if word_col is None:
        return None
    if "pos" not in df.columns:
        return None
    df["headword"] = df[word_col].astype(str).str.strip().str.lower()
    df["pos"] = df["pos"].astype(str).str.strip().str.lower()
    df["CEFR"] = level.upper()
    for col in ("CoreInventory 1", "CoreInventory 2"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


@lru_cache(maxsize=None)
def _combined_pool_dataframe() -> pd.DataFrame:
    frames = []
    for level in LEVELS:
        df = _read_level_dataframe(level)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["headword", "pos", "CEFR", "CoreInventory 1", "CoreInventory 2"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["headword", "pos"])


def clear_cache():
    _read_level_dataframe.cache_clear()
    _combined_pool_dataframe.cache_clear()


def _allowed_bands(level: str) -> list:
    """Return the target band plus one band either side (study design Layer 2)."""
    if level not in LEVEL_INDEX:
        return [level]
    i = LEVEL_INDEX[level]
    bands = [level]
    if i > 0:
        bands.append(LEVELS[i - 1])
    if i < len(LEVELS) - 1:
        bands.append(LEVELS[i + 1])
    return bands


def build_candidate_pool(word: str, level: str, pos: str) -> pd.DataFrame:
    pool = _combined_pool_dataframe()
    if pool.empty:
        return pool
    bands = _allowed_bands(level)
    pos_clean = (pos or "").strip().lower()
    filtered = pool[
        (pool["pos"] == pos_clean)
        & (pool["CEFR"].isin(bands))
        & (pool["headword"] != word.lower())
    ].copy()
    return filtered


def _levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _guess_inflection_suffix(sentence: str, headword: str) -> str:
    tokens = re.findall(r"[A-Za-z']+", sentence.lower())
    stem = headword.lower()
    for tok in tokens:
        if tok == stem:
            return ""
        if tok.startswith(stem) and len(tok) > len(stem):
            return tok[len(stem):]
        if stem.endswith("e") and tok.startswith(stem[:-1]) and len(tok) > len(stem) - 1:
            return tok[len(stem) - 1:]
    return ""


def passes_sentence_frame_filter(sentence: str, target_headword: str, distractor_headword: str) -> bool:
    """
    Lightweight morphological plausibility check (Option B from study design).

    Returns False only when we can be fairly confident the distractor headword
    cannot take the same inflection as the target appears with in the sentence.

    Rules:
    - Common suffixes (ed/ing/s/es): accept everything — all same-POS words
      can take these inflections. The old logic that rejected headwords already
      ending in the suffix was inverted and caused false rejections (e.g. "assess"
      rejected for a target appearing as "assesses").
    - Comparative/superlative (er/est): only short words (≤ 8 chars) are
      plausibly inflectable this way; long adjectives take "more/most" instead.
    - Unknown suffix: accept by default (safe fallback).
    """
    suffix = _guess_inflection_suffix(sentence, target_headword)
    if suffix == "":
        return True
    if suffix in ("ed", "ing", "s", "es"):
        return True  # All same-POS headwords can take these inflections
    if suffix in ("er", "est"):
        return len(distractor_headword) <= 8
    return True


def _shares_topic(row_a, core1: str, core2: str) -> bool:
    topics_a = {row_a.get("CoreInventory 1", ""), row_a.get("CoreInventory 2", "")}
    topics_a.discard("")
    topics_b = {core1, core2}
    topics_b.discard("")
    return bool(topics_a & topics_b)


def select_distractors(
    word: str,
    level: str,
    sentence: str,
    pos: str,
    core1: str = "",
    core2: str = "",
    seed: int = None,
) -> dict:
    rng = random.Random(seed)

    pool = build_candidate_pool(word, level, pos)
    if pool.empty:
        return {"distractors": [], "fallback": True, "fallback_reason": "empty_pool_for_pos_cefr"}

    pool = pool[pool["headword"].apply(
        lambda hw: passes_sentence_frame_filter(sentence, word, hw)
    )].copy()
    if pool.empty:
        return {"distractors": [], "fallback": True, "fallback_reason": "no_candidate_passed_sentence_frame"}

    pool["_dist"] = pool["headword"].apply(lambda hw: _levenshtein(hw, word.lower()))
    pool = pool[pool["_dist"] > TRIVIAL_INFLECTION_MAX_DISTANCE].copy()
    if pool.empty:
        return {"distractors": [], "fallback": True, "fallback_reason": "only_trivial_inflections_remained"}

    chosen = []
    chosen_words = set()

    if core1 or core2:
        topic_matches = pool[pool.apply(lambda r: _shares_topic(r, core1, core2), axis=1)]
        topic_matches = topic_matches[~topic_matches["headword"].isin(chosen_words)]
        if not topic_matches.empty:
            pick = topic_matches.sample(1, random_state=rng.randint(0, 1_000_000)).iloc[0]
            chosen.append({"word": pick["headword"], "type": "semantic"})
            chosen_words.add(pick["headword"])

    form_candidates = pool[
        (pool["_dist"] <= FORM_NEIGHBOR_MAX_DISTANCE)
        & (~pool["headword"].isin(chosen_words))
    ]
    if not form_candidates.empty:
        pick = form_candidates.sample(1, random_state=rng.randint(0, 1_000_000)).iloc[0]
        chosen.append({"word": pick["headword"], "type": "form"})
        chosen_words.add(pick["headword"])

    remaining_pool = pool[~pool["headword"].isin(chosen_words)]
    needed = TARGET_DISTRACTOR_COUNT - len(chosen)
    if needed > 0 and not remaining_pool.empty:
        n = min(needed, len(remaining_pool))
        picks = remaining_pool.sample(n=n, random_state=rng.randint(0, 1_000_000))
        for _, pick in picks.iterrows():
            chosen.append({"word": pick["headword"], "type": "unrelated"})
            chosen_words.add(pick["headword"])

    fallback = len(chosen) < TARGET_DISTRACTOR_COUNT
    reason = None
    if fallback:
        reason = "insufficient_candidates_after_all_slots"

    return {
        "distractors": chosen[:TARGET_DISTRACTOR_COUNT],
        "fallback": fallback,
        "fallback_reason": reason,
    }
