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


_POS_TO_WN = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r"}


def _shares_topic(row_a, core1: str, core2: str) -> bool:
    """CoreInventory topic-tag overlap — used as Slot A fallback when WordNet finds no pool match."""
    topics_a = {str(row_a.get("CoreInventory 1", "")), str(row_a.get("CoreInventory 2", ""))}
    topics_a.discard("")
    topics_b = {core1, core2}
    topics_b.discard("")
    return bool(topics_a & topics_b)


def _get_wordnet_synonyms(word: str, pos: str) -> set:
    """
    Return near-synonym headwords using WordNet synsets (top 3 senses).
    Multi-word lemmas excluded — CEFR list uses single tokens.
    Falls back to an empty set if WordNet is unavailable.
    """
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return set()

    wn_pos = _POS_TO_WN.get((pos or "").strip().lower())
    if wn_pos is None:
        return set()

    synonyms: set = set()
    try:
        synsets = wn.synsets(word.lower(), pos=wn_pos)
        for syn in synsets[:3]:
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name != word.lower() and " " not in name:
                    synonyms.add(name)
    except Exception:
        pass
    return synonyms


def _get_wordnet_siblings(word: str, pos: str) -> set:
    """
    Return co-hyponym headwords via WordNet hypernym→hyponym path.

    Per Susanti et al. (2018): siblings share the same hypernym, so they're
    semantically related but meaningfully distinct — ideal distractors because
    they occupy the same semantic category as the target word.

    E.g. for 'dog' (animal): cat, horse, rabbit, wolf, …
    E.g. for 'happy' (positive emotion adjective): joyful, content, elated, …

    Only top-2 senses checked to avoid noise from rare/metaphorical meanings.
    Multi-word lemmas excluded.
    """
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return set()

    wn_pos = _POS_TO_WN.get((pos or "").strip().lower())
    if wn_pos is None:
        return set()

    siblings: set = set()
    try:
        synsets = wn.synsets(word.lower(), pos=wn_pos)
        for syn in synsets[:2]:
            for hypernym in syn.hypernyms()[:2]:          # check up to 2 parents
                for sibling_syn in hypernym.hyponyms():   # co-hyponyms = siblings
                    if sibling_syn == syn:
                        continue
                    for lemma in sibling_syn.lemmas():
                        name = lemma.name().replace("_", " ").lower()
                        if name != word.lower() and " " not in name:
                            siblings.add(name)
    except Exception:
        pass
    return siblings


def _word_length_ok(word: str, target: str, max_diff: int = 4) -> bool:
    """
    Heaton (1989) criterion 3: distractors should have approximately the same
    length as the target word. We allow ±max_diff characters (default 4).
    Very short words (≤4 chars) get a tighter window of ±2.
    """
    diff = abs(len(word) - len(target))
    if len(target) <= 4:
        return diff <= 2
    return diff <= max_diff


def _are_synonyms(word_a: str, word_b: str, pos: str) -> bool:
    """
    Check whether two words share a WordNet synset (are synonyms).
    Used to avoid synonym pairs in the distractor set per Susanti et al. (2018),
    who note that synonym pairs are 'potentially dangerous' — test-wise learners
    can eliminate both options simultaneously.
    """
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return False

    wn_pos = _POS_TO_WN.get((pos or "").strip().lower())
    if wn_pos is None:
        return False

    try:
        synsets_a = set(wn.synsets(word_a.lower(), pos=wn_pos))
        synsets_b = set(wn.synsets(word_b.lower(), pos=wn_pos))
        return bool(synsets_a & synsets_b)
    except Exception:
        return False


def select_distractors(
    word: str,
    level: str,
    sentence: str,
    pos: str,
    core1: str = "",  # kept for API compatibility, no longer used for slot selection
    core2: str = "",  # kept for API compatibility, no longer used for slot selection
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

    # Heaton (1989) criterion 3 — approximately same length as target word.
    # Applied as a soft filter: only drop if the full pool still has candidates.
    length_filtered = pool[pool["headword"].apply(lambda hw: _word_length_ok(hw, word))]
    if not length_filtered.empty:
        pool = length_filtered

    chosen = []
    chosen_words = set()

    # Slot A — semantic neighbor
    # Priority order per Susanti et al. (2018):
    #   1. WordNet siblings (co-hyponyms) — share same hypernym, same semantic category,
    #      most effective at deceiving learners without being synonymous with the target.
    #   2. WordNet near-synonyms — semantically close but distinct enough to distract.
    #   3. CoreInventory topic-tag match — CEFR-J specific fallback when WordNet
    #      synonyms/siblings are all outside the ±1 CEFR band pool.
    semantic_pick = None

    # Strategy 1: WordNet siblings
    siblings = _get_wordnet_siblings(word, pos)
    if siblings:
        sib_matches = pool[
            pool["headword"].isin(siblings) & ~pool["headword"].isin(chosen_words)
        ]
        if not sib_matches.empty:
            semantic_pick = sib_matches.sample(1, random_state=rng.randint(0, 1_000_000)).iloc[0]
            chosen.append({"word": semantic_pick["headword"], "type": "semantic"})
            chosen_words.add(semantic_pick["headword"])

    # Strategy 2: WordNet synonyms
    if semantic_pick is None:
        synonyms = _get_wordnet_synonyms(word, pos)
        if synonyms:
            syn_matches = pool[
                pool["headword"].isin(synonyms) & ~pool["headword"].isin(chosen_words)
            ]
            if not syn_matches.empty:
                semantic_pick = syn_matches.sample(1, random_state=rng.randint(0, 1_000_000)).iloc[0]
                chosen.append({"word": semantic_pick["headword"], "type": "semantic"})
                chosen_words.add(semantic_pick["headword"])

    # Strategy 3 fallback: CoreInventory topic-tag match
    if semantic_pick is None and (core1 or core2):
        topic_matches = pool[
            pool.apply(lambda r: _shares_topic(r, core1, core2), axis=1)
            & ~pool["headword"].isin(chosen_words)
        ]
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
        # Susanti et al. (2018): avoid synonym pairs in the option set —
        # test-wise learners can eliminate both, making the question easier.
        # Sample a larger candidate set and greedily pick non-synonymous words.
        candidate_count = min(needed * 6, len(remaining_pool))
        candidates = remaining_pool.sample(n=candidate_count, random_state=rng.randint(0, 1_000_000))
        for _, pick in candidates.iterrows():
            if len(chosen) >= TARGET_DISTRACTOR_COUNT:
                break
            hw = pick["headword"]
            # Skip if this word is a synonym of any already-chosen distractor
            is_syn_pair = any(
                _are_synonyms(hw, c["word"], pos) for c in chosen
            )
            if not is_syn_pair:
                chosen.append({"word": hw, "type": "unrelated"})
                chosen_words.add(hw)

    fallback = len(chosen) < TARGET_DISTRACTOR_COUNT
    reason = None
    if fallback:
        reason = "insufficient_candidates_after_all_slots"

    return {
        "distractors": chosen[:TARGET_DISTRACTOR_COUNT],
        "fallback": fallback,
        "fallback_reason": reason,
    }


def llm_frame_check(sentence: str, word: str, distractors: list, client) -> list:
    """
    Option A sentence-frame plausibility filter (study design §6.4).

    Sends one batched Groq call (llama-3.1-8b-instant, temp=0) asking whether
    each distractor headword, substituted into the sentence gap, produces
    grammatically possible English.

    Adds 'llm_ok': True/False to each distractor dict.
    Falls back to llm_ok=True for all if the call fails or parsing is ambiguous,
    so a failed check never silently drops candidates — the human reviewer decides.

    Args:
        sentence: the full example sentence
        word: the target headword that appears in the sentence
        distractors: list of {"word": str, "type": str, ...}
        client: a Groq client instance (or None to skip)

    Returns:
        The same list with 'llm_ok' bool added to each item.
    """
    if not distractors or client is None:
        return [{**d, "llm_ok": True} for d in distractors]

    candidate_lines = "\n".join(f"- {d['word']}" for d in distractors)
    prompt = (
        f'Sentence (with the target word "{word}" replaced by ___): '
        f'"{sentence.replace(word, "___", 1)}"\n\n'
        f"For each word below, would substituting it into the ___ gap produce "
        f"grammatically possible English (ignoring meaning — only grammar)?\n"
        f"Answer YES or NO for each, one per line, same order:\n"
        f"{candidate_lines}"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        raw_lines = [
            ln.strip().lower()
            for ln in response.choices[0].message.content.strip().splitlines()
            if ln.strip()
        ]

        def _parse_yn(line: str) -> bool:
            # Strip bullet/dash/word-prefix so "- pardon: yes" → "yes"
            clean = re.sub(r'^[-•\*\s]*(?:\w[\w\s]*:\s*)?', '', line).strip()
            # A line is a YES if it starts with "yes"; anything else (including
            # "no", "n/a", parse failures) defaults to True to avoid false rejects.
            if clean.startswith("no"):
                return False
            return True  # "yes", unclear, or empty → safe default

        result = []
        for i, d in enumerate(distractors):
            ok = _parse_yn(raw_lines[i]) if i < len(raw_lines) else True
            result.append({**d, "llm_ok": ok})
        return result
    except Exception:
        # Any API error → keep all candidates, flag as unchecked (True = don't block)
        return [{**d, "llm_ok": True} for d in distractors]
