import os
import json
import re
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

import distractor_logic

API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CLIENT = Groq(api_key=API_KEY) if API_KEY else None

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
BASE_DIR = Path(__file__).resolve().parent
LEVELS_DIR = BASE_DIR / "levels"
TRACKING_FILE = BASE_DIR / "tracking" / "used_words.json"
OUTPUT_FILE = BASE_DIR / "output" / "generated_sentences.csv"
DISTRACTOR_FILE = BASE_DIR / "output" / "distractor_candidates.csv"
TRACKING_FILE.parent.mkdir(exist_ok=True)
OUTPUT_FILE.parent.mkdir(exist_ok=True)
DISTRACTOR_FILE.parent.mkdir(exist_ok=True)
TARGET_SENTENCES_PER_WORD = 15
TARGET_DISTRACTORS_PER_SENTENCE = 3
MAX_GENERATION_ATTEMPTS = 4

PROMPT_PATTERNS = [
    {"label": "contrast", "instruction": "Show a clear contrast, mismatch, or surprising comparison."},
    {"label": "cause-effect", "instruction": "Make the sentence show a cause and its effect."},
    {"label": "problem-solution", "instruction": "Present a problem and the response or solution."},
    {"label": "emotional-context", "instruction": "Show a realistic emotional reaction or feeling."},
    {"label": "decision-result", "instruction": "Show a choice, action, and the result that followed."},
]
PROMPT_LABELS = {pattern["label"] for pattern in PROMPT_PATTERNS}

app = Flask(__name__)


# ── Vocabulary helpers ──────────────────────────────────────────────

def is_real_word(token: str) -> bool:
    token = token.strip()
    if re.fullmatch(r"[a-z]+(?:[-'][a-z]+)*", token, re.IGNORECASE) and len(token) >= 2:
        return True
    return False


def find_csv_for_level(level: str):
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
def read_level_dataframe(level: str):
    csv_path = find_csv_for_level(level)
    if csv_path is None:
        return None
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig')
    except Exception:
        return None
    df.columns = df.columns.str.strip()
    return df


@lru_cache(maxsize=None)
def load_level_words(level: str) -> set:
    df = read_level_dataframe(level)
    if df is None:
        return set()
    word_col = None
    for col in df.columns:
        if col.strip().lower() == 'headword':
            word_col = col
            break
    if word_col is None:
        return set()
    words = df[word_col].dropna().astype(str).str.strip().str.lower()
    words = words[words.apply(is_real_word)]
    return set(words)


@lru_cache(maxsize=None)
def load_allowed_vocab(up_to_level: str) -> set:
    allowed = set()
    for lvl in LEVELS:
        if LEVEL_INDEX[lvl] > LEVEL_INDEX[up_to_level]:
            break
        allowed.update(load_level_words(lvl))
    return allowed


def get_word_info(word: str, level: str) -> dict:
    df = read_level_dataframe(level)
    if df is None:
        return {}
    word_col = None
    for col in df.columns:
        if col.strip().lower() == 'headword':
            word_col = col
            break
    if word_col is None:
        return {}
    df[word_col] = df[word_col].astype(str).str.strip().str.lower()
    row = df[df[word_col] == word.lower()]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "pos": str(r.get("pos", "")).strip(),
        "core1": str(r.get("CoreInventory 1", "")).strip(),
        "core2": str(r.get("CoreInventory 2", "")).strip(),
        "threshold": str(r.get("Threshold", "")).strip(),
    }


# ── Tracking ────────────────────────────────────────────────────────

def load_used_words() -> dict:
    if TRACKING_FILE.exists():
        try:
            with open(TRACKING_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_used_words(data: dict):
    with open(TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=2)


def mark_word_used(level: str, word: str):
    used = load_used_words()
    used.setdefault(level, {})[word] = datetime.now().isoformat()
    save_used_words(used)


def get_unused_words(level: str) -> list:
    level_words = load_level_words(level)
    try:
        used_dict = load_used_words()
        used = used_dict.get(level, {})
        if not isinstance(used, dict):
            used = {}
        return sorted([w for w in level_words if w not in used])
    except Exception:
        return sorted(level_words)


# ── Sentence generation ────────────────────────────────────────────

def build_prompt(word: str, level: str, info: dict, allowed_words: set) -> str:
    return build_generation_prompt(word, level, info, allowed_words, TARGET_SENTENCES_PER_WORD, ())


def build_generation_prompt(
    word: str,
    level: str,
    info: dict,
    allowed_words: set,
    target_count: int,
    excluded_sentences: tuple[str, ...] = (),
) -> str:
    pos = info.get("pos", "word")
    core1 = info.get("core1", "")
    core2 = info.get("core2", "")
    topics = "family, school, jobs, money, travel, hobbies, technology, friends"
    if core1 or core2:
        extras = [c for c in [core1, core2] if c]
        topics += ", " + ", ".join(extras)

    # Send a small random sample (60 words) just to calibrate the model's
    # vocabulary register — not the full list. The validation step catches
    # any out-of-level words after generation.
    sample_words = sorted(allowed_words)
    if len(sample_words) > 60:
        sample_words = random.sample(sample_words, 60)
        sample_words.sort()

    pattern_instructions = "\n".join(
        f'- "{pattern["label"]}": {pattern["instruction"]}'
        for pattern in PROMPT_PATTERNS
    )
    excluded_text = ""
    if excluded_sentences:
        preview = "\n".join(f"- {sentence}" for sentence in excluded_sentences[:25])
        excluded_text = f"\nAlready saved or generated sentences to avoid repeating:\n{preview}\n"

    return f"""You are an English language teacher creating example sentences for learners.
Target word: "{word}"
CEFR level: {level}
Part of speech: {pos}

Rules:
- Generate exactly {target_count} items, each using the target word naturally.
- Output ONLY a valid JSON array. Do not add markdown fences, comments, or extra text.
- Every item must be an object with exactly these keys: "label" and "sentence".
- Use only these labels:
{pattern_instructions}
- Spread the labels as evenly as possible across the {target_count} items.
- Situations must be realistic and relatable (e.g., {topics}).
- Keep vocabulary at CEFR {level} level. Avoid advanced or rare words.
- If you cover the target word, a student should be able to guess it from context.
- Each sentence must be distinct from the others.
{excluded_text}
Level calibration sample (words typical at {level}): {', '.join(sample_words)}"""


def normalize_prompt_label(label: str) -> str:
    cleaned = re.sub(r'[^a-z]+', '-', str(label or "").strip().lower()).strip('-')
    aliases = {
        "cause-and-effect": "cause-effect",
        "cause-effect": "cause-effect",
        "contrast": "contrast",
        "problem-and-result": "problem-solution",
        "problem-result": "problem-solution",
        "problem-solution": "problem-solution",
        "solution": "problem-solution",
        "emotional-context": "emotional-context",
        "emotion": "emotional-context",
        "emotional": "emotional-context",
        "decision-result": "decision-result",
        "choice-result": "decision-result",
        "result": "decision-result",
    }
    return aliases.get(cleaned, cleaned if cleaned in PROMPT_LABELS else "unlabeled")


def parse_generated_items(raw: str) -> list:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    json_match = re.search(r"\[[\s\S]*\]", text)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, list):
                parsed = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    sentence = str(item.get("sentence", "")).strip()
                    if len(sentence) < 15:
                        continue
                    parsed.append({
                        "prompt_label": normalize_prompt_label(item.get("label", "")),
                        "sentence": sentence,
                    })
                if parsed:
                    return parsed
        except json.JSONDecodeError:
            pass

    parsed = []
    for line in text.splitlines():
        line = re.sub(r'^\d+[\.\)\-]?\s*', '', line.strip())
        if not line or line.lower().startswith("here are") or line.lower().startswith("sure"):
            continue
        label = "unlabeled"
        sentence = line
        if "|" in line:
            candidate_label, candidate_sentence = [part.strip() for part in line.split("|", 1)]
            if candidate_sentence:
                label = normalize_prompt_label(candidate_label)
                sentence = candidate_sentence
        elif ":" in line:
            candidate_label, candidate_sentence = [part.strip() for part in line.split(":", 1)]
            normalized = normalize_prompt_label(candidate_label)
            if normalized != "unlabeled" and candidate_sentence:
                label = normalized
                sentence = candidate_sentence
        if len(sentence) < 15:
            continue
        parsed.append({
            "prompt_label": label,
            "sentence": sentence,
        })
    return parsed


def normalize_sentence_key(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence.strip().lower())


def generate_sentences(
    word: str,
    level: str,
    info: dict,
    allowed_words: set,
    target_count: int = TARGET_SENTENCES_PER_WORD,
    excluded_sentences: tuple[str, ...] = (),
) -> list:
    prompt = build_generation_prompt(word, level, info, allowed_words, target_count, excluded_sentences)

    response = GROQ_CLIENT.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful English teacher."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048,
        top_p=0.95,
    )
    raw = response.choices[0].message.content.strip()
    return parse_generated_items(raw)[:target_count]


# ── Validation ──────────────────────────────────────────────────────

def validate_sentence(sentence: str, allowed: set) -> list:
    tokens = word_tokenize(sentence)
    unknown = []
    seen = set()
    for word in tokens:
        if word in {'.', ',', '!', '?', ';', ':', "'", '"', '-', '–', '—', '(', ')', '...', "'s", "n't"}:
            continue
        if re.fullmatch(r'\d+', word):
            continue
        clean = word.strip("'\".,;:!?()[]")
        if not clean or len(clean) <= 1:
            continue
        lemmas = set()
        for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]:
            lemmas.add(lemmatizer.lemmatize(clean.lower(), pos=pos))
        if not lemmas.intersection(allowed) and clean.lower() not in seen:
            unknown.append(word)
            seen.add(clean.lower())
    return unknown


# ── Persistence ─────────────────────────────────────────────────────

def load_output_dataframe() -> pd.DataFrame:
    expected_columns = [
        "level",
        "word",
        "prompt_label",
        "sentence",
        "original_sentence",
        "edited",
        "status",
        "timestamp",
    ]
    if not OUTPUT_FILE.exists():
        return pd.DataFrame(columns=expected_columns)

    df = pd.read_csv(OUTPUT_FILE)
    if "prompt_label" not in df.columns:
        df["prompt_label"] = "unlabeled"
    if "original_sentence" not in df.columns:
        df["original_sentence"] = df["sentence"].fillna("") if "sentence" in df.columns else ""
    if "edited" not in df.columns:
        df["edited"] = False
    if "status" not in df.columns:
        df["status"] = "accepted"
    if "timestamp" not in df.columns:
        df["timestamp"] = ""

    for column in ["level", "word", "prompt_label", "sentence", "original_sentence", "status", "timestamp"]:
        df[column] = df[column].fillna("").astype(str)

    df["edited"] = df["edited"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
    return df[expected_columns]


def get_saved_counts(level: str | None = None) -> dict:
    df = load_output_dataframe()
    if df.empty:
        return {}
    accepted = df[df["status"].str.lower() == "accepted"].copy()
    if level:
        accepted = accepted[accepted["level"].str.upper() == level.upper()]
    if accepted.empty:
        return {}
    counts = accepted.groupby(["level", "word"]).size().to_dict()
    if level:
        return {word: int(count) for (_, word), count in counts.items()}
    return {(lvl, word): int(count) for (lvl, word), count in counts.items()}


def get_saved_sentences_for_word(level: str, word: str) -> list:
    df = load_output_dataframe()
    if df.empty:
        return []
    accepted = df[
        (df["status"].str.lower() == "accepted")
        & (df["level"].str.upper() == level.upper())
        & (df["word"].str.lower() == word.lower())
    ]
    if accepted.empty:
        return []
    return accepted["sentence"].dropna().astype(str).tolist()


def get_word_progress(level: str) -> list:
    level_words = sorted(load_level_words(level))
    saved_counts = get_saved_counts(level)
    progress = []
    for word in level_words:
        saved_count = int(saved_counts.get(word, 0))
        remaining = max(TARGET_SENTENCES_PER_WORD - saved_count, 0)
        progress.append({
            "word": word,
            "saved_count": saved_count,
            "remaining": remaining,
            "complete": saved_count >= TARGET_SENTENCES_PER_WORD,
            "incomplete": 0 < saved_count < TARGET_SENTENCES_PER_WORD,
        })
    return progress


def get_level_summary(level: str) -> dict:
    progress = get_word_progress(level)
    total = len(progress)
    complete_count = sum(1 for item in progress if item["complete"])
    incomplete_count = sum(1 for item in progress if item["incomplete"])
    untouched_count = sum(1 for item in progress if item["saved_count"] == 0)
    available_words = [item for item in progress if not item["complete"]]
    incomplete_words = [item for item in progress if item["incomplete"]]
    random_incomplete = random.choice(incomplete_words)["word"] if incomplete_words else None
    return {
        "total": total,
        "complete_count": complete_count,
        "incomplete_count": incomplete_count,
        "untouched_count": untouched_count,
        "available_count": len(available_words),
        "random_incomplete_word": random_incomplete,
        "words": available_words,
    }


def get_word_metadata(level: str, word: str) -> dict:
    info = get_word_info(word, level)
    core1 = info.get("core1", "").strip()
    core2 = info.get("core2", "").strip()
    inventory = [value for value in [core1, core2] if value]
    return {
        "pos": info.get("pos", "").strip() or "unknown",
        "inventory": inventory or ["unknown"],
        "core1": core1,
        "core2": core2,
    }


def build_stats_payload() -> dict:
    df = load_output_dataframe()
    if df.empty:
        return {
            "summary": {
                "total_sentences": 0,
                "total_words_started": 0,
                "complete_words": 0,
                "incomplete_words": 0,
                "average_sentences_per_word": 0,
            },
            "label_summary": [],
            "level_summary": [],
            "prompt_pos_distribution": [],
            "prompt_inventory_distribution": [],
            "top_incomplete_words": [],
        }

    accepted = df[df["status"].str.lower() == "accepted"].copy()
    if accepted.empty:
        return {
            "summary": {
                "total_sentences": 0,
                "total_words_started": 0,
                "complete_words": 0,
                "incomplete_words": 0,
                "average_sentences_per_word": 0,
            },
            "label_summary": [],
            "level_summary": [],
            "prompt_pos_distribution": [],
            "prompt_inventory_distribution": [],
            "top_incomplete_words": [],
        }

    accepted["word"] = accepted["word"].str.lower()
    accepted["level"] = accepted["level"].str.upper()
    accepted["prompt_label"] = accepted["prompt_label"].apply(normalize_prompt_label)

    per_word = (
        accepted.groupby(["level", "word"])
        .size()
        .reset_index(name="sentence_count")
    )
    metadata_rows = []
    for row in per_word.itertuples(index=False):
        metadata = get_word_metadata(row.level, row.word)
        metadata_rows.append({
            "level": row.level,
            "word": row.word,
            "sentence_count": int(row.sentence_count),
            "pos": metadata["pos"],
            "inventory": metadata["inventory"],
        })
    total_words_started = len(per_word)
    complete_words = int((per_word["sentence_count"] >= TARGET_SENTENCES_PER_WORD).sum())
    incomplete_words = int(((per_word["sentence_count"] > 0) & (per_word["sentence_count"] < TARGET_SENTENCES_PER_WORD)).sum())
    average_sentences_per_word = round(float(per_word["sentence_count"].mean()), 2) if total_words_started else 0

    label_sentences = accepted.groupby("prompt_label").size()
    label_words = accepted.groupby("prompt_label")["word"].nunique()
    label_summary = [
        {
            "prompt_label": label,
            "sentence_count": int(label_sentences.get(label, 0)),
            "word_count": int(label_words.get(label, 0)),
        }
        for label in sorted(set(label_sentences.index).union(label_words.index))
    ]

    level_summary = []
    for level in LEVELS:
        level_progress = get_level_summary(level)
        started_counts = per_word[per_word["level"] == level]["sentence_count"]
        level_summary.append({
            "level": level,
            "words_started": int(len(started_counts)),
            "average_sentences_per_word": round(float(started_counts.mean()), 2) if len(started_counts) else 0,
            "complete_words": level_progress["complete_count"],
            "incomplete_words": level_progress["incomplete_count"],
        })

    labeled = accepted.copy()
    labeled["key"] = list(zip(labeled["level"], labeled["word"]))
    meta_lookup = {
        (row["level"], row["word"]): {
            "pos": row["pos"],
            "inventory": row["inventory"],
        }
        for row in metadata_rows
    }
    labeled["pos"] = labeled["key"].map(lambda key: meta_lookup.get(key, {}).get("pos", "unknown"))
    labeled["inventory"] = labeled["key"].map(lambda key: meta_lookup.get(key, {}).get("inventory", ["unknown"]))

    prompt_pos_distribution = []
    pos_grouped = labeled.groupby(["prompt_label", "pos"]).size().reset_index(name="sentence_count")
    for row in pos_grouped.itertuples(index=False):
        prompt_pos_distribution.append({
            "prompt_label": row.prompt_label,
            "pos": row.pos,
            "sentence_count": int(row.sentence_count),
        })

    inventory_rows = []
    for row in labeled[["prompt_label", "inventory"]].itertuples(index=False):
        for inventory in row.inventory:
            inventory_rows.append({
                "prompt_label": row.prompt_label,
                "inventory": inventory,
            })
    inventory_df = pd.DataFrame(inventory_rows)
    prompt_inventory_distribution = []
    if not inventory_df.empty:
        inv_grouped = inventory_df.groupby(["prompt_label", "inventory"]).size().reset_index(name="sentence_count")
        for row in inv_grouped.itertuples(index=False):
            prompt_inventory_distribution.append({
                "prompt_label": row.prompt_label,
                "inventory": row.inventory,
                "sentence_count": int(row.sentence_count),
            })

    incomplete_df = per_word[per_word["sentence_count"] < TARGET_SENTENCES_PER_WORD].copy()
    incomplete_df["remaining"] = TARGET_SENTENCES_PER_WORD - incomplete_df["sentence_count"]
    top_incomplete_words = incomplete_df.sort_values(["remaining", "level", "word"], ascending=[False, True, True]).head(20)
    top_incomplete_words = [
        {
            "level": row.level,
            "word": row.word,
            "sentence_count": int(row.sentence_count),
            "remaining": int(row.remaining),
        }
        for row in top_incomplete_words.itertuples(index=False)
    ]

    return {
        "summary": {
            "total_sentences": int(len(accepted)),
            "total_words_started": int(total_words_started),
            "complete_words": int(complete_words),
            "incomplete_words": int(incomplete_words),
            "average_sentences_per_word": average_sentences_per_word,
        },
        "label_summary": label_summary,
        "level_summary": level_summary,
        "prompt_pos_distribution": prompt_pos_distribution,
        "prompt_inventory_distribution": prompt_inventory_distribution,
        "top_incomplete_words": top_incomplete_words,
    }

def save_accepted_sentences(level: str, word: str, accepted: list):
    timestamp = datetime.now().isoformat()
    normalized = []
    for item in accepted:
        if isinstance(item, dict):
            sentence = str(item.get("sentence", "")).strip()
            if not sentence:
                continue
            normalized.append({
                "level": level,
                "word": word,
                "prompt_label": normalize_prompt_label(item.get("prompt_label", "")),
                "sentence": sentence,
                "original_sentence": str(item.get("original_sentence", sentence)).strip(),
                "edited": bool(item.get("edited", False)),
                "status": "accepted",
                "timestamp": timestamp,
            })
        elif isinstance(item, str) and item.strip():
            normalized.append({
                "level": level,
                "word": word,
                "prompt_label": "unlabeled",
                "sentence": item.strip(),
                "original_sentence": item.strip(),
                "edited": False,
                "status": "accepted",
                "timestamp": timestamp,
            })

    if not normalized:
        return

    df_new = pd.DataFrame(normalized)
    if OUTPUT_FILE.exists():
        df_old = pd.read_csv(OUTPUT_FILE)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(OUTPUT_FILE, index=False)


# ── Flask routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", levels=LEVELS)


@app.route("/api/words/<level>")
def api_words(level):
    if level not in LEVELS:
        return jsonify(error="Invalid level"), 400
    summary = get_level_summary(level)
    return jsonify(
        words=summary["words"],
        total=summary["total"],
        complete_count=summary["complete_count"],
        incomplete_count=summary["incomplete_count"],
        untouched_count=summary["untouched_count"],
        available_count=summary["available_count"],
        random_incomplete_word=summary["random_incomplete_word"],
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json
    word = data.get("word", "").strip().lower()
    level = data.get("level", "").strip().upper()
    if not word or level not in LEVELS:
        return jsonify(error="Invalid word or level"), 400
    if not API_KEY:
        return jsonify(error="GROQ_API_KEY not set"), 500

    allowed = load_allowed_vocab(level)
    info = get_word_info(word, level)
    if not info:
        info = {"pos": "unknown", "core1": "", "core2": ""}

    existing_sentences = get_saved_sentences_for_word(level, word)
    existing_keys = {normalize_sentence_key(sentence) for sentence in existing_sentences}
    saved_count = len(existing_sentences)
    remaining_needed = max(TARGET_SENTENCES_PER_WORD - saved_count, 0)
    if remaining_needed == 0:
        return jsonify(
            sentences=[],
            allowed_levels=LEVELS[:LEVEL_INDEX[level] + 1],
            saved_count=saved_count,
            target_count=TARGET_SENTENCES_PER_WORD,
            remaining_needed=0,
            attempts=0,
            already_complete=True,
        )

    generated_items = []
    generated_keys = set()
    attempts = 0
    try:
        while len(generated_items) < remaining_needed and attempts < MAX_GENERATION_ATTEMPTS:
            attempts += 1
            excluded = tuple((existing_sentences + [item["sentence"] for item in generated_items])[:25])
            batch = generate_sentences(
                word,
                level,
                info,
                allowed,
                target_count=remaining_needed - len(generated_items),
                excluded_sentences=excluded,
            )
            for item in batch:
                sentence = item.get("sentence", "").strip()
                key = normalize_sentence_key(sentence)
                if not sentence or key in existing_keys or key in generated_keys:
                    continue
                generated_items.append(item)
                generated_keys.add(key)
                if len(generated_items) >= remaining_needed:
                    break
    except Exception as e:
        return jsonify(error=str(e)), 500

    results = []
    for item in generated_items:
        sentence = item.get("sentence", "").strip()
        prompt_label = item.get("prompt_label", "unlabeled")
        unknown = validate_sentence(sentence, allowed)
        results.append({
            "sentence": sentence,
            "prompt_label": prompt_label,
            "valid": len(unknown) == 0,
            "unknown_words": unknown,
        })
    return jsonify(
        sentences=results,
        allowed_levels=LEVELS[:LEVEL_INDEX[level] + 1],
        saved_count=saved_count,
        target_count=TARGET_SENTENCES_PER_WORD,
        remaining_needed=max(TARGET_SENTENCES_PER_WORD - saved_count, 0),
        generated_count=len(results),
        attempts=attempts,
        already_complete=False,
    )


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.json
    word = data.get("word", "").strip().lower()
    level = data.get("level", "").strip().upper()
    sentences = data.get("sentences", [])
    if not word or level not in LEVELS or not sentences:
        return jsonify(error="Invalid data"), 400
    accepted_items = []
    for item in sentences:
        if isinstance(item, str) and item.strip():
            accepted_items.append({
                "sentence": item.strip(),
                "original_sentence": item.strip(),
                "prompt_label": "unlabeled",
                "edited": False,
            })
        elif isinstance(item, dict):
            sentence = str(item.get("sentence", "")).strip()
            if sentence:
                accepted_items.append({
                    "sentence": sentence,
                    "original_sentence": str(item.get("original_sentence", sentence)).strip() or sentence,
                    "prompt_label": item.get("prompt_label", "unlabeled"),
                    "edited": bool(item.get("edited", False)),
                })
    if not accepted_items:
        return jsonify(error="No valid sentences to save"), 400

    save_accepted_sentences(level, word, accepted_items)
    new_total = len(get_saved_sentences_for_word(level, word))
    if new_total >= TARGET_SENTENCES_PER_WORD:
        mark_word_used(level, word)
    return jsonify(
        ok=True,
        count=len(accepted_items),
        total_saved=new_total,
        remaining=max(TARGET_SENTENCES_PER_WORD - new_total, 0),
        complete=new_total >= TARGET_SENTENCES_PER_WORD,
    )


@app.route("/api/history")
def api_history():
    df = load_output_dataframe()
    if df.empty:
        return jsonify(sentences=[])
    df = df.sort_values("timestamp", ascending=False)
    records = df.to_dict(orient="records")
    return jsonify(sentences=records)


@app.route("/api/levels")
def api_levels():
    descs = {
        "A1": {"name": "Beginner", "desc": "Basic everyday phrases and very simple sentences."},
        "A2": {"name": "Elementary", "desc": "Simple, routine exchanges on familiar topics."},
        "B1": {"name": "Intermediate", "desc": "Cope with most travel situations and describe experiences."},
        "B2": {"name": "Upper-intermediate", "desc": "Interact fluently and spontaneously on a wide range of topics."},
        "C1": {"name": "Advanced", "desc": "Use language flexibly for academic and professional aims."},
        "C2": {"name": "Proficiency", "desc": "Understand almost everything and express meaning precisely."},
    }
    result = []
    for level in LEVELS:
        summary = get_level_summary(level)
        info = descs.get(level, {"name": level, "desc": ""})
        df = load_output_dataframe()
        sentence_count = 0
        if not df.empty:
            sentence_count = int(((df["status"].str.lower() == "accepted") & (df["level"].str.upper() == level)).sum())
        result.append({
            "code": level,
            "name": info["name"],
            "desc": info["desc"],
            "total_words": summary["total"],
            "complete_count": summary["complete_count"],
            "sentence_count": sentence_count,
        })
    return jsonify(levels=result)


@app.route("/api/phrasebook/<level>")
def api_phrasebook(level):
    if level not in LEVELS:
        return jsonify(error="Invalid level"), 400
    df = load_output_dataframe()
    if df.empty:
        return jsonify(topics=[])

    accepted = df[
        (df["status"].str.lower() == "accepted")
        & (df["level"].str.upper() == level.upper())
    ].copy()
    if accepted.empty:
        return jsonify(topics=[])

    accepted["word"] = accepted["word"].str.lower()
    accepted["prompt_label"] = accepted["prompt_label"].apply(normalize_prompt_label)

    word_info_cache = {}
    topics_map = {}
    for _, row in accepted.iterrows():
        word = row["word"]
        if word not in word_info_cache:
            info = get_word_info(word, level)
            pos = info.get("pos", "").strip() or "unknown"
            raw_topics = set()
            for field in ("core1", "core2", "threshold"):
                val = info.get(field, "").strip()
                if val and val.lower() != "nan":
                    raw_topics.add(val)
            word_info_cache[word] = {
                "pos": pos,
                "topics": raw_topics if raw_topics else {"General"},
            }
        cached = word_info_cache[word]
        sentence_data = {
            "prompt_label": row["prompt_label"],
            "sentence": row["sentence"],
        }
        for topic in cached["topics"]:
            if topic not in topics_map:
                topics_map[topic] = {}
            if word not in topics_map[topic]:
                topics_map[topic][word] = {"pos": cached["pos"], "sentences": []}
            topics_map[topic][word]["sentences"].append(sentence_data)

    result = []
    for topic_name in sorted(topics_map.keys()):
        words_data = topics_map[topic_name]
        words_list = []
        for w in sorted(words_data.keys()):
            wd = words_data[w]
            words_list.append({
                "word": w,
                "pos": wd["pos"],
                "count": len(wd["sentences"]),
                "sentences": wd["sentences"],
            })
        result.append({
            "topic": topic_name,
            "word_count": len(words_list),
            "sentence_count": sum(w["count"] for w in words_list),
            "words": words_list,
        })

    return jsonify(topics=result)


# ── Distractor persistence ───────────────────────────────────────────

def load_distractor_dataframe() -> pd.DataFrame:
    expected_columns = [
        "level", "word", "pos", "prompt_label", "sentence",
        "sentence_id", "distractor_word", "distractor_type",
        "status", "fallback", "fallback_reason", "timestamp",
    ]
    if not DISTRACTOR_FILE.exists():
        return pd.DataFrame(columns=expected_columns)
    df = pd.read_csv(DISTRACTOR_FILE)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""
    for col in ["level", "word", "pos", "prompt_label", "sentence", "sentence_id",
                "distractor_word", "distractor_type", "status", "fallback_reason", "timestamp"]:
        df[col] = df[col].fillna("").astype(str)
    df["fallback"] = df["fallback"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
    return df[expected_columns]


def _make_sentence_id(level: str, word: str, sentence: str) -> str:
    import hashlib
    key = f"{level.upper()}|{word.lower()}|{re.sub(r'\\s+', ' ', sentence.strip().lower())}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _accepted_sentences_with_meta(level: str | None = None) -> list:
    df = load_output_dataframe()
    if df.empty:
        return []
    accepted = df[df["status"].str.lower() == "accepted"].copy()
    if level:
        accepted = accepted[accepted["level"].str.upper() == level.upper()]
    items = []
    for row in accepted.itertuples(index=False):
        info = get_word_info(row.word, row.level)
        items.append({
            "level": row.level, "word": row.word,
            "pos": info.get("pos", ""),
            "core1": info.get("core1", ""), "core2": info.get("core2", ""),
            "prompt_label": row.prompt_label, "sentence": row.sentence,
            "sentence_id": _make_sentence_id(row.level, row.word, row.sentence),
        })
    return items


def _distractor_progress(items: list) -> dict:
    df = load_distractor_dataframe()
    progress = {}
    if df.empty:
        for item in items:
            progress[item["sentence_id"]] = {"approved_count": 0, "total_candidates": 0, "complete": False}
        return progress
    approved = df[df["status"].str.lower() == "approved"]
    approved_counts = approved.groupby("sentence_id").size().to_dict()
    all_counts = df.groupby("sentence_id").size().to_dict()
    for item in items:
        sid = item["sentence_id"]
        ac = int(approved_counts.get(sid, 0))
        progress[sid] = {
            "approved_count": ac,
            "total_candidates": int(all_counts.get(sid, 0)),
            "complete": ac >= TARGET_DISTRACTORS_PER_SENTENCE,
        }
    return progress


def _save_distractor_decisions(decisions: list):
    if not decisions:
        return
    timestamp = datetime.now().isoformat()
    for d in decisions:
        d["timestamp"] = timestamp
    df_new = pd.DataFrame(decisions)
    if DISTRACTOR_FILE.exists():
        df_old = pd.read_csv(DISTRACTOR_FILE)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(DISTRACTOR_FILE, index=False)


# ── Distractor API routes ────────────────────────────────────────────

@app.route("/api/distractors/queue/<level>")
def api_distractor_queue(level):
    if level not in LEVELS:
        return jsonify(error="Invalid level"), 400
    items = _accepted_sentences_with_meta(level=level)
    progress = _distractor_progress(items)
    pending = []
    complete_count = 0
    for item in items:
        p = progress[item["sentence_id"]]
        if p["complete"]:
            complete_count += 1
        else:
            pending.append({**item, **p})
    return jsonify(
        pending=pending, total_sentences=len(items),
        complete_count=complete_count, pending_count=len(pending),
    )


@app.route("/api/distractors/generate", methods=["POST"])
def api_distractor_generate():
    data = request.json or {}
    level = (data.get("level") or "").strip().upper()
    sentence_ids = data.get("sentence_ids") or []
    if level not in LEVELS or not sentence_ids:
        return jsonify(error="Invalid level or sentence_ids"), 400
    items = _accepted_sentences_with_meta(level=level)
    by_id = {item["sentence_id"]: item for item in items}
    results = []
    for sid in sentence_ids:
        item = by_id.get(sid)
        if not item:
            continue
        outcome = distractor_logic.select_distractors(
            word=item["word"], level=item["level"], sentence=item["sentence"],
            pos=item["pos"], core1=item.get("core1", ""), core2=item.get("core2", ""),
        )
        results.append({**item, "distractors": outcome["distractors"],
                        "fallback": outcome["fallback"], "fallback_reason": outcome["fallback_reason"]})
    return jsonify(results=results)


@app.route("/api/distractors/save", methods=["POST"])
def api_distractor_save():
    data = request.json or {}
    decisions = data.get("decisions") or []
    if not decisions:
        return jsonify(error="No decisions to save"), 400
    cleaned = []
    for d in decisions:
        if not d.get("sentence_id") or not d.get("distractor_word"):
            continue
        cleaned.append({
            "level": d.get("level", ""), "word": d.get("word", ""),
            "pos": d.get("pos", ""), "prompt_label": d.get("prompt_label", ""),
            "sentence": d.get("sentence", ""), "sentence_id": d.get("sentence_id", ""),
            "distractor_word": d.get("distractor_word", ""),
            "distractor_type": d.get("distractor_type", ""),
            "status": d.get("status", "approved"),
            "fallback": bool(d.get("fallback", False)),
            "fallback_reason": d.get("fallback_reason") or "",
        })
    if not cleaned:
        return jsonify(error="No valid decisions"), 400
    _save_distractor_decisions(cleaned)
    approved = sum(1 for d in cleaned if d["status"] == "approved")
    return jsonify(ok=True, saved=len(cleaned), approved=approved)


@app.route("/api/distractors/stats")
def api_distractor_stats():
    df = load_distractor_dataframe()
    if df.empty:
        return jsonify(total_candidates=0, total_approved=0, fallback_rate_by_pos=[], coverage_by_level=[])
    total_approved = int((df["status"].str.lower() == "approved").sum())
    fallback_sentences = df[df["fallback"] == True].drop_duplicates(subset=["sentence_id"])
    all_sentences = df.drop_duplicates(subset=["sentence_id"])
    by_pos = []
    if not all_sentences.empty:
        grouped_total = all_sentences.groupby("pos").size()
        grouped_fallback = fallback_sentences.groupby("pos").size() if not fallback_sentences.empty else pd.Series(dtype=int)
        for pos in grouped_total.index:
            total = int(grouped_total.get(pos, 0))
            fb = int(grouped_fallback.get(pos, 0))
            by_pos.append({"pos": pos, "total_sentences": total, "fallback_sentences": fb,
                           "fallback_rate": round(fb / total, 3) if total else 0})
    coverage_by_level = []
    for level in LEVELS:
        level_df = df[df["level"].str.upper() == level]
        if level_df.empty:
            continue
        level_sentences = level_df.drop_duplicates(subset=["sentence_id"])
        approved_per = level_df[level_df["status"].str.lower() == "approved"].groupby("sentence_id").size()
        complete = int((approved_per >= TARGET_DISTRACTORS_PER_SENTENCE).sum())
        coverage_by_level.append({"level": level, "sentences_with_candidates": int(len(level_sentences)),
                                  "sentences_complete": complete})
    return jsonify(total_candidates=int(len(df)), total_approved=total_approved,
                   fallback_rate_by_pos=by_pos, coverage_by_level=coverage_by_level)


@app.route("/books")
def books_page():
    return render_template("books.html", levels=LEVELS)


@app.route("/api/stats")
def api_stats():
    return jsonify(build_stats_payload())


if __name__ == "__main__":
    app.run(debug=True, port=5001)
