import os
import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("Please set GROQ_API_KEY environment variable.")

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
BASE_DIR = Path(__file__).resolve().parent
LEVELS_DIR = BASE_DIR / "levels"
TRACKING_FILE = BASE_DIR / "tracking" / "used_words.json"
OUTPUT_FILE = BASE_DIR / "output" / "generated_sentences.csv"
TRACKING_FILE.parent.mkdir(exist_ok=True)
OUTPUT_FILE.parent.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# 2. HELPER: Load vocabulary (robust reading + filter junk)
# -------------------------------------------------------------------
def is_real_word(token: str) -> bool:
    """Keep only words that consist of letters (allowing hyphens) and length >= 2."""
    token = token.strip()
    # Allow only alphabetic characters and hyphens, but not a single punctuation
    if re.fullmatch(r"[a-z]+(?:[-'][a-z]+)*", token, re.IGNORECASE) and len(token) >= 2:
        return True
    return False

def find_csv_for_level(level: str) -> Path | None:
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

def load_level_words(level: str) -> set:
    csv_path = find_csv_for_level(level)
    if csv_path is None:
        print(f"Warning: no CSV found for {level} – skipping.")
        return set()
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig')
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return set()
    df.columns = df.columns.str.strip()
    word_col = None
    for col in df.columns:
        if col.strip().lower() == 'headword':
            word_col = col
            break
    if word_col is None:
        print(f"Warning: no 'headword' column in {csv_path}. Columns: {df.columns.tolist()}")
        return set()
    words = df[word_col].dropna().astype(str).str.strip().str.lower()
    words = words[words.apply(is_real_word)]
    return set(words)

def load_allowed_vocab(up_to_level: str) -> set:
    allowed = set()
    for lvl in LEVELS:
        if LEVELS.index(lvl) > LEVELS.index(up_to_level):
            break
        allowed.update(load_level_words(lvl))
    return allowed

def get_word_info(word: str, level: str) -> dict:
    csv_path = find_csv_for_level(level)
    if csv_path is None:
        return {}
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig')
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return {}

    df.columns = df.columns.str.strip()
    # Find headword column
    word_col = None
    for col in df.columns:
        if col.strip().lower() == 'headword':
            word_col = col
            break
    if word_col is None:
        print(f"Warning: no 'headword' column in {csv_path}")
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
    }

# -------------------------------------------------------------------
# 3. TRACKING FUNCTIONS
# -------------------------------------------------------------------
def load_used_words() -> dict:
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, "r") as f:
            return json.load(f)
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
    except Exception as e:
        print(f"Error loading used words: {e}")
        return sorted(level_words)

# -------------------------------------------------------------------
# 4. SENTENCE GENERATION (Groq – Qwen 2.5 7B)
# -------------------------------------------------------------------
def build_prompt(word: str, level: str, info: dict, allowed_words: set) -> str:
    pos = info.get("pos", "word")
    core1 = info.get("core1", "")
    core2 = info.get("core2", "")
    topics = "family, school, jobs, money, travel, hobbies, technology, friends"
    if core1 or core2:
        extras = [c for c in [core1, core2] if c]
        topics += ", " + ", ".join(extras)

    allowed_list = sorted(allowed_words)
    if len(allowed_list) > 3000:
        allowed_list = allowed_list[:3000]

    instruction = f"""You are an English language teacher creating example sentences for learners.
Target word: "{word}"
CEFR level: {level}
Part of speech: {pos}

Rules:
- Generate exactly 15 sentences, each using the target word naturally.
- Output ONLY the sentences, one per line. Do NOT include any numbering, bullet points, introductory text, or explanations.
- Vary the sentence patterns: cause and effect, contrast, problem and result, emotional context.
- Situations must be realistic and relatable (e.g., {topics}).
- If you cover the target word, a student should be able to guess it from context.
- CRITICAL: Use ONLY words from the allowed vocabulary list below. You may use common inflected forms (plural, past tense, -ing, etc.). If a needed word is missing, rephrase.

Allowed vocabulary: {', '.join(allowed_list)}"""

    return instruction

def generate_sentences(word: str, level: str, info: dict, allowed_words: set) -> list:
    client = Groq(api_key=API_KEY)
    prompt = build_prompt(word, level, info, allowed_words)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",                # Correct model ID
        messages=[
            {"role": "system", "content": "You are a helpful English teacher."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048,
        top_p=0.95,
    )
    raw = response.choices[0].message.content.strip()
    sentences = []
    for line in raw.split("\n"):
        line = line.strip()
        # Skip empty lines, and lines that are clearly meta like "Here are..."
        if not line or line.lower().startswith("here are") or line.lower().startswith("sure"):
            continue
        # Remove any leading numbering like "1." or "1)" or "- "
        line = re.sub(r'^\d+[\.\)\-]?\s*', '', line).strip()
        # Skip if line is still empty or very short (less than 15 chars)
        if len(line) < 15:
            continue
        sentences.append(line)
    
    return sentences[:15]

# -------------------------------------------------------------------
# 5. VALIDATION (lemma check against allowed words)
# -------------------------------------------------------------------
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

def validate_sentence(sentence: str, allowed: set) -> list:
    tokens = word_tokenize(sentence)
    unknown = []
    for word in tokens:
                # Skip punctuation, pure numbers, and very short tokens
        if word in {'.', ',', '!', '?', ';', ':', '\'', '\"', '-', '–', '—', '(', ')', '...'}:
            continue
        if re.fullmatch(r'\d+', word):  # skip pure numbers
            continue
        clean = word.strip("'\".,;:!?()[]")
        if not clean or len(clean) <= 1:
            
            continue
        # try all possible lemmas (noun, verb, adj, adv) – if none is in allowed, flag it
        lemmas = set()
        for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]:
            lemmas.add(lemmatizer.lemmatize(clean.lower(), pos=pos))
        if not lemmas.intersection(allowed):
            unknown.append(f"{word} (lemma candidates: {', '.join(lemmas)})")
    return unknown
# -------------------------------------------------------------------
# 6. SAVE ACCEPTED SENTENCES
# -------------------------------------------------------------------
def save_accepted_sentences(level: str, word: str, accepted: list):
    df_new = pd.DataFrame({
        "level": [level] * len(accepted),
        "word": [word] * len(accepted),
        "sentence": accepted,
        "status": "accepted",
        "timestamp": datetime.now().isoformat()
    })
    if OUTPUT_FILE.exists():
        df_old = pd.read_csv(OUTPUT_FILE)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(accepted)} sentences to {OUTPUT_FILE}")

# -------------------------------------------------------------------
# 7. MAIN INTERACTIVE LOOP
# -------------------------------------------------------------------
def main():
    print("===== English Example Sentence Generator =====\n")
    print("Available levels:", ", ".join(LEVELS))
    while True:
        level = input("Choose a level (e.g., B1): ").strip().upper()
        if level in LEVELS:
            break
        print("Invalid level. Try again.")

    allowed = load_allowed_vocab(level)
    if not allowed:
        print(f"No vocabulary found up to {level}. Check your CSV files.")
        return

    unused = get_unused_words(level)
    if not unused:
        print("All words for this level have been used already. Reset tracking if needed.")
        return

    print(f"\nUnused words in {level} ({len(unused)} total):")
    for i, w in enumerate(unused[:20]):
        print(f"  {i+1}. {w}")
    if len(unused) > 20:
        print(f"  ... and {len(unused)-20} more")

    while True:
        choice = input("\nType a word from the list (or 'q' to quit): ").strip().lower()
        if choice == 'q':
            return
        if choice in unused:
            word = choice
            break
        print("Word not in unused list. Try again.")

    info = get_word_info(word, level)
    if not info:
        print(f"Could not find info for '{word}' in {level}/words.csv. Continuing anyway.")
        info = {"pos": "unknown", "core1": "", "core2": ""}

    print("\nGenerating 15 sentences... (this may take a moment)")
    try:
        sentences = generate_sentences(word, level, info, allowed)
    except Exception as e:
        print(f"Error during generation: {e}")
        return

    if not sentences:
        print("No sentences received. Try again.")
        return

    print("\n--- Generated Sentences ---")
    accepted = []
    for idx, sent in enumerate(sentences, 1):
        unknown = validate_sentence(sent, allowed)
        flag = "⚠️" if unknown else "✅"
        print(f"\n{idx}. {flag} {sent}")
        if unknown:
            print("   Unknown words:", ", ".join(unknown[:5]))

        while True:
            action = input("   Accept (a) / Reject (r) / Edit (e)? ").strip().lower()
            if action == 'a':
                accepted.append(sent)
                break
            elif action == 'r':
                break
            elif action == 'e':
                new_sent = input("   Enter your own sentence: ").strip()
                if new_sent:
                    accepted.append(new_sent)
                break
            else:
                print("   Please enter a, r, or e.")

    if accepted:
        save_accepted_sentences(level, word, accepted)
    mark_word_used(level, word)
    print(f"\nDone! Marked '{word}' as used. Accepted {len(accepted)} sentences.")

if __name__ == "__main__":
    main()