"""
Disease Predictor — DDXPlus Dataset
=====================================
Uses the DDXPlus dataset (NeurIPS 2022, Mila / McGill University):
  • 1.3 million patients  •  49 diseases  •  223 symptoms/antecedents
  • Includes differential diagnosis (ranked list of possible conditions)
  • AGE and SEX as additional features

Dataset download (free, CC-BY licence)
---------------------------------------
  https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585

After downloading, unzip and place these files in the same folder as this script:
  release_evidences.json
  release_conditions.json
  release_train_patients.zip   (or .csv)
  release_validate_patients.zip
  release_test_patients.zip    (or .csv)

Supported NLP backends (optional — set ONE env var)
-----------------------------------------------------
  ANTHROPIC_API_KEY  →  Claude     https://console.anthropic.com/settings/keys
  OPENAI_API_KEY     →  GPT-4o     https://platform.openai.com/api-keys
  GEMINI_API_KEY     →  Gemini     https://aistudio.google.com/app/apikey
  (none)             →  offline fuzzy matching

Usage
------
  pip install scikit-learn pandas numpy rapidfuzz anthropic
  python disease_predictor_ddxplus.py
"""

import os, sys, io, json, csv, zipfile, textwrap, datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import process as fz_process, fuzz

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
DATA_DIR   = "."          # folder containing the DDXPlus files
SAMPLE_DIR = "/home/claude/ddxplus_sample"  # fallback sample for testing

EVIDENCES_FILE   = "release_evidences.json"
CONDITIONS_FILE  = "release_conditions.json"
TRAIN_FILE       = "release_train_patients"      # .zip or .csv
VALIDATE_FILE    = "release_validate_patients"
TEST_FILE        = "release_test_patients"

UNKNOWN_SYMPTOMS_FILE   = "unknown_symptoms.json"
FEEDBACK_LOG_FILE       = "feedback_log.json"
MIN_FEEDBACK_TO_RETRAIN = 10

MAX_TRAIN_ROWS = 50_000    # 50k rows ≈ same accuracy, ~4× faster; set None for full 1.3M
FUZZY_THRESHOLD = 70
FUZZY_TOP_N     = 3

# ── API detection ─────────────────────────────────────────
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "")

if ANTHROPIC_KEY:   NLP_BACKEND = "anthropic"
elif OPENAI_KEY:    NLP_BACKEND = "openai"
elif GEMINI_KEY:    NLP_BACKEND = "gemini"
else:               NLP_BACKEND = "fuzzy"

ANTHROPIC_MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
OPENAI_MODELS    = ["gpt-4o-mini", "gpt-3.5-turbo"]
GEMINI_MODELS    = ["gemini-2.0-flash", "gemini-1.5-flash"]

NLP_SYSTEM_PROMPT = """You are a medical symptom extraction assistant.
Extract every symptom or antecedent the patient mentions.

Rules:
- Include casual language ("feel terrible" → fatigue, "joints ache" → joint pain).
- Detect NEGATED symptoms ("no cough") — list separately.
- Detect UNCERTAIN symptoms ("maybe a rash") — list separately.
- Normalise to short plain-English phrases.
- Exclude diagnoses (flu, COVID) — only symptoms/signs.

Respond ONLY with valid JSON (no markdown):
{"symptoms": [...], "negated": [...], "uncertain": [...]}"""


# ─────────────────────────────────────────────────────────
# STEP 1 — FIND DATA FILES
# ─────────────────────────────────────────────────────────

def _find_file(stem: str, dirs: list[str]) -> str | None:
    """Look for stem.csv, stem.zip in each directory."""
    for d in dirs:
        for ext in (".csv", ".zip"):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return p
    return None


def _find_json(name: str, dirs: list[str]) -> str | None:
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def locate_files() -> dict:
    search = [DATA_DIR, SAMPLE_DIR]
    files = {
        "evidences":  _find_json(EVIDENCES_FILE,  search),
        "conditions": _find_json(CONDITIONS_FILE, search),
        "train":      _find_file(TRAIN_FILE,      search),
        "validate":   _find_file(VALIDATE_FILE,   search),
        "test":       _find_file(TEST_FILE,        search),
    }
    missing = [k for k, v in files.items() if v is None and k != "validate"]
    if missing:
        print("\n" + "=" * 62)
        print("  DDXPlus dataset files not found.")
        print("  Download from: https://figshare.com/articles/dataset/")
        print("                 DDXPlus_Dataset_English_/22687585")
        print(f"\n  Place these files in: {os.path.abspath(DATA_DIR)}")
        for f in [EVIDENCES_FILE, CONDITIONS_FILE,
                  TRAIN_FILE+".zip", TEST_FILE+".zip"]:
            print(f"    • {f}")
        print("=" * 62 + "\n")
        sys.exit(1)

    using_sample = SAMPLE_DIR in (files["train"] or "")
    if using_sample:
        print("  ⚠  Using built-in sample data (200 patients, 5 diseases).")
        print("     Download the full DDXPlus dataset for production use.")
    else:
        print(f"  ✔  Found DDXPlus files in {DATA_DIR}")
    return files


# ─────────────────────────────────────────────────────────
# STEP 2 — LOAD METADATA
# ─────────────────────────────────────────────────────────

def load_metadata(files: dict) -> tuple[dict, dict]:
    """
    Returns:
      evidences  : {code → {question_en, data_type, ...}}
      conditions : {name → {icd10-id, symptoms, antecedents, ...}}
    """
    evidences  = json.load(open(files["evidences"],  encoding="utf-8"))
    conditions = json.load(open(files["conditions"], encoding="utf-8"))

    # Build readable name lookup:  code → English question text
    for code, info in evidences.items():
        q = info.get("question_en") or info.get("question", code)
        info["readable"] = q.lower().rstrip("?").strip()

    print(f"  Evidences (symptoms + antecedents): {len(evidences)}")
    print(f"  Conditions (diseases)              : {len(conditions)}")
    return evidences, conditions


# ─────────────────────────────────────────────────────────
# STEP 3 — LOAD PATIENT CSV (from .zip or .csv)
# ─────────────────────────────────────────────────────────

def _open_csv(path: str):
    """
    Return a csv.DictReader from a plain .csv file or a .zip archive.

    DDXPlus zips may contain entries with no extension, a non-.csv name,
    or a folder prefix — so we search by extension first, then fall back
    to the first non-directory entry found.
    """
    if path.endswith(".zip"):
        z    = zipfile.ZipFile(path)
        names = z.namelist()

        # 1. Prefer a .csv entry
        csvname = next((n for n in names if n.lower().endswith(".csv")), None)

        # 2. Fall back to first non-directory entry with any extension
        if csvname is None:
            csvname = next(
                (n for n in names if not n.endswith("/") and "." in n), None
            )

        # 3. Last resort — first non-directory entry regardless
        if csvname is None:
            csvname = next((n for n in names if not n.endswith("/")), None)

        if csvname is None:
            raise ValueError(
                f"No readable file found inside {path}.\n"
                f"ZIP contents: {names}"
            )

        print(f"  Reading from zip: {csvname}")
        raw  = z.read(csvname)
        text = raw.decode("utf-8-sig")   # strips BOM if present
        return csv.DictReader(io.StringIO(text))

    return csv.DictReader(open(path, encoding="utf-8-sig"))


def parse_evidence_vector(evidences_json: str,
                          all_codes: list[str],
                          code_index: dict) -> list[int]:
    """
    Convert a patient EVIDENCES JSON string into a binary feature vector.

    DDXPlus encodes evidences as:  ["E_1_@_Y", "E_7_@_moderate", ...]
      Binary    : E_1_@_Y  → column E_1 = 1
      Categorical: E_7_@_moderate → column E_7_moderate = 1
    """
    vec = [0] * len(all_codes)
    try:
        ev_list = json.loads(evidences_json)
    except Exception:
        return vec

    for ev in ev_list:
        if "_@_" in ev:
            code, value = ev.split("_@_", 1)
            # For binary just set the base code
            if code in code_index:
                vec[code_index[code]] = 1
            # For categorical/multi also set code_value column if it exists
            col_val = f"{code}_{value}"
            if col_val in code_index:
                vec[code_index[col_val]] = 1
        else:
            if ev in code_index:
                vec[code_index[ev]] = 1
    return vec


def load_patients(path: str,
                  all_codes: list[str],
                  code_index: dict,
                  le: LabelEncoder,
                  max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a patient CSV into (X, y) arrays."""
    X_rows, y_rows = [], []
    known = set(le.classes_)

    for i, row in enumerate(_open_csv(path)):
        if max_rows and i >= max_rows:
            break
        label = row.get("PATHOLOGY", "").strip()
        if label not in known:
            continue

        vec = parse_evidence_vector(row["EVIDENCES"], all_codes, code_index)

        # Add AGE and SEX as extra features
        try:
            age = float(row.get("AGE", 40)) / 100.0
        except ValueError:
            age = 0.4
        sex = 1.0 if row.get("SEX", "M") == "M" else 0.0
        vec = vec + [age, sex]

        X_rows.append(vec)
        y_rows.append(label)

    X = np.array(X_rows, dtype=np.float32)
    y = le.transform(y_rows)
    return X, y


def build_feature_schema(evidences: dict) -> tuple[list[str], dict]:
    """
    Build the ordered list of feature column names and a fast index dict.
    Binary evidences → one column per code
    Categorical/multi → one column per (code, value) pair  +  one base column
    Returns (all_codes, code_index)
    Note: AGE and SEX are appended as the last two columns in load_patients()
    """
    all_codes = []
    seen = set()

    def add(col):
        if col not in seen:
            all_codes.append(col)
            seen.add(col)

    for code, info in evidences.items():
        add(code)
        if info.get("data_type") in ("C", "M"):
            for val in (info.get("possible-values") or []):
                add(f"{code}_{val}")

    code_index = {c: i for i, c in enumerate(all_codes)}
    return all_codes, code_index


# ─────────────────────────────────────────────────────────
# STEP 4 — TRAIN
# ─────────────────────────────────────────────────────────

def train_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Speed-optimised classifier set for large datasets.

    Random Forest          — 100 trees, parallel, depth-capped
    HistGradientBoosting   — replaces slow GradientBoostingClassifier;
                             uses histogram binning, 10-50× faster on
                             large data, supports native NaN handling
    Decision Tree          — depth-capped for speed + regularisation
    """
    import time
    from sklearn.ensemble import HistGradientBoostingClassifier

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,       # 200 → 100: halves training time, ~same accuracy
            max_features="sqrt",
            max_depth=30,           # cap prevents overfitting + speeds up
            min_samples_leaf=2,
            n_jobs=-1,              # use all CPU cores
            random_state=42,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100,           # equivalent to n_estimators
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            # 10-50× faster than GradientBoostingClassifier on large data
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,           # shallower = faster + less overfit
            min_samples_leaf=5,
            random_state=42,
        ),
    }

    n_samples = X_train.shape[0]
    print(f"  Dataset: {n_samples:,} samples × {X_train.shape[1]} features")
    print()

    for name, model in models.items():
        t0 = time.time()
        print(f"  ⏳ Training {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s")

    return models


# ─────────────────────────────────────────────────────────
# STEP 5 — EVALUATE
# ─────────────────────────────────────────────────────────

def evaluate_models(models: dict, X_test: np.ndarray,
                    y_test: np.ndarray, le: LabelEncoder):
    print("\n" + "=" * 66)
    print("MODEL EVALUATION  (held-out test set)")
    print("=" * 66)
    for name, model in models.items():
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        print(f"\n{'─'*66}\n  {name:<26} Accuracy: {acc:.1%}\n{'─'*66}")
        print(classification_report(y_test, preds,
                                    target_names=le.classes_, zero_division=0))


# ─────────────────────────────────────────────────────────
# STEP 6 — NLP SYMPTOM RESOLVER
# ─────────────────────────────────────────────────────────

def _is_rate_limit(exc) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ("429", "rate limit", "too many requests", "quota"))

def _is_not_found(exc) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ("404", "not found", "does not exist", "invalid model"))

def _with_retry(fn, models_list: list, backend_name: str) -> dict:
    for model in models_list:
        try:
            return fn(model)
        except Exception as exc:
            if _is_not_found(exc):
                print(f"  ⚠  [{backend_name}] 404 — '{model}' not found.")
                print(f"  ℹ  Switching to offline fuzzy matching.")
                return {}
            elif _is_rate_limit(exc):
                print(f"  ⚠  [{backend_name}] 429 — rate limited.")
                print(f"  ℹ  Switching to offline fuzzy matching.")
                return {}
            else:
                print(f"  ⚠  [{backend_name}] '{model}' error: {exc} — trying next model")
    print(f"  ✗  [{backend_name}] all models exhausted — falling back to fuzzy matching")
    return {}

def _call_anthropic(text: str, model: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    msg = client.messages.create(
        model=model, max_tokens=400, system=NLP_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}]
    )
    return json.loads(msg.content[0].text.strip())

def _call_openai(text: str, model: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=OPENAI_KEY)
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=400,
        messages=[{"role": "system", "content": NLP_SYSTEM_PROMPT},
                  {"role": "user",   "content": text}]
    )
    return json.loads(resp.choices[0].message.content.strip())

def _call_gemini(text: str, model: str) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_KEY)
    resp = client.models.generate_content(
        model=model, contents=text,
        config=types.GenerateContentConfig(
            system_instruction=NLP_SYSTEM_PROMPT,
            temperature=0, max_output_tokens=400,
            response_mime_type="application/json",
        )
    )
    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw.strip())

def extract_symptoms_via_llm(text: str) -> dict:
    if NLP_BACKEND == "anthropic":
        return _with_retry(lambda m: _call_anthropic(text, m), ANTHROPIC_MODELS, "Anthropic")
    elif NLP_BACKEND == "openai":
        return _with_retry(lambda m: _call_openai(text, m),    OPENAI_MODELS,    "OpenAI")
    elif NLP_BACKEND == "gemini":
        return _with_retry(lambda m: _call_gemini(text, m),    GEMINI_MODELS,    "Gemini")
    return {}


# ─────────────────────────────────────────────────────────
# STEP 7 — FUZZY SYMPTOM → EVIDENCE CODE MAPPER
# ─────────────────────────────────────────────────────────

def build_symptom_lookup(evidences: dict) -> tuple[dict, list]:
    """
    Build a lookup: readable question text → evidence code
    e.g. "do you have a fever" → "E_1"
    """
    readable_to_code = {}
    readable_phrases = []
    for code, info in evidences.items():
        phrase = info["readable"]
        readable_to_code[phrase] = code
        readable_phrases.append(phrase)
    return readable_to_code, readable_phrases


def fuzzy_match_symptom(phrase: str,
                        readable_to_code: dict,
                        readable_phrases: list,
                        evidences: dict,
                        interactive: bool = True) -> str | None:
    """Match a natural-language phrase to the best evidence code.
    
    Args:
        interactive: If True, prompt user for ambiguous matches (CLI mode).
                     If False, auto-select best match (API mode).
    """
    phrase = phrase.lower().strip()

    # Exact match
    if phrase in readable_to_code:
        return readable_to_code[phrase]

    # Fuzzy match
    hits = fz_process.extract(phrase, readable_phrases,
                               scorer=fuzz.WRatio, limit=FUZZY_TOP_N)
    best_str, best_score, _ = hits[0]
    if best_score < FUZZY_THRESHOLD:
        return None

    # Ambiguous?
    close = [(h[0], h[1]) for h in hits if h[1] >= hits[0][1] - 5]
    if len(close) > 1:
        if interactive:
            # CLI mode: prompt user
            print(f"\n  ❓ '{phrase}' is ambiguous — did you mean:")
            for i, (opt, sc) in enumerate(close, 1):
                print(f"      [{i}] {opt}  (score {sc:.0f})")
            print("      [s] Skip")
            choice = input("     Your choice: ").strip().lower()
            if choice.isdigit() and 1 <= int(choice) <= len(close):
                return readable_to_code[close[int(choice)-1][0]]
            return None
        else:
            # API mode: auto-select best match
            best_str = close[0][0]
            best_score = close[0][1]

    code = readable_to_code[best_str]
    if interactive and phrase != best_str:
        print(f"  ✔  '{phrase}'  →  '{best_str}'  (score {best_score:.0f})")
    return code


def resolve_symptoms(user_text: str,
                     evidences: dict,
                     readable_to_code: dict,
                     readable_phrases: list,
                     all_codes: list,
                     code_index: dict,
                     age: float = 40.0,
                     sex: str = "M",
                     interactive: bool = True) -> tuple[np.ndarray, list, list]:
    """
    Parse free-text input → feature vector (including AGE and SEX).
    Returns (vector, matched_readable_phrases, negated_phrases)
    
    Args:
        interactive: If True, prompts user for ambiguous matches (CLI mode).
                     If False, auto-selects best matches (API mode).
    """
    extraction = extract_symptoms_via_llm(user_text)
    unknowns = []

    if extraction:
        raw_symptoms  = extraction.get("symptoms",  [])
        raw_negated   = extraction.get("negated",   [])
        raw_uncertain = extraction.get("uncertain", [])
        if interactive:
            print(f"\n  LLM extracted → symptoms : {raw_symptoms}")
            if raw_negated:   print(f"                  negated  : {raw_negated}")
            if raw_uncertain: print(f"                  uncertain: {raw_uncertain}")
            print()
        symptom_phrases = raw_symptoms + raw_uncertain
        negated_phrases = raw_negated
    else:
        # Fallback: treat whole input as comma-separated
        if interactive:
            print("  ℹ  Using offline fuzzy matching.")
        symptom_phrases = [t.strip() for t in user_text.split(",") if t.strip()]
        negated_phrases = []

    # Build feature vector
    vec = [0] * len(all_codes)
    matched = []

    for phrase in symptom_phrases:
        code = fuzzy_match_symptom(
            phrase, readable_to_code, readable_phrases, evidences,
            interactive=interactive
        )
        if code and code in code_index:
            vec[code_index[code]] = 1
            matched.append(evidences[code]["readable"])
        elif code is None:
            unknowns.append(phrase)
            if interactive:
                print(f"  ✗  '{phrase}'  →  no match found — logging")

    if unknowns:
        log_unknown_symptoms(unknowns)

    # AGE and SEX (appended as last two features, matching load_patients)
    vec_with_demo = vec + [age / 100.0, 1.0 if sex == "M" else 0.0]
    return np.array([vec_with_demo], dtype=np.float32), matched, negated_phrases


# ─────────────────────────────────────────────────────────
# STEP 8 — LEARNING (unknown log + feedback + retrain)
# ─────────────────────────────────────────────────────────

def _load_json(path, default):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    return default

def _save_json(path, data):
    json.dump(data, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def log_unknown_symptoms(phrases: list):
    if not phrases: return
    store = _load_json(UNKNOWN_SYMPTOMS_FILE, {})
    today = datetime.date.today().isoformat()
    for p in phrases:
        key = p.lower().strip()
        if key in store:
            store[key]["count"] += 1
            store[key]["last_seen"] = today
        else:
            store[key] = {"count": 1, "first_seen": today, "last_seen": today}
    _save_json(UNKNOWN_SYMPTOMS_FILE, store)
    print(f"  📝 Logged {len(phrases)} unknown symptom(s) → {UNKNOWN_SYMPTOMS_FILE}")

def show_unknown_symptoms():
    store = _load_json(UNKNOWN_SYMPTOMS_FILE, {})
    if not store:
        print("\n  No unknown symptoms logged yet.")
        return
    print(f"\n  Unknown symptoms — {len(store)} unique phrases:")
    print(f"  {'Phrase':<42} {'Seen':>5}  First        Last")
    print(f"  {'─'*42} {'─'*5}  {'─'*10}   {'─'*10}")
    for p, info in sorted(store.items(), key=lambda x: -x[1]["count"]):
        print(f"  {p:<42} {info['count']:>5}  "
              f"{info['first_seen']}   {info['last_seen']}")

def ask_feedback(matched: list, predicted_disease: str):
    print()
    ans = input("  💬 Was this prediction correct? [y/n/skip]: ").strip().lower()
    if ans not in ("y", "n"): return
    if ans == "n":
        label = input("  Correct disease (or Enter to skip): ").strip()
        if not label: return
    else:
        label = predicted_disease
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "symptoms": matched, "label": label, "correct": ans == "y",
    }
    log = _load_json(FEEDBACK_LOG_FILE, [])
    log.append(record)
    _save_json(FEEDBACK_LOG_FILE, log)
    print(f"  ✔  Feedback saved ({len(log)} total entries)")
    if len(log) >= MIN_FEEDBACK_TO_RETRAIN:
        print(f"  ℹ  {len(log)} entries collected — use [4] to retrain.")

def show_feedback_summary():
    log = _load_json(FEEDBACK_LOG_FILE, [])
    if not log:
        print("\n  No feedback collected yet.")
        return
    correct  = sum(1 for r in log if r.get("correct"))
    diseases = {}
    for r in log:
        diseases[r["label"]] = diseases.get(r["label"], 0) + 1
    print(f"\n  Feedback — {len(log)} entries  |  "
          f"correct: {correct}  |  corrected: {len(log)-correct}")
    for d, c in sorted(diseases.items(), key=lambda x: -x[1]):
        print(f"    {d:<44} {c:>3}  {'█'*c}")


# ─────────────────────────────────────────────────────────
# STEP 9 — PREDICT HELPERS
# ─────────────────────────────────────────────────────────

def top_predictions(model, vec: np.ndarray,
                    le: LabelEncoder, top_n: int = 5):
    if hasattr(model, "predict_proba"):
        probs   = model.predict_proba(vec)[0]
        indices = np.argsort(probs)[::-1][:top_n]
        return [(le.classes_[i], probs[i]) for i in indices]
    pred = model.predict(vec)[0]
    return [(le.classes_[pred], 1.0)]


# ─────────────────────────────────────────────────────────
# STEP 10 — INTERACTIVE CLI
# ─────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         DISEASE PREDICTOR  —  DDXPlus Edition                   ║
║   Dataset: DDXPlus (NeurIPS 2022)  •  49 diseases               ║
║   NLP    : {backend:<46}║
╚══════════════════════════════════════════════════════════════════╝"""

def print_symptom_list(evidences: dict):
    print(f"\n  Available symptoms / antecedents ({len(evidences)} total):\n")
    items = [(code, info["readable"]) for code, info in evidences.items()]
    col_w = 48
    for i in range(0, len(items), 2):
        row = items[i : i + 2]
        print("".join(f"  {code:<6}  {label:<{col_w}}" for code, label in row))


def get_demographics() -> tuple[float, str]:
    """Ask for age and sex to improve prediction accuracy."""
    print("\n  For better accuracy, please enter your demographics:")
    try:
        age_str = input("  Age (press Enter to skip): ").strip()
        age = float(age_str) if age_str else 40.0
        age = max(1.0, min(120.0, age))
    except ValueError:
        age = 40.0
    sex_str = input("  Sex — M or F (press Enter to skip): ").strip().upper()
    sex = sex_str if sex_str in ("M", "F") else "M"
    return age, sex


def run_prediction(models, evidences, readable_to_code, readable_phrases,
                   all_codes, code_index, le, compare=False, interactive=True,
                   user_text: str = None, age: float = 40.0, sex: str = "M"):
    """Run a single prediction.
    
    Args:
        interactive: If True, uses CLI prompts. If False, accepts parameters directly.
        user_text: Required when interactive=False (API mode).
        age: Patient age (only used when interactive=False).
        sex: Patient sex (only used when interactive=False).
    """
    if interactive:
        backend_label = {
            "anthropic": "Claude (Anthropic API)",
            "openai":    "GPT-4o-mini (OpenAI API)",
            "gemini":    "Gemini 2.0 Flash (Google API)",
            "fuzzy":     "Offline fuzzy matching",
        }[NLP_BACKEND]

        if NLP_BACKEND != "fuzzy":
            print("\n  Describe your symptoms in plain English:")
            print('  e.g. "I\'ve had a high fever for two days, my joints ache')
            print('        terribly and I have a rash on my arms"')
        else:
            print("\n  Enter symptoms as comma-separated phrases:")
            print("  e.g. fever, joint pain, skin rash, fatigue")
        print()

        raw = input("Symptoms: ").strip()
        if not raw:
            print("  ⚠  No input.")
            return

        age, sex = get_demographics()
        user_text = raw
    else:
        # API mode: user_text must be provided
        if not user_text:
            print("  ⚠  No symptoms provided.")
            return

    vec, matched, negated = resolve_symptoms(
        user_text, evidences, readable_to_code, readable_phrases,
        all_codes, code_index, age=age, sex=sex, interactive=interactive
    )

    if not matched:
        if interactive:
            print("\n  ✗  No symptoms matched. Try [2] to browse available symptoms.")
        return

    if interactive:
        print(f"\n  ✔  Symptoms matched : {', '.join(matched)}")
        if negated: print(f"  ✗  Negated          : {', '.join(negated)}")
        print(f"  👤 Demographics    : age={int(age)}, sex={sex}")

    if not compare:
        results = top_predictions(models["Random Forest"], vec, le, top_n=5)
        if interactive:
            print(f"\n  ┌─ Random Forest — Top Predictions {'─'*27}┐")
            for rank, (disease, prob) in enumerate(results, 1):
                bar = "█" * int(prob * 30)
                pad = " " * (30 - len(bar))
                print(f"  │  {rank}. {disease:<44} {prob:5.1%}  {bar}{pad} │")
            print(f"  └{'─'*66}┘")
            # Feedback
            ask_feedback(matched, results[0][0])
    else:
        if interactive:
            print()
            for name, model in models.items():
                results = top_predictions(model, vec, le, top_n=3)
                print(f"  [{name}]")
                for rank, (disease, prob) in enumerate(results, 1):
                    bar = "█" * int(prob * 25)
                    print(f"    {rank}. {disease:<46} {prob:5.1%}  {bar}")
                print()
            ask_feedback(matched, top_predictions(models.get("Random Forest", list(models.values())[0]), vec, le, top_n=1)[0][0])

    if interactive:
        print("  ⚕  For educational purposes only. Consult a healthcare professional.")


def interactive_session(models, evidences, readable_to_code, readable_phrases,
                        all_codes, code_index, le):
    backend_desc = {
        "anthropic": "Claude API (NLP mode — full sentence input)",
        "openai":    "OpenAI API (NLP mode — full sentence input)",
        "gemini":    "Gemini API (NLP mode — full sentence input)",
        "fuzzy":     "Offline fallback (comma-separated input)",
    }[NLP_BACKEND]
    print(BANNER.format(backend=backend_desc))

    while True:
        print("\nOptions:")
        print("  [1] Predict disease from symptoms")
        print("  [2] Browse all symptoms / antecedents")
        print("  [3] Compare all three classifiers")
        print("  [4] View unknown symptom log")
        print("  [5] View feedback summary")
        print("  [q] Quit")
        choice = input("\nChoice: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("\nGoodbye! Always consult a licensed physician.\n")
            break
        elif choice == "1":
            run_prediction(models, evidences, readable_to_code, readable_phrases,
                           all_codes, code_index, le, compare=False, interactive=True)
        elif choice == "2":
            print_symptom_list(evidences)
        elif choice == "3":
            run_prediction(models, evidences, readable_to_code, readable_phrases,
                           all_codes, code_index, le, compare=True, interactive=True)
        elif choice == "4":
            show_unknown_symptoms()
        elif choice == "5":
            show_feedback_summary()
        else:
            print("  Invalid choice.")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    backend_label = {
        "anthropic": "Anthropic (Claude)",
        "openai":    "OpenAI (GPT-4o-mini)",
        "gemini":    "Google (Gemini 2.0 Flash)",
        "fuzzy":     "offline fuzzy matching",
    }[NLP_BACKEND]
    print(f"\n🔬 Disease Predictor — DDXPlus Edition | NLP: {backend_label}\n")
    if NLP_BACKEND == "fuzzy":
        print("  ℹ  No API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY,")
        print("     or GEMINI_API_KEY for full sentence NLP parsing.\n")

    print("[1/5] Locating dataset files …")
    files = locate_files()

    print("\n[2/5] Loading metadata …")
    evidences, conditions = load_metadata(files)

    print("\n[3/5] Building feature schema …")
    all_codes, code_index = build_feature_schema(evidences)
    # +2 for AGE and SEX
    print(f"  Feature columns: {len(all_codes)} evidence codes + 2 demographics "
          f"= {len(all_codes)+2} total features")

    # Build label encoder from conditions (all known diseases)
    le = LabelEncoder()
    le.fit(list(conditions.keys()))

    print("\n[4/5] Loading patients …")
    cap = MAX_TRAIN_ROWS
    X_train, y_train = load_patients(files["train"],    all_codes, code_index, le, cap)
    X_test,  y_test  = load_patients(files["test"],     all_codes, code_index, le)
    print(f"  Train : {len(X_train):>7,} patients")
    print(f"  Test  : {len(X_test):>7,} patients")
    print(f"  Shape : {X_train.shape[1]} features")

    print("\n[5/5] Training classifiers …")
    models = train_models(X_train, y_train)

    evaluate_models(models, X_test, y_test, le)

    # Build NLP resolver lookup
    readable_to_code, readable_phrases = build_symptom_lookup(evidences)

    print("\n✅ Ready.\n")
    interactive_session(models, evidences, readable_to_code, readable_phrases,
                        all_codes, code_index, le)


if __name__ == "__main__":
    main()