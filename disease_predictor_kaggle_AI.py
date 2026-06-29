"""
Disease Predictor — Kaggle/Kaushil268 Dataset
with NLP Symptom Intake (LLM extraction + fuzzy/synonym fallback)
==================================================================
Dataset : https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
132 binary symptom features  →  41 disease prognoses

NLP intake pipeline
-------------------
  Primary   : LLM API (Claude / OpenAI / Gemini) — understands full sentences,
              handles negation, uncertainty, and conversational phrasing.
  Fallback  : synonym map + rapidfuzz — works offline, no API key needed.

Supported NLP backends (set ONE environment variable)
------------------------------------------------------
  ANTHROPIC_API_KEY   → Claude Sonnet    https://console.anthropic.com/settings/keys
  OPENAI_API_KEY      → GPT-4o-mini      https://platform.openai.com/api-keys
  GEMINI_API_KEY      → Gemini 1.5 Flash https://aistudio.google.com/app/apikey
  (none)              → offline fuzzy matching (no API needed)

Usage
-----
  pip install scikit-learn pandas numpy rapidfuzz anthropic   # Claude
  pip install scikit-learn pandas numpy rapidfuzz openai      # OpenAI
  pip install scikit-learn pandas numpy rapidfuzz google-genai          # Gemini

  export ANTHROPIC_API_KEY=sk-ant-...
  python disease_predictor_kaggle.py
"""

import os, sys, json, textwrap, urllib.request, datetime
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
TRAIN_URL       = ("https://raw.githubusercontent.com/fabiannoda/"
                   "disease_prediction/master/Training.csv")
TEST_URL        = ("https://raw.githubusercontent.com/fabiannoda/"
                   "disease_prediction/master/Testing.csv")
TRAIN_CSV       = "training.csv"
TEST_CSV        = "testing.csv"
FUZZY_THRESHOLD = 72
FUZZY_TOP_N     = 3

# ── Learning / feedback files ────────────────────────────
UNKNOWN_SYMPTOMS_FILE = "unknown_symptoms.json"  # unmatched symptom phrases
FEEDBACK_LOG_FILE     = "feedback_log.json"      # user-confirmed predictions
MIN_FEEDBACK_TO_RETRAIN = 10                      # minimum feedback rows before retraining

# ── API detection ─────────────────────────────────────────
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "")

if ANTHROPIC_KEY:
    NLP_BACKEND = "anthropic"
elif OPENAI_KEY:
    NLP_BACKEND = "openai"
elif GEMINI_KEY:
    NLP_BACKEND = "gemini"
else:
    NLP_BACKEND = "fuzzy"          # offline fallback

# ─────────────────────────────────────────────────────────
# SYNONYM MAP  (offline fallback — common casual terms)
# ─────────────────────────────────────────────────────────
SYNONYM_MAP = {
    "loose motions":"diarrhoea","loose stool":"diarrhoea","diarrhea":"diarrhoea",
    "throwing up":"vomiting","puking":"vomiting",
    "stomach ache":"stomach_pain","tummy ache":"stomach_pain","belly ache":"belly_pain",
    "indigestion":"indigestion","gas":"passage_of_gases","bloating":"distention_of_abdomen",
    "shortness of breath":"breathlessness","short of breath":"breathlessness",
    "difficulty breathing":"breathlessness","cant breathe":"breathlessness",
    "blocked nose":"congestion","stuffy nose":"congestion",
    "dizzy":"dizziness","feel dizzy":"dizziness","lightheaded":"dizziness",
    "off balance":"loss_of_balance","headache":"headache","head pain":"headache",
    "spinning":"spinning_movements","vertigo":"spinning_movements",
    "neck stiffness":"stiff_neck","stiff neck":"stiff_neck",
    "blurry vision":"blurred_and_distorted_vision",
    "blurred vision":"blurred_and_distorted_vision",
    "itchy skin":"itching","skin itch":"itching","itch":"itching",
    "rash":"skin_rash","skin rash":"skin_rash",
    "yellow skin":"yellowish_skin","yellow eyes":"yellowing_of_eyes","jaundice":"yellowish_skin",
    "peeling skin":"skin_peeling","blisters":"blister","pimples":"pus_filled_pimples",
    "muscle ache":"muscle_pain","body ache":"muscle_pain","body pain":"muscle_pain",
    "joint ache":"joint_pain","sore joints":"joint_pain","back ache":"back_pain",
    "tired":"fatigue","exhausted":"fatigue","no energy":"fatigue",
    "fever":"high_fever","temperature":"mild_fever",
    "weight loss":"weight_loss","losing weight":"weight_loss",
    "not hungry":"loss_of_appetite","no appetite":"loss_of_appetite",
    "swollen lymph nodes":"swelled_lymph_nodes",
    "chest tightness":"chest_pain","heart racing":"fast_heart_rate",
    "sore throat":"throat_irritation","throat pain":"throat_irritation",
    "cant smell":"loss_of_smell","no smell":"loss_of_smell",
}


# ─────────────────────────────────────────────────────────
# STEP 1 — DOWNLOAD DATA
# ─────────────────────────────────────────────────────────
def download_datasets():
    for url, path in [(TRAIN_URL, TRAIN_CSV), (TEST_URL, TEST_CSV)]:
        if not os.path.exists(path):
            print(f"  Downloading {path} …")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:
                sys.exit(f"\n✗ Could not download {path}.\n  {exc}")
        else:
            print(f"  Found cached {path}")


# ─────────────────────────────────────────────────────────
# STEP 2 — LOAD & CLEAN
# ─────────────────────────────────────────────────────────
def clean_col(col):
    return col.strip().lower().replace(" ", "_")

def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    train.dropna(axis=1, how="all", inplace=True)
    test.dropna(axis=1,  how="all", inplace=True)
    train.columns = [clean_col(c) for c in train.columns]
    test.columns  = [clean_col(c) for c in test.columns]
    symptom_cols = [c for c in train.columns if c != "prognosis"]
    train[symptom_cols] = train[symptom_cols].fillna(0).astype(int)
    test[symptom_cols]  = test[symptom_cols].fillna(0).astype(int)
    le = LabelEncoder()
    le.fit(pd.concat([train["prognosis"], test["prognosis"]]))
    return (train[symptom_cols].values, le.transform(train["prognosis"]),
            test[symptom_cols].values,  le.transform(test["prognosis"]),
            symptom_cols, le)


# ─────────────────────────────────────────────────────────
# STEP 3 — TRAIN
# ─────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes":   GaussianNB(),
    }
    for name, model in models.items():
        print(f"  Training {name} …")
        model.fit(X_train, y_train)
    return models


# ─────────────────────────────────────────────────────────
# STEP 4 — EVALUATE
# ─────────────────────────────────────────────────────────
def evaluate_models(models, X_test, y_test, le):
    print("\n" + "=" * 62)
    print("MODEL EVALUATION  (held-out test set)")
    print("=" * 62)
    for name, model in models.items():
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        print(f"\n{'─'*62}\n  {name:<24} Accuracy: {acc:.1%}\n{'─'*62}")
        print(classification_report(y_test, preds,
                                    target_names=le.classes_, zero_division=0))


# ─────────────────────────────────────────────────────────
# STEP 5a — NLP EXTRACTION  (LLM-powered)
# ─────────────────────────────────────────────────────────

NLP_SYSTEM_PROMPT = """You are a medical symptom extraction assistant.
Your job is to read a patient's free-text description and extract every symptom mentioned.

Rules:
- Extract symptoms even when expressed casually (e.g. "feel terrible", "skin is all yellow").
- Detect NEGATED symptoms (e.g. "no cough", "I don't have fever") — list them separately.
- Detect UNCERTAIN symptoms (e.g. "I think maybe a rash", "possibly dizzy") — list them separately.
- Normalise each symptom to a short, plain English phrase (e.g. "joint pain", "dark urine").
- Do NOT include body parts alone (e.g. "stomach") unless combined with a symptom word.
- Do NOT include diagnoses (e.g. "flu", "COVID") — only symptoms.

Respond ONLY with valid JSON (no markdown, no explanation):
{
  "symptoms":  ["<symptom1>", "<symptom2>", ...],
  "negated":   ["<symptom>", ...],
  "uncertain": ["<symptom>", ...]
}"""

# ── Model fallback chains — tried in order on 404 ─────────
ANTHROPIC_MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
OPENAI_MODELS    = ["gpt-4o-mini", "gpt-3.5-turbo"]
GEMINI_MODELS    = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]


def _is_rate_limit(exc) -> bool:
    """Return True if the exception signals a 429 rate-limit error."""
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg or "quota" in msg


def _is_not_found(exc) -> bool:
    """Return True if the exception signals a 404 / model-not-found error."""
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg or "does not exist" in msg or "invalid model" in msg


def _with_retry(fn, models_list: list, backend_name: str) -> dict:
    """
    Call fn(model_name) for each model in models_list.

    404 not found   → skip ALL remaining models, return {} immediately
    429 rate limit  → skip ALL remaining models, return {} immediately
    other error     → try next model in chain
    all models fail → return {} so fuzzy fallback takes over
    """
    for model in models_list:
        try:
            return fn(model)

        except Exception as exc:
            if _is_not_found(exc):
                print(f"  ⚠  [{backend_name}] 404 — model '{model}' not found.")
                print(f"  ℹ  Skipping AI entirely — switching to offline fuzzy matching.")
                return {}   # bail out immediately, no more models tried

            elif _is_rate_limit(exc):
                print(f"  ⚠  [{backend_name}] 429 — rate limit / quota exceeded.")
                print(f"  ℹ  Skipping AI entirely — switching to offline fuzzy matching.")
                return {}   # bail out immediately, no waiting

            else:
                # Unexpected error — try next model in chain
                print(f"  ⚠  [{backend_name}] '{model}' error: {exc} — trying next model")

    print(f"  ✗  [{backend_name}] all models exhausted — falling back to fuzzy matching")
    return {}


# ── Per-backend callers (accept model name as argument) ───

def _call_anthropic(text: str, model: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    msg = client.messages.create(
        model=model,
        max_tokens=300,
        system=NLP_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}]
    )
    return json.loads(msg.content[0].text.strip())


def _call_openai(text: str, model: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=OPENAI_KEY)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": NLP_SYSTEM_PROMPT},
            {"role": "user",   "content": text}
        ]
    )
    return json.loads(resp.choices[0].message.content.strip())


def _call_gemini(text: str, model: str) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_KEY)
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=NLP_SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=300,
            response_mime_type="application/json",
        )
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def extract_symptoms_via_llm(text: str) -> dict:
    """
    Call the configured LLM with automatic retry + model fallback.
    Returns {"symptoms": [...], "negated": [...], "uncertain": [...]}
    or {} if all attempts fail (triggers fuzzy fallback).

    Error handling:
      404 not found   → skip AI entirely, fuzzy matching immediately
      429 rate limit  → skip AI entirely, fuzzy matching immediately
      other error     → try next model in chain
      all models fail → return {} → fuzzy matching takes over
    """
    if NLP_BACKEND == "anthropic":
        return _with_retry(lambda m: _call_anthropic(text, m), ANTHROPIC_MODELS, "Anthropic")
    elif NLP_BACKEND == "openai":
        return _with_retry(lambda m: _call_openai(text, m),    OPENAI_MODELS,    "OpenAI")
    elif NLP_BACKEND == "gemini":
        return _with_retry(lambda m: _call_gemini(text, m),    GEMINI_MODELS,    "Gemini")
    return {}


# ─────────────────────────────────────────────────────────
# STEP 5b — FUZZY + SYNONYM RESOLVER  (offline fallback)
# ─────────────────────────────────────────────────────────
def display_name(col: str) -> str:
    return col.replace("_", " ")

def _fuzzy_resolve_token(token: str, symptom_cols: list,
                         display_names: list, display_to_col: dict) -> str | None:
    """Resolve one token to a canonical column name. Returns None if no match."""
    normalised = token.replace(" ", "_")

    # 1. Synonym map
    syn = SYNONYM_MAP.get(token)
    if syn and syn in symptom_cols:
        return syn

    # 2. Exact match
    if normalised in symptom_cols:
        return normalised

    # 3. Fuzzy
    hits = fz_process.extract(token, display_names, scorer=fuzz.WRatio, limit=FUZZY_TOP_N)
    best_str, best_score, _ = hits[0]
    if best_score >= FUZZY_THRESHOLD:
        close = [(h[0], h[1]) for h in hits if h[1] >= hits[0][1] - 5]
        if len(close) > 1:
            print(f"\n  ❓ '{token}' is ambiguous — did you mean:")
            for i, (opt, sc) in enumerate(close, 1):
                print(f"      [{i}] {opt}  (score {sc:.0f})")
            print("      [s] Skip")
            choice = input("     Your choice: ").strip().lower()
            if choice.isdigit() and 1 <= int(choice) <= len(close):
                return display_to_col[close[int(choice)-1][0]]
            return None
        col = display_to_col[best_str]
        if col != normalised:
            print(f"  ✔  '{token}'  →  '{display_name(col)}'  (score {best_score:.0f})")
        return col

    return None

def resolve_via_fuzzy(raw_input: str, symptom_cols: list) -> list:
    tokens = [t.strip().lower() for t in raw_input.split(",") if t.strip()]
    display_to_col = {display_name(c): c for c in symptom_cols}
    display_names  = list(display_to_col.keys())
    resolved = []
    for token in tokens:
        col = _fuzzy_resolve_token(token, symptom_cols, display_names, display_to_col)
        if col:
            resolved.append(col)
            print(f"  ✔  '{token}'  →  '{display_name(col)}'")
        else:
            print(f"  ✗  '{token}'  →  no match  (use [2] to browse symptoms)")
    return list(dict.fromkeys(resolved))


# ─────────────────────────────────────────────────────────
# STEP 5c — UNIFIED RESOLVER
#   LLM extracts symptoms from natural text, then fuzzy
#   maps each extracted phrase to a dataset column name.
# ─────────────────────────────────────────────────────────
def resolve_symptoms_nlp(user_text: str, symptom_cols: list) -> tuple[list, list, list]:
    """
    Returns (matched_cols, negated_labels, uncertain_labels).
    matched_cols   : canonical column names to set = 1 in the feature vector
    negated_labels : symptom phrases the user said they DON'T have
    uncertain_labels: symptom phrases the user is unsure about
    """
    display_to_col = {display_name(c): c for c in symptom_cols}
    display_names  = list(display_to_col.keys())

    extraction = extract_symptoms_via_llm(user_text)
    if not extraction:
        # Full fallback: treat the whole input as a comma-separated symptom list
        print("  ℹ  Using offline fuzzy matching.")
        matched = resolve_via_fuzzy(user_text, symptom_cols)
        return matched, [], []

    raw_symptoms  = extraction.get("symptoms",  [])
    raw_negated   = extraction.get("negated",   [])
    raw_uncertain = extraction.get("uncertain", [])

    print(f"\n  LLM extracted  → symptoms: {raw_symptoms}")
    if raw_negated:   print(f"                   negated : {raw_negated}")
    if raw_uncertain: print(f"                   unsure  : {raw_uncertain}")
    print()

    def map_list(phrases, track_unknowns=False):
        mapped, unknowns = [], []
        for phrase in phrases:
            col = _fuzzy_resolve_token(
                phrase.lower(), symptom_cols, display_names, display_to_col
            )
            if col:
                mapped.append(col)
            else:
                print(f"  ⚠  Could not map '{phrase}' to any known symptom — logging.")
                if track_unknowns:
                    unknowns.append(phrase)
        if unknowns:
            log_unknown_symptoms(unknowns)
        return list(dict.fromkeys(mapped))

    matched   = map_list(raw_symptoms, track_unknowns=True)
    negated   = map_list(raw_negated)
    uncertain = map_list(raw_uncertain)

    # Warn about uncertain symptoms and ask user whether to include them
    confirmed_uncertain = []
    for col in uncertain:
        ans = input(f"  ❓ You seemed unsure about '{display_name(col)}'. Include it? [y/N]: ").strip().lower()
        if ans == "y":
            confirmed_uncertain.append(col)
    matched = list(dict.fromkeys(matched + confirmed_uncertain))

    return matched, negated, uncertain



# ─────────────────────────────────────────────────────────
# STEP 5d — LEARNING: UNKNOWN SYMPTOM LOG + FEEDBACK STORE
# ─────────────────────────────────────────────────────────

def _load_json(path: str, default) -> dict | list:
    """Load a JSON file, returning default if missing or corrupt."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return default


def _save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Unknown symptom logger ────────────────────────────────

def log_unknown_symptoms(phrases: list[str]) -> None:
    """
    Record symptom phrases the LLM extracted but fuzzy-matching
    could not map to any known dataset column.
    Each entry records the phrase, how often it has been seen,
    and the date it was first and last encountered.
    """
    if not phrases:
        return
    store: dict = _load_json(UNKNOWN_SYMPTOMS_FILE, {})
    today = datetime.date.today().isoformat()
    for phrase in phrases:
        key = phrase.lower().strip()
        if key in store:
            store[key]["count"] += 1
            store[key]["last_seen"] = today
        else:
            store[key] = {"count": 1, "first_seen": today, "last_seen": today}
    _save_json(UNKNOWN_SYMPTOMS_FILE, store)
    print(f"  📝 Logged {len(phrases)} unknown symptom(s) to {UNKNOWN_SYMPTOMS_FILE}")


def show_unknown_symptoms() -> None:
    """Print unknown symptoms sorted by frequency."""
    store: dict = _load_json(UNKNOWN_SYMPTOMS_FILE, {})
    if not store:
        print("\n  No unknown symptoms logged yet.")
        return
    print(f"\n  Unknown symptoms logged ({len(store)} unique phrases):")
    print(f"  {'Symptom phrase':<40} {'Seen':>5}  First seen   Last seen")
    print(f"  {'─'*40} {'─'*5}  {'─'*10}   {'─'*10}")
    for phrase, info in sorted(store.items(), key=lambda x: -x[1]["count"]):
        print(f"  {phrase:<40} {info['count']:>5}  "
              f"{info['first_seen']}   {info['last_seen']}")


# ── Feedback store ────────────────────────────────────────

def ask_feedback(matched_cols: list[str], predicted_disease: str) -> None:
    """
    Ask the user whether the top prediction was correct and store
    the symptom vector + confirmed label in feedback_log.json.
    This data can later be used to retrain the model.
    """
    print()
    ans = input("  💬 Was this prediction correct? [y/n/skip]: ").strip().lower()
    if ans not in ("y", "n"):
        return

    if ans == "n":
        print("  What disease do you think it actually is?")
        correct = input("  Correct disease (or press Enter to skip): ").strip()
        if not correct:
            return
        label = correct
    else:
        label = predicted_disease

    record = {
        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        "symptoms":   matched_cols,
        "label":      label,
        "correct":    ans == "y",
    }
    log: list = _load_json(FEEDBACK_LOG_FILE, [])
    log.append(record)
    _save_json(FEEDBACK_LOG_FILE, log)
    print(f"  ✔  Feedback saved ({len(log)} total entries in {FEEDBACK_LOG_FILE})")

    if len(log) >= MIN_FEEDBACK_TO_RETRAIN:
        print(f"\n  ℹ  You have {len(log)} feedback entries — enough to retrain.")
        print("     Use option [4] in the menu to rebuild the model with this data.")


def show_feedback_summary() -> None:
    """Print a summary of collected feedback entries."""
    log: list = _load_json(FEEDBACK_LOG_FILE, [])
    if not log:
        print("\n  No feedback collected yet.")
        return
    correct   = sum(1 for r in log if r.get("correct"))
    incorrect = len(log) - correct
    diseases  = {}
    for r in log:
        diseases[r["label"]] = diseases.get(r["label"], 0) + 1
    print(f"\n  Feedback log — {len(log)} entries")
    print(f"  Correct predictions : {correct}")
    print(f"  Corrected by user   : {incorrect}")
    print(f"  Diseases in log     : {len(diseases)}")
    print(f"\n  Label distribution:")
    for disease, count in sorted(diseases.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {disease:<44} {count:>3}  {bar}")


# ── Retrainer ─────────────────────────────────────────────

def retrain_from_feedback(models: dict, symptom_cols: list,
                          le: LabelEncoder) -> dict:
    """
    Combine the original training CSV with confirmed feedback rows
    and retrain all three classifiers on the merged dataset.

    Only feedback rows whose label exists in le.classes_ are used
    (unknown disease names are logged but skipped).
    """
    log: list = _load_json(FEEDBACK_LOG_FILE, [])
    if not log:
        print("  ✗  No feedback data found. Use the predictor first.")
        return models

    # Build feedback feature matrix
    known_diseases = set(le.classes_)
    rows_X, rows_y = [], []
    skipped = 0
    for record in log:
        label = record.get("label", "").strip()
        if label not in known_diseases:
            skipped += 1
            continue
        vec = [1 if col in record.get("symptoms", []) else 0
               for col in symptom_cols]
        rows_X.append(vec)
        rows_y.append(label)

    if not rows_X:
        print(f"  ✗  No usable feedback rows "
              f"({skipped} skipped — unknown disease labels).")
        return models

    # Load original training data
    train_df = pd.read_csv(TRAIN_CSV)
    train_df.dropna(axis=1, how="all", inplace=True)
    train_df.columns = [c.strip().lower().replace(" ", "_")
                        for c in train_df.columns]
    cols = [c for c in train_df.columns if c != "prognosis"]
    X_orig = train_df[cols].fillna(0).astype(int).values
    y_orig = train_df["prognosis"].values

    # Merge
    X_fb = np.array(rows_X)
    y_fb = np.array(rows_y)
    X_all = np.vstack([X_orig, X_fb])
    y_all = np.concatenate([y_orig, y_fb])

    print(f"\n  Retraining on {len(X_orig)} original + "
          f"{len(X_fb)} feedback rows ({skipped} skipped) …")

    new_models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes":   GaussianNB(),
    }
    for name, model in new_models.items():
        model.fit(X_all, y_all)
        print(f"  ✔  {name} retrained")

    print("  ✅ Retraining complete! New models are active for this session.")
    print("     (Models are not saved to disk — retrain each run or add pickle support.)")
    return new_models

# ─────────────────────────────────────────────────────────
# STEP 6 — PREDICT
# ─────────────────────────────────────────────────────────
def symptoms_to_vector(matched_cols, symptom_cols):
    return np.array([[1 if col in matched_cols else 0 for col in symptom_cols]])

def top_predictions(model, vec, le, top_n=5):
    if hasattr(model, "predict_proba"):
        probs   = model.predict_proba(vec)[0]
        indices = np.argsort(probs)[::-1][:top_n]
        return [(le.classes_[i], probs[i]) for i in indices]
    pred = model.predict(vec)[0]
    return [(le.classes_[pred], 1.0)]


# ─────────────────────────────────────────────────────────
# STEP 7 — INTERACTIVE CLI
# ─────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          DISEASE PREDICTOR  —  NLP Symptom Checker              ║
║   Dataset : Kaggle / kaushil268  •  41 diseases                 ║
║   NLP     : {backend:<46}║
╚══════════════════════════════════════════════════════════════════╝"""

def print_symptom_list(symptom_cols):
    print("\nAvailable symptoms (132 total):\n")
    readable = [display_name(c) for c in symptom_cols]
    col_w = 36
    for i in range(0, len(readable), 3):
        row = readable[i : i + 3]
        print("".join(f"  {i+j+1:3}. {s:<{col_w}}" for j, s in enumerate(row)))


def run_prediction(models, symptom_cols, le, compare=False):
    backend_label = {
        "anthropic": "Claude (Anthropic API)",
        "openai":    "GPT-4o-mini (OpenAI API)",
        "gemini":    "Gemini 1.5 Flash (Google API)",
        "fuzzy":     "Offline fuzzy matching",
    }[NLP_BACKEND]

    if NLP_BACKEND != "fuzzy":
        print(f"\n  Speak naturally — describe your symptoms in your own words.")
        print(f"  Example: \"I've had a high fever for two days, my joints ache")
        print(f"            terribly, and I noticed a rash on my arms this morning.\"")
    else:
        print(f"\n  Enter symptoms as comma-separated phrases.")
        print(f"  Example: fever, joint pain, skin rash, fatigue")

    print()
    raw = input("Describe symptoms: ").strip()
    if not raw:
        print("  ⚠  No input.")
        return

    print()
    matched, negated, uncertain = resolve_symptoms_nlp(raw, symptom_cols)

    if not matched:
        print("\n  ✗  No symptoms matched. Try [2] to browse all 132 symptoms.")
        return

    print(f"\n  ✔  Symptoms used : {', '.join(display_name(c) for c in matched)}")
    if negated:
        print(f"  ✗  Symptoms absent: {', '.join(display_name(c) for c in negated)}")
    vec = symptoms_to_vector(matched, symptom_cols)

    if not compare:
        results = top_predictions(models["Random Forest"], vec, le, top_n=5)
        print(f"\n  ┌─ Random Forest — Top Predictions {'─'*27}┐")
        for rank, (disease, prob) in enumerate(results, 1):
            bar = "█" * int(prob * 30)
            pad = " " * (30 - len(bar))
            print(f"  │  {rank}. {disease:<44} {prob:5.1%}  {bar}{pad} │")
        print(f"  └{'─'*66}┘")
    else:
        print()
        for name, model in models.items():
            results = top_predictions(model, vec, le, top_n=3)
            print(f"  [{name}]")
            for rank, (disease, prob) in enumerate(results, 1):
                bar = "█" * int(prob * 25)
                print(f"    {rank}. {disease:<46} {prob:5.1%}  {bar}")
            print()

    print("  ⚕  For educational purposes only. Consult a healthcare professional.")

    # Ask for feedback on the top prediction (Random Forest result)
    top_rf = top_predictions(models["Random Forest"], vec, le, top_n=1)
    if top_rf:
        ask_feedback(matched, top_rf[0][0])


def interactive_session(models, symptom_cols, le):
    backend_desc = {
        "anthropic": "Claude API (NLP mode — full sentence input)",
        "openai":    "OpenAI API (NLP mode — full sentence input)",
        "gemini":    "Gemini API (NLP mode — full sentence input)",
        "fuzzy":     "Offline fallback (no API key set — comma-separated input)",
    }[NLP_BACKEND]
    print(BANNER.format(backend=backend_desc))

    while True:
        print("\nOptions:")
        print("  [1] Predict disease from symptoms")
        print("  [2] List all 132 available symptoms")
        print("  [3] Compare all three classifiers")
        print("  [4] Retrain model with feedback data")
        print("  [5] View unknown symptom log")
        print("  [6] View feedback summary")
        print("  [q] Quit")
        choice = input("\nChoice: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("\nGoodbye! Always consult a licensed physician.\n")
            break
        elif choice == "1":
            run_prediction(models, symptom_cols, le, compare=False)
        elif choice == "2":
            print_symptom_list(symptom_cols)
        elif choice == "3":
            run_prediction(models, symptom_cols, le, compare=True)
        elif choice == "4":
            models = retrain_from_feedback(models, symptom_cols, le)
        elif choice == "5":
            show_unknown_symptoms()
        elif choice == "6":
            show_feedback_summary()
        else:
            print("  Invalid choice.")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    backend_label = {"anthropic":"Anthropic (Claude)","openai":"OpenAI (GPT-4o-mini)","gemini":"Google (Gemini 1.5 Flash)","fuzzy":"offline fuzzy matching"}[NLP_BACKEND]
    print(f"\n🔬 Disease Predictor  |  NLP backend: {backend_label}\n")
    if NLP_BACKEND == "fuzzy":
        print("  ℹ  No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        print("     to enable full NLP sentence parsing.\n")

    print("[1/4] Checking dataset …")
    download_datasets()

    print("\n[2/4] Loading & cleaning data …")
    X_train, y_train, X_test, y_test, symptom_cols, le = load_data()
    print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows | "
          f"Symptoms: {len(symptom_cols)} | Diseases: {len(le.classes_)}")

    print("\n[3/4] Training classifiers …")
    models = train_models(X_train, y_train)

    print("\n[4/4] Evaluating on test set …")
    evaluate_models(models, X_test, y_test, le)

    print("\n✅ Ready.\n")
    interactive_session(models, symptom_cols, le)


if __name__ == "__main__":
    main()