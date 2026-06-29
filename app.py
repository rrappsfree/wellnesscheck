"""
app.py — Unified Flask API: Kaggle + DDXPlus Wellness Check
==============================================================
ALL interactions are handled via JSON API — no input() calls,
no CLI prompts, no blocking. The frontend drives everything.

Ambiguous symptom matching
--------------------------
When a symptom phrase matches multiple candidates within 5 points,
the API returns them in an "ambiguous" list. The frontend shows a
dialog and POSTs the resolved choice back to /api/resolve before
running the final prediction.

Endpoints
---------
  GET  /                          → index.html
  GET  /api/status                → loaded datasets + stats
  GET  /api/symptoms?dataset=X    → full symptom list
  POST /api/predict               → main prediction endpoint
  POST /api/resolve               → submit resolved ambiguous symptoms
  POST /api/feedback              → save user correction
  GET  /api/unknown-symptoms      → unmatched symptom log
  GET  /api/feedback-summary      → feedback stats
"""

import os, sys, json, datetime
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from rapidfuzz import process as fz_process, fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UNKNOWN_FILE  = "unknown_symptoms.json"
FEEDBACK_FILE = "feedback_log.json"
FUZZY_THRESHOLD = 70
FUZZY_TOP_N     = 3

def _load_json(path, default):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    return default

def _save_json(path, data):
    json.dump(data, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def _log_unknowns(phrases):
    if not phrases: return
    store = _load_json(UNKNOWN_FILE, {})
    today = datetime.date.today().isoformat()
    for p in phrases:
        key = p.lower().strip()
        store[key] = store.get(key, {"count": 0,
                                      "first_seen": today,
                                      "last_seen": today})
        store[key]["count"]    += 1
        store[key]["last_seen"] = today
    _save_json(UNKNOWN_FILE, store)

# ═══════════════════════════════════════════════════════════
# DATASET A — KAGGLE
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  Loading Dataset A: Kaggle (kaushil268)")
print("="*60)

from disease_predictor_kaggle_AI import (
    download_datasets, load_data,
    train_models        as kaggle_train_models,
    symptoms_to_vector  as kaggle_sym_vec,
    top_predictions     as kaggle_top_preds,
    display_name        as kaggle_display,
    extract_symptoms_via_llm, NLP_BACKEND,
)

download_datasets()
_, _ytr, _, _, KAGGLE_COLS, KAGGLE_LE = load_data()
# retrain on full set
from disease_predictor_kaggle_AI import load_data as _kload
_Xtr,_ytr,_Xte,_yte,KAGGLE_COLS,KAGGLE_LE = _kload()
KAGGLE_MODELS = kaggle_train_models(_Xtr, _ytr)

KAGGLE_DISPLAY_TO_COL = {kaggle_display(c): c for c in KAGGLE_COLS}
KAGGLE_DISPLAY_NAMES  = list(KAGGLE_DISPLAY_TO_COL.keys())
print(f"  ✔  Kaggle ready — {len(KAGGLE_COLS)} symptoms | {len(KAGGLE_LE.classes_)} diseases\n")

# ═══════════════════════════════════════════════════════════
# DATASET B — DDXPlus
# ═══════════════════════════════════════════════════════════
print("="*60)
print("  Loading Dataset B: DDXPlus (NeurIPS 2022)")
print("="*60)

DDX_AVAILABLE        = False
DDX_META             = {}
DDX_MODELS           = {}
DDX_EVIDENCES        = {}
DDX_ALL_CODES        = []
DDX_CODE_INDEX       = {}
DDX_LE               = None
DDX_READABLE_TO_CODE = {}
DDX_READABLE_PHRASES = []

try:
    from disease_predictor_ddxplus import (
        locate_files, load_metadata, build_feature_schema,
        load_patients, train_models as ddx_train_models,
        top_predictions as ddx_top_preds,
        build_symptom_lookup, MAX_TRAIN_ROWS,
    )
    from sklearn.preprocessing import LabelEncoder as _LE

    _files             = locate_files()
    _evs, _conds       = load_metadata(_files)
    _codes, _cidx      = build_feature_schema(_evs)
    _ddx_le            = _LE()
    _ddx_le.fit(list(_conds.keys()))

    _Xtr_d,_ytr_d = load_patients(_files["train"],_codes,_cidx,_ddx_le,MAX_TRAIN_ROWS)
    _Xte_d,_yte_d = load_patients(_files["test"], _codes,_cidx,_ddx_le)

    DDX_MODELS           = ddx_train_models(_Xtr_d, _ytr_d)
    DDX_EVIDENCES        = _evs
    DDX_ALL_CODES        = _codes
    DDX_CODE_INDEX       = _cidx
    DDX_LE               = _ddx_le
    DDX_READABLE_TO_CODE, DDX_READABLE_PHRASES = build_symptom_lookup(_evs)
    DDX_META = {
        "evidences":  len(_evs),
        "diseases":   len(_conds),
        "train_rows": int(_Xtr_d.shape[0]),
        "features":   int(_Xtr_d.shape[1]),
    }
    DDX_AVAILABLE = True
    print(f"  ✔  DDXPlus ready — {len(_evs)} evidences | {len(_conds)} diseases\n")

except SystemExit:
    print("  ⚠  DDXPlus files not found.")
    print("     Download: https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585\n")
except Exception as e:
    print(f"  ⚠  DDXPlus load error: {e}\n")


# ═══════════════════════════════════════════════════════════
# NON-INTERACTIVE FUZZY RESOLVERS
# These never call input() — ambiguous matches are returned
# to the frontend for the user to resolve via UI.
# ═══════════════════════════════════════════════════════════

def _fuzzy_match_web(phrase, display_names, display_to_col):
    """
    Match one phrase against known symptom names.
    Returns:
      ("match",   col_name,          None)           — unambiguous match
      ("ambiguous", best_col,   candidates_list)     — multiple close matches
      ("unknown", None,          None)               — no match above threshold
    """
    hits = fz_process.extract(phrase, display_names,
                               scorer=fuzz.WRatio, limit=FUZZY_TOP_N)
    if not hits:
        return "unknown", None, None

    best_str, best_score, _ = hits[0]
    if best_score < FUZZY_THRESHOLD:
        return "unknown", None, None

    close = [(h[0], round(h[1], 1)) for h in hits if h[1] >= hits[0][1] - 5]
    if len(close) > 1:
        # Return all candidates for the frontend to show a picker
        candidates = [{"label": label, "col": display_to_col[label], "score": sc}
                      for label, sc in close]
        return "ambiguous", display_to_col[best_str], candidates

    return "match", display_to_col[best_str], None


def resolve_kaggle_symptoms(phrases, track_unknowns=False):
    """
    Resolve a list of phrases against Kaggle symptom columns.
    Returns: { matched, ambiguous, unknowns }
      matched   : list of resolved column names
      ambiguous : [{ phrase, candidates:[{label,col,score}] }]
      unknowns  : phrases with no match
    """
    matched, ambiguous, unknowns = [], [], []

    for phrase in phrases:
        phrase_lower = phrase.lower().strip()

        # Exact match on internal col name
        norm = phrase_lower.replace(" ", "_")
        if norm in KAGGLE_COLS:
            matched.append(norm)
            continue

        # Exact match on display name
        if phrase_lower in KAGGLE_DISPLAY_TO_COL:
            matched.append(KAGGLE_DISPLAY_TO_COL[phrase_lower])
            continue

        status, col, candidates = _fuzzy_match_web(
            phrase_lower, KAGGLE_DISPLAY_NAMES, KAGGLE_DISPLAY_TO_COL
        )
        if status == "match":
            matched.append(col)
        elif status == "ambiguous":
            ambiguous.append({"phrase": phrase, "candidates": candidates})
        else:
            if track_unknowns:
                unknowns.append(phrase)

    return {
        "matched":   list(dict.fromkeys(matched)),
        "ambiguous": ambiguous,
        "unknowns":  unknowns,
    }


def resolve_ddxplus_symptoms(phrases):
    """
    Resolve a list of phrases against DDXPlus evidence readable labels.
    Returns: { matched_codes, matched_labels, ambiguous, unknowns }
    """
    matched_codes, matched_labels, ambiguous, unknowns = [], [], [], []

    for phrase in phrases:
        phrase_lower = phrase.lower().strip()

        # Exact match
        if phrase_lower in DDX_READABLE_TO_CODE:
            code = DDX_READABLE_TO_CODE[phrase_lower]
            matched_codes.append(code)
            matched_labels.append(DDX_EVIDENCES[code]["readable"])
            continue

        status, col, candidates = _fuzzy_match_web(
            phrase_lower, DDX_READABLE_PHRASES, DDX_READABLE_TO_CODE
        )
        if status == "match":
            matched_codes.append(col)
            matched_labels.append(DDX_EVIDENCES[col]["readable"])
        elif status == "ambiguous":
            # candidates col = evidence code here
            ambiguous.append({"phrase": phrase, "candidates": candidates})
        else:
            unknowns.append(phrase)

    return {
        "matched_codes":  list(dict.fromkeys(matched_codes)),
        "matched_labels": list(dict.fromkeys(matched_labels)),
        "ambiguous":      ambiguous,
        "unknowns":       unknowns,
    }


# ═══════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════
import os as _os
app = Flask(__name__, template_folder=".",
           static_folder=_os.path.dirname(_os.path.abspath(__file__)),
           static_url_path='/static')
CORS(app)

def _err(msg, code=400):
    return jsonify({"error": msg}), code


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "kaggle": {
            "available":    True,
            "symptoms":     len(KAGGLE_COLS),
            "diseases":     len(KAGGLE_LE.classes_),
            "disease_list": list(KAGGLE_LE.classes_),
        },
        "ddxplus": {
            "available": DDX_AVAILABLE,
            **DDX_META,
            **({"disease_list": list(DDX_LE.classes_)} if DDX_AVAILABLE else {}),
        },
        "nlp_backend": NLP_BACKEND,
    })


@app.route("/api/symptoms")
def symptoms():
    ds = request.args.get("dataset", "kaggle").lower()
    if ds == "ddxplus":
        if not DDX_AVAILABLE:
            return _err("DDXPlus not loaded.", 503)
        return jsonify({
            "dataset":  "ddxplus",
            "symptoms": [{"code": c, "label": info["readable"]}
                         for c, info in DDX_EVIDENCES.items()],
            "count": len(DDX_EVIDENCES),
        })
    return jsonify({
        "dataset":  "kaggle",
        "symptoms": [kaggle_display(c) for c in KAGGLE_COLS],
        "count":    len(KAGGLE_COLS),
    })


# ─────────────────────────────────────────────────────────
# POST /api/predict
# ─────────────────────────────────────────────────────────
# Response shape:
#   If all symptoms resolved cleanly:
#     { status:"ok", predictions, matched_symptoms, negated_symptoms,
#       all_model_predictions, dataset, [demographics] }
#
#   If some symptoms are ambiguous:
#     { status:"ambiguous",
#       ambiguous: [{ phrase, candidates:[{label,col,score}] }],
#       resolved:  [...already matched cols...],
#       negated:   [...],
#       partial_text: original text,
#       dataset, age, sex }
#     → Frontend shows picker, user picks, POSTs to /api/resolve
#
# ─────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    body    = request.get_json(silent=True) or {}
    text    = (body.get("text") or "").strip()
    dataset = (body.get("dataset") or "kaggle").lower()
    age     = float(body.get("age") or 40)
    sex     = (body.get("sex") or "M").upper()

    if not text:
        return _err("Missing 'text' field.")

    # ── Extract symptoms via LLM (or treat as comma list) ──
    extraction = extract_symptoms_via_llm(text)

    if extraction:
        symptom_phrases  = (extraction.get("symptoms",  []) +
                            extraction.get("uncertain", []))
        negated_phrases  =  extraction.get("negated",   [])
    else:
        # Fallback: comma-separated
        symptom_phrases  = [t.strip() for t in text.split(",") if t.strip()]
        negated_phrases  = []

    if dataset == "ddxplus":
        return _handle_ddxplus(symptom_phrases, negated_phrases, age, sex, body)
    return _handle_kaggle(symptom_phrases, negated_phrases, body)


def _handle_kaggle(symptom_phrases, negated_phrases, body):
    resolved = resolve_kaggle_symptoms(symptom_phrases, track_unknowns=True)

    if resolved["unknowns"]:
        _log_unknowns(resolved["unknowns"])

    # If ambiguous symptoms exist, ask the frontend to resolve them
    if resolved["ambiguous"]:
        return jsonify({
            "status":       "ambiguous",
            "dataset":      "kaggle",
            "ambiguous":    resolved["ambiguous"],
            "resolved":     resolved["matched"],    # already confirmed cols
            "negated":      [kaggle_display(c) for c in
                             resolve_kaggle_symptoms(negated_phrases)["matched"]],
            "partial_text": body.get("text", ""),
        })

    matched = resolved["matched"]
    if not matched:
        return jsonify({"status": "ok", "dataset": "kaggle",
                        "predictions": [],
                        "message": "No symptoms matched. Try rephrasing."})

    return _run_kaggle_prediction(matched, negated_phrases)


def _handle_ddxplus(symptom_phrases, negated_phrases, age, sex, body):
    if not DDX_AVAILABLE:
        return _err("DDXPlus not loaded. Download from figshare and restart.", 503)

    resolved = resolve_ddxplus_symptoms(symptom_phrases)

    if resolved["unknowns"]:
        _log_unknowns(resolved["unknowns"])

    if resolved["ambiguous"]:
        return jsonify({
            "status":       "ambiguous",
            "dataset":      "ddxplus",
            "ambiguous":    resolved["ambiguous"],
            "resolved":     resolved["matched_codes"],
            "negated":      negated_phrases,
            "partial_text": body.get("text", ""),
            "age":          age,
            "sex":          sex,
        })

    matched_codes  = resolved["matched_codes"]
    matched_labels = resolved["matched_labels"]
    if not matched_codes:
        return jsonify({"status": "ok", "dataset": "ddxplus",
                        "predictions": [],
                        "message": "No symptoms matched. Try rephrasing."})

    return _run_ddxplus_prediction(matched_codes, matched_labels,
                                   negated_phrases, age, sex)


# ─────────────────────────────────────────────────────────
# POST /api/resolve
# Called after the user picks from the ambiguous-symptom dialog.
# Body: { dataset, resolved_cols, negated, age, sex }
#   resolved_cols : all matched cols (already resolved + user's new picks)
# ─────────────────────────────────────────────────────────
@app.route("/api/resolve", methods=["POST"])
def resolve():
    body    = request.get_json(silent=True) or {}
    dataset = (body.get("dataset") or "kaggle").lower()
    cols    = body.get("resolved_cols", [])
    negated = body.get("negated", [])
    age     = float(body.get("age") or 40)
    sex     = (body.get("sex") or "M").upper()

    if not cols:
        return _err("No resolved symptoms provided.")

    if dataset == "ddxplus":
        # cols are evidence codes; rebuild labels
        labels = [DDX_EVIDENCES[c]["readable"]
                  for c in cols if c in DDX_EVIDENCES]
        return _run_ddxplus_prediction(cols, labels, negated, age, sex)

    return _run_kaggle_prediction(cols, negated)


# ── Prediction runners ─────────────────────────────────────

def _run_kaggle_prediction(matched_cols, negated_phrases):
    neg_resolved = resolve_kaggle_symptoms(negated_phrases)["matched"]

    vec       = kaggle_sym_vec(matched_cols, KAGGLE_COLS)
    all_preds = []
    for mname, model in KAGGLE_MODELS.items():
        for disease, prob in kaggle_top_preds(model, vec, KAGGLE_LE, top_n=5):
            all_preds.append({"disease": disease,
                               "probability": round(float(prob), 4),
                               "model": mname})

    rf = sorted([p for p in all_preds if p["model"] == "Random Forest"],
                key=lambda x: -x["probability"])

    return jsonify({
        "status":               "ok",
        "dataset":              "kaggle",
        "matched_symptoms":     [kaggle_display(c) for c in matched_cols],
        "negated_symptoms":     [kaggle_display(c) for c in neg_resolved],
        "predictions":           rf,
        "all_model_predictions": all_preds,
    })


def _run_ddxplus_prediction(matched_codes, matched_labels,
                             negated_phrases, age, sex):
    vec_data = [0] * len(DDX_ALL_CODES)
    for code in matched_codes:
        if code in DDX_CODE_INDEX:
            vec_data[DDX_CODE_INDEX[code]] = 1

    vec = np.array([vec_data + [age / 100.0, 1.0 if sex == "M" else 0.0]],
                   dtype=np.float32)

    all_preds = []
    for mname, model in DDX_MODELS.items():
        for disease, prob in ddx_top_preds(model, vec, DDX_LE, top_n=5):
            all_preds.append({"disease": disease,
                               "probability": round(float(prob), 4),
                               "model": mname})

    rf_key = next((k for k in DDX_MODELS if "Random" in k),
                  list(DDX_MODELS.keys())[0])
    rf     = sorted([p for p in all_preds if p["model"] == rf_key],
                    key=lambda x: -x["probability"])

    return jsonify({
        "status":               "ok",
        "dataset":              "ddxplus",
        "matched_symptoms":     matched_labels,
        "negated_symptoms":     negated_phrases,
        "predictions":           rf,
        "all_model_predictions": all_preds,
        "demographics":          {"age": int(age), "sex": sex},
    })


# ── Other endpoints ────────────────────────────────────────

@app.route("/api/feedback", methods=["POST"])
def feedback():
    body     = request.get_json(silent=True) or {}
    symptoms = body.get("symptoms", [])
    label    = (body.get("label") or "").strip()
    correct  = bool(body.get("correct", True))
    dataset  = (body.get("dataset") or "kaggle").lower()
    if not symptoms or not label:
        return _err("'symptoms' and 'label' are required.")
    record = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
              "dataset": dataset, "symptoms": symptoms,
              "label": label, "correct": correct}
    log = _load_json(FEEDBACK_FILE, [])
    log.append(record)
    _save_json(FEEDBACK_FILE, log)
    return jsonify({"saved": True, "total": len(log),
                    "message": f"Feedback saved ({len(log)} total)."})


@app.route("/api/unknown-symptoms")
def unknown_symptoms():
    store = _load_json(UNKNOWN_FILE, {})
    items = sorted(store.items(), key=lambda x: -x[1]["count"])
    return jsonify({"count": len(items),
                    "unknown_symptoms": [{"phrase": k, **v} for k, v in items]})


@app.route("/api/feedback-summary")
def feedback_summary():
    log     = _load_json(FEEDBACK_FILE, [])
    correct = sum(1 for r in log if r.get("correct"))
    diseases, by_ds = {}, {}
    for r in log:
        diseases[r["label"]] = diseases.get(r["label"], 0) + 1
        ds = r.get("dataset", "kaggle")
        by_ds[ds] = by_ds.get(ds, 0) + 1
    return jsonify({"total": len(log), "correct": correct,
                    "incorrect": len(log) - correct,
                    "by_dataset": by_ds, "diseases": diseases})



@app.route("/admin")
def admin():
    return send_from_directory(".", "admin.html")



# ═══════════════════════════════════════════════════════════
# BLOG
# ═══════════════════════════════════════════════════════════

# ── Article helpers ────────────────────────────────────────
ARTICLES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "articles.json")

def _load_articles() -> list:
    """
    Load articles from articles.json on every call.
    No restart needed — just save the file and the next request picks it up.
    Returns [] if the file is missing or malformed.
    """
    try:
        with open(ARTICLES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  ⚠  {ARTICLES_FILE} not found — no articles loaded.")
        return []
    except json.JSONDecodeError as e:
        print(f"  ⚠  articles.json is invalid JSON: {e}")
        return []


@app.route("/blog")
def blog_index():
    return send_from_directory(".", "blog.html")


@app.route("/blog/<slug>")
def blog_article(slug):
    return send_from_directory(".", "article.html")


@app.route("/api/articles")
def api_articles():
    """Return article list without content bodies (for the listing page)."""
    articles = _load_articles()
    return jsonify([{k: v for k, v in a.items() if k != "content"}
                    for a in articles])


@app.route("/api/articles/<slug>")
def api_article(slug):
    """Return a single article including its full HTML content."""
    articles = _load_articles()
    article  = next((a for a in articles if a["slug"] == slug), None)
    if not article:
        return jsonify({"error": "Article not found"}), 404
    return jsonify(article)


@app.route("/api/articles", methods=["POST"])
def api_add_article():
    """
    Add a new article by POSTing JSON to /api/articles.
    Required fields: slug, title, category, date, author,
                     summary, read_time, image_emoji, content
    The article is appended to articles.json immediately —
    no server restart required.
    """
    body = request.get_json(silent=True) or {}

    required = ["slug", "title", "category", "date",
                "author", "summary", "read_time", "image_emoji", "content"]
    missing  = [f for f in required if not body.get(f)]
    if missing:
        return _err(f"Missing required fields: {', '.join(missing)}")

    articles = _load_articles()

    # Prevent duplicate slugs
    if any(a["slug"] == body["slug"] for a in articles):
        return _err(f"An article with slug \"{body['slug']}\" already exists.")

    articles.append(body)
    try:
        with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
    except IOError as e:
        return _err(f"Could not save articles.json: {e}", 500)

    return jsonify({"saved": True, "total": len(articles),
                    "slug": body["slug"],
                    "message": f"Article \"{body['title']}\" added successfully."})


@app.route("/api/articles/<slug>", methods=["PUT"])
def api_update_article(slug):
    """Update an existing article in-place."""
    body     = request.get_json(silent=True) or {}
    articles = _load_articles()
    idx      = next((i for i, a in enumerate(articles) if a["slug"] == slug), None)
    if idx is None:
        return _err("Article not found.", 404)
    articles[idx].update(body)
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    return jsonify({"updated": True, "slug": slug})


@app.route("/api/articles/<slug>", methods=["DELETE"])
def api_delete_article(slug):
    """Delete an article by slug."""
    articles = _load_articles()
    new_list = [a for a in articles if a["slug"] != slug]
    if len(new_list) == len(articles):
        return _err("Article not found.", 404)
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(new_list, f, indent=2, ensure_ascii=False)
    return jsonify({"deleted": True, "slug": slug,
                    "remaining": len(new_list)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🌐  Running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
