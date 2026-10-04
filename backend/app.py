import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flask import Flask, jsonify, request, send_from_directory
from backend.config import ROOT, PROC, OUT, RAW_CSV, EE_PROJECT, STUDY_STATE, S2_START, S2_END, FEATURES, UserError
from ml import preprocessing as pp

app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
for d in (PROC, OUT, ROOT / "models"): d.mkdir(parents=True, exist_ok=True)

@app.errorhandler(UserError)
def user_err(e): return jsonify(error=str(e)), 400
@app.errorhandler(Exception)
def any_err(e): return jsonify(error=f"Unexpected error: {str(e)[:300]}"), 500

def _json(name):
    f = PROC / name
    return json.loads(f.read_text()) if f.exists() else None

@app.route("/")
def index(): return send_from_directory(app.static_folder, "index.html")

@app.route("/api/status")
def status():
    return jsonify(dataset_present=RAW_CSV.exists(), ee_project_set=bool(EE_PROJECT) and not EE_PROJECT.startswith("your-"),
                   s2_period=[S2_START, S2_END], study_state=STUDY_STATE, features=FEATURES,
                   features_built=(PROC / "features.csv").exists(), metrics_ready=(PROC / "metrics.json").exists(),
                   prediction_ready=(OUT / "prospectivity.geojson").exists(),
                   message=None if RAW_CSV.exists() else pp.MISSING_MSG)

@app.route("/api/occurrences")
def occurrences():
    state = request.args.get("state", "").strip()

    df, rep = pp.load_occurrences()

    if state:
        df = df[
            df["state"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            == state.lower()
        ].copy()

    return jsonify(
        report=rep,
        points=df[
            ["lat", "lon", "locality", "state", "host_rock"]
        ]
        .fillna("")
        .to_dict("records")
    )
@app.route("/api/build-features", methods=["POST"])
def build_features():
    from ml.feature_extraction import extract
    state = (request.json or {}).get("state", STUDY_STATE)
    df, rep = pp.load_occurrences()
    occ, bbox = pp.study_area(df, state)
    bg = pp.background_points(occ, bbox)
    pos = extract(occ[["lat", "lon"]], bbox); pos["label"] = 1
    neg = extract(bg, bbox); neg["label"] = 0
    feats = __import__("pandas").concat([pos, neg], ignore_index=True)
    feats.to_csv(PROC / "features.csv", index=False)
    summary = {"validation": rep, "bbox": bbox, "state_filter": state or "none",
               "occurrences_used": len(occ), "positives_with_satellite_data": len(pos), "background_with_satellite_data": len(neg),
               "class_balance": {"positive": len(pos), "background": len(neg)},
               "missing_values": int(feats[FEATURES].isna().sum().sum()),
               "note": "Background points are random locations >=10 km from known occurrences; they are NOT confirmed barren."}
    (PROC / "study_area.json").write_text(json.dumps(summary, indent=2))
    return jsonify(summary)

@app.route("/api/summary")
def summary(): return jsonify(_json("study_area.json") or {})

@app.route("/api/train", methods=["POST"])
def train():
    from ml.train import train_all
    names = (request.json or {}).get("models", ["random_forest", "svm", "xgboost"])
    try: res = train_all(names)
    except UserError: raise
    return jsonify(res)

@app.route("/api/metrics")
def metrics(): return jsonify(_json("metrics.json") or {})

@app.route("/api/predict", methods=["POST"])
def predict():
    from ml.predict import run_prediction
    return jsonify(run_prediction((request.json or {}).get("model", "random_forest")))

@app.route("/api/prospectivity")
def prospectivity():
    f = OUT / "prospectivity.geojson"
    if not f.exists(): raise UserError("No prospectivity map yet. Run the prediction first.")
    return app.response_class(f.read_text(), mimetype="application/json")

@app.route("/api/download/<name>")
def download(name):
    paths = {"prediction.csv": OUT, "prospectivity.geojson": OUT, "features.csv": PROC}
    if name not in paths or not (paths[name] / name).exists(): raise UserError(f"{name} has not been generated yet.")
    return send_from_directory(paths[name], name, as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
