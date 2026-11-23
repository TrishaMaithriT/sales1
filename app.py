"""
app.py — SalesCast Multi-Page Sales Prediction System
======================================================
Page Routes:
  GET  /             -> Home (landing page)
  GET  /upload       -> Upload Data page
  GET  /dashboard    -> Dashboard (charts & stats)
  GET  /forecast     -> Sales Forecast page
  GET  /about        -> About page

API Routes:
  POST /api/upload      -> Upload CSV, train model
  POST /api/use_sample  -> Train on built-in sample CSV
  POST /api/predict     -> Forecast future months
  GET  /api/session_data -> Return session data for JS
"""

import os, io, pickle, uuid, datetime
import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "salescast_multipage_2024"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ── Model persistence via /tmp (works locally AND on Netlify/serverless) ──
_TMP = "/tmp" if os.path.exists("/tmp") else os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(_TMP, exist_ok=True)

def _save_model(model_id, store):
    with open(os.path.join(_TMP, f"mdl_{model_id}.pkl"), "wb") as f:
        pickle.dump(store, f)

def _load_model(model_id):
    if not model_id:
        return None
    path = os.path.join(_TMP, f"mdl_{model_id}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ═══════════════════════════════════════════════════
#  ML HELPERS
# ═══════════════════════════════════════════════════

def parse_csv(file_obj):
    df = pd.read_csv(file_obj)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def detect_columns(df):
    date_col = sales_col = None
    date_kw  = ["date", "month", "year", "period", "time"]
    sales_kw = ["sales", "revenue", "amount", "total", "value", "quantity", "units"]
    for col in df.columns:
        if any(k in col for k in date_kw)  and date_col  is None: date_col  = col
        if any(k in col for k in sales_kw) and sales_col is None: sales_col = col
    if date_col  is None: date_col  = df.columns[0]
    if sales_col is None: sales_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    return date_col, sales_col

def build_features(series):
    X, y = [], []
    for i in range(3, len(series)):
        X.append([i, series[i-1], series[i-2], series[i-3], np.mean(series[i-3:i])])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_and_store(sales, labels, date_col, sales_col):
    """Train model on sales array, persist to /tmp, update session."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(sales.reshape(-1, 1)).flatten()

    X, y  = build_features(scaled)
    split = max(1, int(len(X) * 0.8))
    model = LinearRegression()
    model.fit(X[:split], y[:split])

    y_pred_sc   = model.predict(X[split:])
    y_test_act  = scaler.inverse_transform(y[split:].reshape(-1, 1)).flatten()
    y_test_pred = scaler.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()

    mae  = float(mean_absolute_error(y_test_act, y_test_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test_act, y_test_pred)))
    r2   = float(r2_score(y_test_act, y_test_pred))
    if r2 != r2: r2 = 0.0  # nan guard

    all_pred = scaler.inverse_transform(
        model.predict(X).reshape(-1, 1)).flatten().tolist()

    # Save model to /tmp
    model_id = str(uuid.uuid4())
    _save_model(model_id, {
        "model":      model,
        "scaler":     scaler,
        "last_sales": scaled[-3:].tolist(),
        "last_index": len(sales) - 1,
    })

    # Store everything in session
    session["trained"]    = True
    session["model_id"]   = model_id
    session["metrics"]    = {"mae": round(mae,2), "rmse": round(rmse,2), "r2": round(r2*100,2)}
    session["full_actual"]= {"labels": labels, "values": [round(v,2) for v in sales.tolist()]}
    session["historical"] = {
        "labels":    labels[3:],
        "actual":    [round(v,2) for v in sales[3:].tolist()],
        "predicted": [round(v,2) for v in all_pred],
    }
    session["row_count"]  = len(sales)
    session["date_col"]   = date_col
    session["sales_col"]  = sales_col


# ═══════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("home.html", active="home")

@app.route("/upload")
def upload_page():
    return render_template("upload.html", active="upload")

@app.route("/dashboard")
def dashboard():
    if not session.get("trained"):
        return redirect(url_for("upload_page"))
    data = {
        "metrics":     session.get("metrics"),
        "full_actual": session.get("full_actual"),
        "historical":  session.get("historical"),
        "row_count":   session.get("row_count"),
        "date_col":    session.get("date_col"),
        "sales_col":   session.get("sales_col"),
    }
    return render_template("dashboard.html", active="dashboard", data=data)

@app.route("/forecast")
def forecast_page():
    if not session.get("trained"):
        return redirect(url_for("upload_page"))
    return render_template("forecast.html", active="forecast",
                           metrics=session.get("metrics"),
                           full_actual=session.get("full_actual"))

@app.route("/about")
def about():
    return render_template("about.html", active="about")


# ═══════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400
    try:
        df = parse_csv(io.StringIO(file.stream.read().decode("utf-8")))
    except Exception as e:
        return jsonify({"error": f"Cannot read CSV: {e}"}), 400
    if len(df) < 6:
        return jsonify({"error": "Need at least 6 rows of data."}), 400

    date_col, sales_col = detect_columns(df)
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        labels = df[date_col].dt.strftime("%b %Y").tolist()
    except Exception:
        labels = [str(i + 1) for i in range(len(df))]

    sales = df[sales_col].astype(float).values
    if len(build_features(MinMaxScaler().fit_transform(
            sales.reshape(-1,1)).flatten())[0]) < 4:
        return jsonify({"error": "Need at least 7 rows to train."}), 400

    train_and_store(sales, labels, date_col, sales_col)
    return jsonify({
        "status":      "ok",
        "redirect":    url_for("dashboard"),
        "metrics":     session["metrics"],
        "full_actual": session["full_actual"],
        "historical":  session["historical"],
        "row_count":   session["row_count"],
    })


@app.route("/api/use_sample", methods=["POST"])
def api_use_sample():
    sample_path = os.path.join(os.path.dirname(__file__), "static", "sample_sales.csv")
    try:
        df = parse_csv(open(sample_path, "r"))
    except Exception as e:
        return jsonify({"error": f"Could not load sample data: {e}"}), 500

    date_col, sales_col = detect_columns(df)
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        labels = df[date_col].dt.strftime("%b %Y").tolist()
    except Exception:
        labels = [str(i + 1) for i in range(len(df))]

    sales = df[sales_col].astype(float).values
    train_and_store(sales, labels, date_col, sales_col)

    return jsonify({
        "status":    "ok",
        "redirect":  url_for("dashboard"),
        "metrics":   session["metrics"],
        "row_count": session["row_count"],
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    model_id = session.get("model_id")
    store    = _load_model(model_id)
    if store is None:
        return jsonify({"error": "No trained model found. Please train first."}), 400

    body   = request.get_json(silent=True) or {}
    months = max(1, min(int(body.get("months", 6)), 24))

    model, scaler = store["model"], store["scaler"]
    lag  = list(store["last_sales"])
    idx  = store["last_index"]

    preds_sc = []
    for _ in range(months):
        idx += 1
        feat = [idx, lag[-1], lag[-2], lag[-3], np.mean(lag[-3:])]
        p    = model.predict([feat])[0]
        preds_sc.append(p)
        lag.append(p)

    preds = scaler.inverse_transform(np.array(preds_sc).reshape(-1, 1)).flatten()
    preds = [round(max(0.0, v), 2) for v in preds]

    now   = datetime.date.today().replace(day=1)
    flbls = []
    for i in range(1, months + 1):
        m = (now.month + i - 1) % 12 + 1
        y = now.year + (now.month + i - 1) // 12
        flbls.append(datetime.date(y, m, 1).strftime("%b %Y"))

    return jsonify({"status": "ok", "months": months, "labels": flbls, "predictions": preds})


@app.route("/api/session_data")
def api_session_data():
    return jsonify({
        "trained":     session.get("trained", False),
        "full_actual": session.get("full_actual"),
        "historical":  session.get("historical"),
        "metrics":     session.get("metrics"),
    })


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    print("\n" + "="*60)
    print(f"  SalesCast  |  http://127.0.0.1:{port}")
    print("="*60 + "\n")
    app.run(debug=debug, host="0.0.0.0", port=port)
