"""
Lightweight Web Server & Validated Real-Time ML Inference API
UE24CS352A - Machine Learning Mini-Project
PES University
Authors: Rottela Haritej (PES1UG24AM231) & Pradyun P Urs (PES1UG24AM195)
"""

import http.server
import socketserver
import os
import webbrowser
import sys
import threading
import time
import json
import numpy as np
import pandas as pd
import joblib

PORT = 8000
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEB_DIR)

# -----------------------------------------------------------------------------
# 1. Load Model, Pipeline Metadata, and Real Aligned Historical Prices
# -----------------------------------------------------------------------------
MODELS = {}
MODEL = None
FEATURE_COLS = []
MODEL_INFO = {}
ALIGNED_PRICES = None
HIST_RETURNS = None
N_LAGS = 5
LAG_START = {"EURUSD_X": 1}

try:
    features_path = os.path.join(PROJECT_ROOT, "models", "feature_columns.json")
    info_path = os.path.join(PROJECT_ROOT, "models", "model_info.json")
    prices_path = os.path.join(PROJECT_ROOT, "data", "processed", "aligned_prices.csv")

    if os.path.exists(features_path):
        FEATURE_COLS = json.load(open(features_path))
    if os.path.exists(info_path):
        MODEL_INFO = json.load(open(info_path))

    for m_key, f_name in [
        ("linear_reg", "linear_regression.joblib"),
        ("elastic_net", "elastic_net_active.joblib"),
        ("decision_tree", "decision_tree.joblib"),
        ("elastic_net_sparse", "final_model.joblib"),
    ]:
        p = os.path.join(PROJECT_ROOT, "models", f_name)
        if os.path.exists(p):
            MODELS[m_key] = joblib.load(p)
            print(f"  [OK] Model artifact loaded: {m_key} ({f_name})")

    MODEL = MODELS.get("linear_reg", MODELS.get("elastic_net_sparse"))

    if os.path.exists(prices_path):
        ALIGNED_PRICES = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        HIST_RETURNS = ALIGNED_PRICES.pct_change().iloc[1:]
        print(f"  [OK] Historical Price Series loaded: {len(ALIGNED_PRICES)} rows ({ALIGNED_PRICES.index.min().date()} to {ALIGNED_PRICES.index.max().date()})")
except Exception as e:
    print(f"  [ERROR] Initialization error: {e}")


def construct_full_feature_vector(shock_returns=None):
    """
    Constructs a complete, valid 50-feature vector adhering to the notebook pipeline:
    - Trailing lags 1..4 are extracted from real historical returns.
    - Lag 0 takes the current day's return or user-injected shock return.
    - EUR/USD strictly starts at lag 1 (lag 0 is omitted to prevent lookahead).
    - Strictly validates all 50 features exist and are finite numbers.
    """
    if MODEL is None or not FEATURE_COLS:
        raise RuntimeError("Trained machine learning model artifact is not loaded on server.")
    if HIST_RETURNS is None or len(HIST_RETURNS) < N_LAGS + 2:
        raise RuntimeError("Historical aligned return data is unavailable or insufficient.")

    shock_returns = shock_returns or {}
    norm_shocks = {}
    for k, v in shock_returns.items():
        k_str = str(k).strip()
        norm_shocks[k_str] = v
        norm_shocks[k_str.replace(".", "_")] = v
        norm_shocks[k_str.replace("_", ".")] = v
        norm_shocks[k_str.replace("^", "")] = v
        norm_shocks[k_str.replace("=", "_")] = v
    
    # Base feature dictionary
    feat_dict = {}
    
    # Use the most recent trading days from the historical return series
    # The last row of HIST_RETURNS corresponds to the latest available day t
    latest_date = HIST_RETURNS.index[-1]
    
    for c in HIST_RETURNS.columns:
        s0 = LAG_START.get(c, 0)
        c_clean = c.replace(".", "_").replace("^", "").replace("=", "_")
        for l in range(s0, s0 + N_LAGS):
            col_name = f"{c}_lag{l}"
            if l == 0:
                # If shock provided by user, use it; otherwise use real return on day t
                if c in norm_shocks:
                    val = float(norm_shocks[c]) / 100.0  # convert % to decimal
                elif c_clean in norm_shocks:
                    val = float(norm_shocks[c_clean]) / 100.0
                elif c.replace("_", ".") in norm_shocks:
                    val = float(norm_shocks[c.replace("_", ".")]) / 100.0
                else:
                    val = float(HIST_RETURNS[c].iloc[-1])
            else:
                # Real trailing historical return at lag l (l days prior)
                # iloc[-1 - l] is the real return l days before day t
                val = float(HIST_RETURNS[c].iloc[-1 - l])
            
            feat_dict[col_name] = val

    # Validate against FEATURE_COLS
    vector = []
    missing_cols = []
    for col in FEATURE_COLS:
        if col not in feat_dict:
            missing_cols.append(col)
        else:
            val = feat_dict[col]
            if not np.isfinite(val):
                raise ValueError(f"Feature '{col}' contains non-finite value: {val}")
            vector.append(val)

    if missing_cols:
        raise ValueError(f"Feature vector is missing required pipeline columns: {missing_cols}")

    return np.array(vector).reshape(1, -1), feat_dict, str(latest_date.date())


class ProjectHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path in ["/api/predict", "/api/live"]:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 100000:
                self.send_error_response(400, "Payload too large.")
                return

            post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(post_data.decode('utf-8')) if post_data else {}
            except Exception as e:
                self.send_error_response(400, f"Malformed JSON: {e}")
                return

            try:
                threshold = float(data.get("threshold", 0.0025))
                if not np.isfinite(threshold) or threshold < 0:
                    self.send_error_response(400, "Threshold 'c' must be a non-negative finite float.")
                    return

                # Validate incoming shocks (support 'returns' and 'shocks')
                raw_shocks = data.get("returns", data.get("shocks", {}))
                valid_shocks = {}
                if isinstance(raw_shocks, dict):
                    for k, v in raw_shocks.items():
                        try:
                            val_f = float(v)
                            if not np.isfinite(val_f):
                                self.send_error_response(400, f"Non-finite return provided for {k}: {v}")
                                return
                            valid_shocks[str(k)] = val_f
                        except (ValueError, TypeError):
                            self.send_error_response(400, f"Invalid numeric value for {k}: {v}")
                            return

                # Construct complete 50-feature vector using real pipeline
                X_vec, feature_dict, feature_date = construct_full_feature_vector(valid_shocks)

                chosen_key = data.get("model", "linear_reg")
                if chosen_key not in MODELS:
                    chosen_key = "linear_reg" if "linear_reg" in MODELS else list(MODELS.keys())[0]

                chosen_model = MODELS[chosen_key]

                # Execute trained scikit-learn model
                pred_return = float(chosen_model.predict(X_vec)[0])
                if not np.isfinite(pred_return):
                    raise RuntimeError("Model produced non-finite prediction.")

                pred_pct = pred_return * 100.0
                if pred_return > threshold:
                    signal = "LONG"
                    pos = 1.0
                elif pred_return < -threshold:
                    signal = "SHORT"
                    pos = -1.0
                else:
                    signal = "FLAT"
                    pos = 0.0

                # Compute exact factor attribution for lag0 shock inputs
                attributions = []
                scaler = None
                est = chosen_model
                if hasattr(chosen_model, "named_steps"):
                    scaler = chosen_model.named_steps.get("standardscaler")
                    est = chosen_model[-1]
                elif hasattr(chosen_model, "steps"):
                    est = chosen_model[-1]

                if hasattr(est, "coef_"):
                    coefs = est.coef_
                    scales = scaler.scale_ if (scaler is not None and hasattr(scaler, "scale_")) else np.ones(len(coefs))
                    if len(coefs) == len(FEATURE_COLS):
                        for col_name, c_weight, s_val in zip(FEATURE_COLS, coefs, scales):
                            if "_lag0" in col_name and c_weight != 0:
                                ticker = col_name.replace("_lag0", "")
                                raw_beta = float(c_weight / max(s_val, 1e-8))
                                # Impact in % = (shock in %) * raw_beta
                                att_impact = float((feature_dict[col_name] * 100.0) * raw_beta)
                                attributions.append({
                                    "feature": col_name,
                                    "ticker": ticker,
                                    "input_pct": round(feature_dict[col_name] * 100.0, 2),
                                    "impact_pct": round(att_impact, 4),
                                    "raw_beta": round(raw_beta, 4),
                                    "abs_impact": abs(att_impact)
                                })
                        attributions.sort(key=lambda x: x["abs_impact"], reverse=True)
                elif hasattr(est, "feature_importances_"):
                    imps = est.feature_importances_
                    for col_name, imp in zip(FEATURE_COLS, imps):
                        if "_lag0" in col_name and imp > 0.001:
                            ticker = col_name.replace("_lag0", "")
                            inp = feature_dict[col_name] * 100.0
                            att_impact = inp * imp
                            attributions.append({
                                "feature": col_name,
                                "ticker": ticker,
                                "input_pct": round(inp, 2),
                                "impact_pct": round(att_impact, 4),
                                "abs_impact": abs(att_impact)
                            })
                    attributions.sort(key=lambda x: x["abs_impact"], reverse=True)

                conviction = round(abs(pred_return) / max(threshold, 0.0001), 2) if threshold > 0 else (1.0 if abs(pred_return) > 0 else 0.0)

                model_labels = {
                    "linear_reg": "Linear Regression (Active \u03b2)",
                    "elastic_net": "Elastic Net (Active Regularized)",
                    "decision_tree": "Decision Tree (h=5d)",
                    "elastic_net_sparse": "Elastic Net (Constant Mean / Cash)"
                }

                resp = {
                    "success": True,
                    "model_key": chosen_key,
                    "model": model_labels.get(chosen_key, chosen_key),
                    "horizon_trading_days": 5 if chosen_key == "decision_tree" else 1,
                    "feature_date": feature_date,
                    "predicted_return": pred_return,
                    "predicted_return_pct": round(pred_pct, 4),
                    "threshold": threshold,
                    "signal": signal,
                    "position": pos,
                    "conviction_ratio": conviction,
                    "feature_count": len(FEATURE_COLS),
                    "pipeline_validated": True,
                    "attributions": attributions[:6]
                }

                self.send_json_response(200, resp)
                return

            except Exception as e:
                self.send_error_response(500, f"Inference pipeline execution error: {e}")
                return

        self.send_error_response(404, f"Endpoint not found: {self.path}")

    def do_GET(self):
        clean_path = self.path.split('?', 1)[0].split('#', 1)[0]
        
        # Support GET /api/predict for instant health & latest market prediction
        if clean_path in ["/api/predict", "/api/live"]:
            try:
                X_vec, feature_dict, feature_date = construct_full_feature_vector({})
                pred = float(MODEL.predict(X_vec)[0])
                threshold = 0.0025
                if pred > threshold:
                    sig = "LONG"
                    pos = 1.0
                elif pred < -threshold:
                    sig = "SHORT"
                    pos = -1.0
                else:
                    sig = "FLAT"
                    pos = 0.0
                conviction = round(abs(pred) / max(threshold, 0.0001), 2)
                resp = {
                    "success": True,
                    "status": "ready",
                    "model": MODEL_INFO.get("model", "Elastic Net"),
                    "horizon_trading_days": MODEL_INFO.get("horizon_trading_days", 1),
                    "feature_date": feature_date,
                    "predicted_return": pred,
                    "predicted_return_pct": round(pred * 100.0, 4),
                    "threshold": threshold,
                    "signal": sig,
                    "position": pos,
                    "conviction_ratio": conviction,
                    "feature_count": len(FEATURE_COLS),
                    "pipeline_validated": True
                }
                self.send_json_response(200, resp)
                return
            except Exception as e:
                self.send_error_response(500, str(e))
                return

        elif clean_path == "/api/equity":
            equity_path = os.path.join(PROJECT_ROOT, "results", "holdout_equity_series.json")
            if os.path.exists(equity_path):
                with open(equity_path, "r") as f:
                    data = json.load(f)
                self.send_json_response(200, {"success": True, "equity": data})
            else:
                self.send_error_response(404, "Holdout equity series artifact not found.")
            return

        elif clean_path == "/api/metrics":
            pm_path = os.path.join(PROJECT_ROOT, "results", "prediction_metrics.csv")
            if os.path.exists(pm_path):
                df = pd.read_csv(pm_path)
                hold_df = df[df["period"] == "hold"]
                records = hold_df.to_dict(orient="records")
                self.send_json_response(200, {"success": True, "metrics": records})
            else:
                self.send_error_response(404, "Prediction metrics artifact not found.")
            return

        elif clean_path == "/api/summary":
            sum_path = os.path.join(PROJECT_ROOT, "results", "pipeline_summary.json")
            if os.path.exists(sum_path):
                with open(sum_path, "r") as f:
                    data = json.load(f)
                self.send_json_response(200, {"success": True, "summary": data})
            else:
                self.send_error_response(404, "Pipeline summary artifact not found.")
            return

        return super().do_GET()

    def send_json_response(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))


    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"success": False, "error": message}).encode('utf-8'))

    def translate_path(self, path):
        clean_path = path.split('?', 1)[0].split('#', 1)[0]
        
        # Route root requests to web/index.html
        if clean_path in ["/", "/index.html"]:
            return os.path.join(WEB_DIR, "index.html")
        elif clean_path in ["/style.css", "/app.js"]:
            return os.path.join(WEB_DIR, clean_path.lstrip("/"))
        elif clean_path.startswith("/web/"):
            rel = clean_path[len("/web/"):]
            return os.path.join(WEB_DIR, rel)
        elif clean_path.startswith("/presentation/"):
            rel = clean_path[len("/presentation/"):]
            return os.path.join(PROJECT_ROOT, "presentation", rel)
        elif clean_path.startswith("/reports/"):
            rel = clean_path[len("/reports/"):]
            return os.path.join(PROJECT_ROOT, "reports", rel)
        elif clean_path.startswith("/results/"):
            rel = clean_path[len("/results/"):]
            return os.path.join(PROJECT_ROOT, "results", rel)
        
        candidate = os.path.join(WEB_DIR, clean_path.lstrip("/"))
        if os.path.exists(candidate):
            return candidate
        return os.path.join(PROJECT_ROOT, clean_path.lstrip("/"))

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()


def open_browser(url):
    time.sleep(1.0)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def run():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ProjectHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print("=" * 70)
        print("  ALPHACHAIN QUANTITATIVE TRADING DEMONSTRATION")
        print("  UE24CS352A Machine Learning Mini-Project | PES University")
        print("  Team: Rottela Haritej (PES1UG24AM231) & Pradyun P Urs (PES1UG24AM195)")
        print("=" * 70)
        print(f"\n  [OK] Web Server & ML API Active: {url}")
        print("  [OK] Interactive Dashboard:      http://localhost:8000")
        print("  [OK] Real-time Inference API:    http://localhost:8000/api/predict")
        print("  [OK] Presentation Deck:          http://localhost:8000/presentation/index.html")
        print("  [OK] 2-Page PDF Report:          http://localhost:8000/reports/UE24CS352A_MiniProject_Writeup.pdf")
        print("\n  Opening browser automatically... Press Ctrl+C in terminal to stop.")
        print("=" * 70)

        threading.Thread(target=open_browser, args=(url,), daemon=True).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.shutdown()


if __name__ == "__main__":
    run()
