"""
NSAP Scheme Eligibility Prediction - Flask API
Deploy this on IBM Cloud Code Engine (Lite/free tier)
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained artifacts
model = joblib.load("nsap_model.pkl")
scaler = joblib.load("nsap_scaler.pkl")
le = joblib.load("nsap_label_encoder.pkl")
FEATURES = joblib.load("nsap_features.pkl")

SCHEME_NAMES = {
    "IGNOAPS": "Indira Gandhi National Old Age Pension Scheme",
    "IGNWPS": "Indira Gandhi National Widow Pension Scheme",
    "IGNDPS": "Indira Gandhi National Disability Pension Scheme",
}

HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>NSAP Scheme Eligibility Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background:#f5f7fa;}
        h1 { color: #1a3c6e; }
        .card { background: white; padding: 24px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        label { display:block; margin-top: 12px; font-weight: 600; }
        input { width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 6px; }
        button { margin-top: 20px; padding: 10px 20px; background: #1a3c6e; color: white; border: none; border-radius: 6px; cursor: pointer; }
        #result { margin-top: 20px; padding: 14px; border-radius: 6px; background: #eaf3ea; display:none; }
    </style>
</head>
<body>
    <h1>NSAP Scheme Eligibility Predictor</h1>
    <div class="card">
        <p>Enter district-level beneficiary demographic percentages (0-1):</p>
        <label>Male %</label><input id="male_pct" type="number" step="0.01" value="0.5">
        <label>Female %</label><input id="female_pct" type="number" step="0.01" value="0.5">
        <label>Transgender %</label><input id="transgender_pct" type="number" step="0.01" value="0.0">
        <label>SC %</label><input id="sc_pct" type="number" step="0.01" value="0.1">
        <label>ST %</label><input id="st_pct" type="number" step="0.01" value="0.1">
        <label>General %</label><input id="gen_pct" type="number" step="0.01" value="0.6">
        <label>OBC %</label><input id="obc_pct" type="number" step="0.01" value="0.2">
        <label>Aadhaar Coverage %</label><input id="aadhaar_coverage_pct" type="number" step="0.01" value="0.9">
        <label>Mobile Coverage %</label><input id="mobile_coverage_pct" type="number" step="0.01" value="0.7">
        <label>Beneficiary Density (0-1)</label><input id="beneficiary_density" type="number" step="0.01" value="0.3">
        <button onclick="predict()">Predict Scheme</button>
        <div id="result"></div>
    </div>
<script>
async function predict() {
    const payload = {};
    document.querySelectorAll('input').forEach(el => payload[el.id] = parseFloat(el.value));
    const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    const box = document.getElementById('result');
    box.style.display = 'block';
    box.innerHTML = `<b>Predicted Scheme:</b> ${data.scheme_code} — ${data.scheme_name}<br><b>Confidence:</b> ${(data.confidence*100).toFixed(1)}%`;
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HOME_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        x = np.array([[data.get(f, 0) for f in FEATURES]])
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)[0]
        proba = model.predict_proba(x_scaled)[0]
        scheme_code = le.inverse_transform([pred])[0]
        confidence = float(np.max(proba))
        return jsonify({
            "scheme_code": scheme_code,
            "scheme_name": SCHEME_NAMES.get(scheme_code, scheme_code),
            "confidence": confidence,
            "all_probabilities": {
                le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
