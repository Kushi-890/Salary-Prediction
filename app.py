import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).parent
app = Flask(__name__)

model = joblib.load(BASE / "model" / "salary_model.joblib")
options = json.loads((BASE / "model" / "options.json").read_text())
metrics = json.loads((BASE / "model" / "metrics.json").read_text())

FEATURES = options["feature_order"]
NUMERIC = {"work_year", "remote_ratio"}


def allowed_values(field):
    values = options[field]
    return {str(v["value"] if isinstance(v, dict) else v) for v in values}


@app.get("/")
def home():
    return render_template("index.html", options=options, metrics=metrics, form={}, result=None, error=None)


@app.post("/predict")
def predict():
    form = {f: request.form.get(f, "").strip() for f in FEATURES}
    try:
        for field, value in form.items():
            if value not in allowed_values(field):
                raise ValueError(f"Please choose a valid value for {field.replace('_', ' ')}.")
        row = {f: (int(form[f]) if f in NUMERIC else form[f]) for f in FEATURES}
        prediction = float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])
        result = f"${max(prediction, 0):,.0f} per year (USD)"
        return render_template("index.html", options=options, metrics=metrics, form=form, result=result, error=None)
    except ValueError as e:
        return render_template("index.html", options=options, metrics=metrics, form=form, result=None, error=str(e)), 400


if __name__ == "__main__":
    app.run(debug=True)
