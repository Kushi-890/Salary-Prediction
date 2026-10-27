# Data Science Salary Predictor

Predicts a yearly data-science salary (in USD) from job title, experience level, employment type,
company location, company size, remote ratio and year. Built with Python, pandas, scikit-learn,
matplotlib and Flask, using the Kaggle *Data Science Job Salaries* dataset.

## What it does
- Cleans the data and predicts `salary_in_usd` (one currency, so errors are meaningful).
- Compares Linear Regression, Decision Tree and Random Forest regressors with RMSE, MAE and R².
- Saves the best model (a scikit-learn Pipeline with one-hot encoding) and serves it with a Flask web app
  that uses dropdowns, so users never type encoded numbers.

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 1. Copy your Kaggle CSV to data/ds_salaries.csv
# 2. Train (or open salary_prediction.ipynb and Run All)
python train.py

# 3. Start the web app
python app.py                   # open http://127.0.0.1:5000
```

## Results
Run `python train.py`; the scores print in the terminal and are saved to `model/metrics.json`.
(Fill these in after you run it.)

| Model | RMSE (USD) | MAE (USD) | R² |
|---|---|---|---|
| Linear Regression | | | |
| Decision Tree | | | |
| Random Forest | | | |

## Project layout
```
train.py / salary_prediction.ipynb   data cleaning, analysis, model training
app.py                               Flask app
templates/index.html                 web form
model/                               saved model, dropdown options, metrics, plots (created by train.py)
data/                                put ds_salaries.csv here
```

## Notes
- The dataset is small, so predictions are estimates, not financial advice.
- Predictions only cover categories that appear in the training data.
