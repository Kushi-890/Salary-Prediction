# %% [markdown]
# # Data Science Salary Prediction
# Predicts yearly salary (in **USD**) from job details, using the Kaggle *Data Science Job Salaries* dataset.
# Put your CSV at `data/ds_salaries.csv`, then run this notebook (or `python train.py`).

# %%
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path("data/ds_salaries.csv")
OUT_DIR = Path("model")
OUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Load and clean the data

# %%
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)
df.head()

# %%
# Drop the index column and columns we do not use as features.
# salary + salary_currency are in mixed currencies, so we predict salary_in_usd instead.
# employee_residence is almost the same as company_location, so we keep only one.
drop_cols = ["Unnamed: 0", "salary", "salary_currency", "employee_residence"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Remove duplicate rows (done AFTER dropping the index column, so it really works)
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows, {len(df)} remain")
print("Missing values:", int(df.isna().sum().sum()))

# %% [markdown]
# ## 2. Quick data analysis

# %%
print(df["salary_in_usd"].describe().round(0))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df["salary_in_usd"].plot.hist(bins=30, ax=ax[0], title="Salary distribution (USD)")
df.groupby("experience_level")["salary_in_usd"].mean().sort_values().plot.bar(
    ax=ax[1], title="Average salary by experience level (USD)"
)
plt.tight_layout()
fig.savefig(OUT_DIR / "eda.png", dpi=120)

# %% [markdown]
# ## 3. Train and compare models
# Categorical columns are one-hot encoded inside a scikit-learn `Pipeline`,
# so the web app can send plain text values (e.g. `SE`, `Data Scientist`).

# %%
TARGET = "salary_in_usd"
CAT = ["experience_level", "employment_type", "job_title", "company_location", "company_size"]
NUM = ["work_year", "remote_ratio"]
FEATURES = CAT + NUM

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough"
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
}

results, fitted = {}, {}
for name, model in models.items():
    pipe = Pipeline([("prep", clone(preprocess)), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred)),
    }
    fitted[name] = pipe

pd.DataFrame(results).T.round(3)

# %% [markdown]
# ## 4. Pick the best model and check it

# %%
best_name = max(results, key=lambda n: results[n]["R2"])
best = fitted[best_name]
print("Best model:", best_name, results[best_name])

pred = best.predict(X_test)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_test, pred, alpha=0.5)
lims = [0, max(y_test.max(), pred.max())]
ax.plot(lims, lims, "r--")
ax.set_xlabel("Actual salary (USD)")
ax.set_ylabel("Predicted salary (USD)")
ax.set_title(f"{best_name}: actual vs predicted")
fig.savefig(OUT_DIR / "actual_vs_predicted.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# ## 5. Save the model and the dropdown options for the web app

# %%
# Refit the best model on ALL data before saving
final_model = Pipeline([("prep", clone(preprocess)), ("model", clone(models[best_name]))])
final_model.fit(X, y)
joblib.dump(final_model, OUT_DIR / "salary_model.joblib", compress=3)

EXPERIENCE = {"EN": "Entry level", "MI": "Mid level", "SE": "Senior", "EX": "Executive"}
EMPLOYMENT = {"FT": "Full time", "PT": "Part time", "CT": "Contract", "FL": "Freelance"}
SIZE = {"S": "Small (< 50)", "M": "Medium (50-250)", "L": "Large (> 250)"}


def labelled(column, names):
    return [{"value": v, "label": f"{names.get(v, v)} ({v})" if v in names else v}
            for v in sorted(df[column].unique())]


options = {
    "feature_order": FEATURES,
    "experience_level": labelled("experience_level", EXPERIENCE),
    "employment_type": labelled("employment_type", EMPLOYMENT),
    "company_size": labelled("company_size", SIZE),
    "job_title": [{"value": v, "label": v} for v in sorted(df["job_title"].unique())],
    "company_location": [{"value": v, "label": v} for v in sorted(df["company_location"].unique())],
    "work_year": [int(v) for v in sorted(df["work_year"].unique())],
    "remote_ratio": [0, 50, 100],
}
(OUT_DIR / "options.json").write_text(json.dumps(options, indent=2))
(OUT_DIR / "metrics.json").write_text(
    json.dumps({"best_model": best_name, "rows": int(len(df)), "results": results}, indent=2)
)
print("Saved model to", OUT_DIR / "salary_model.joblib")
