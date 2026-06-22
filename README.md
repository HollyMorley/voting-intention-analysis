# UK Voting Intention — analysis & dashboard

A take-home built from a survey of **1,468 UK respondents**. Two parts:

- **Task 1 — analysis.** Clean the data, find what matters, and build a simple,
  validated model that predicts vote intention. Write-up with figures:
  [`report/writeup.md`](report/writeup.md). Full working:
  [`analysis/analysis2.ipynb`](analysis/analysis2.ipynb).
- **Task 2 — dashboard.** A lightweight, interactive web app for exploring the findings
  and the model: [`docs/`](docs/). It's a static site (HTML + Plotly + pre-computed
  JSON) — no backend, ready for GitHub Pages.

**[📊 View the live dashboard](https://hollymorley.github.io/voting-intention-analysis/)**

## Run the dashboard locally

**Just open `docs/index.html` in a browser** (double-click it). 

## Requirements

Needs Python 3 with the packages in `requirements.txt`
(`pip install -r requirements.txt`).

```bash
# 1. regenerate the dashboard's data from the survey + model
python analysis/export_dashboard.py     # -> docs/data/*.json + docs/data.js

# 2. regenerate the write-up figures
python analysis/make_figures.py         # -> report/figures/*.png
```

These scripts reproduce the exact pipeline in `analysis/analysis2.ipynb`, so the dashboard
and the write-up always agree. If you change the analysis (e.g. edit the model in
`analysis/model.py` or the feature groups in `analysis/config.py`), re-run the two scripts
to refresh the outputs.

## Repo layout

```
analysis/
  analysis2.ipynb       full analysis narrative (EDA -> model)
  clean.py  encode.py   cleaning + feature engineering
  model.py  config.py   model definitions + the curated feature taxonomy
  helpers.py            labels + small shared utilities
  export_dashboard.py   builds docs/data/*.json + docs/data.js
  make_figures.py       builds report/figures/*.png
docs/                   the dashboard (deploy this folder)
  index.html  app.js  style.css
  data.js               bundled survey + model outputs (what the page loads)
  data/*.json           the same outputs as raw JSON
  vendor/plotly*.js     charting library, local copy (works offline)
report/
  writeup.md            Task 1 write-up
  figures/*.png
data/                   raw + cleaned survey data
```