# ✈️ AeroPredict: Flight Delay & Cancelation Intelligence
**STAT 628 Module 3 | Spring 2026 | Group 6**

> A machine learning system that predicts U.S. domestic flight delays and cancelations using Q4 2023–2025 DOT data, deployed as an interactive Plotly Dash web application.

---

## 📁 Project Structure

```
m3/
├── stat628_airplanes/          # Raw monthly CSV data (NOT pushed to GitHub - too large)
│   ├── airlines_2023_9.csv     # One file per month (Sep-Dec 2023, 2024; Sep-Nov 2025)
│   ├── airports_timezone.csv   # IANA timezone mapping for airports
│   └── airports_elevation.csv  # Airport elevation and type info
│
├── scripts/
│   ├── preprocess_data.py      # Step 1: Clean raw data → processed_flights_utc.csv
│   └── train_models.py         # Step 2: Train ML models → models/
│
├── models/                     # Trained model artifacts (joblib)
│   ├── encoder.joblib          # OrdinalEncoder for categorical features
│   ├── clf_cancelled.joblib    # RandomForest cancellation classifier
│   ├── reg_dep_delay.joblib    # Quantile regressors for departure delay
│   ├── reg_arr_delay.joblib    # Quantile regressors for arrival delay
│   └── taxi_stats.joblib       # Avg TaxiOut/TaxiIn per airport
│
├── app/
│   ├── app.py                  # Step 3: Plotly Dash web application
│   └── assets/style.css        # Custom CSS for dark theme UI
│
├── processed_flights_utc.csv   # Cleaned dataset (~76MB, 588,993 rows)
├── preprocess_datetime.qmd     # Professor's reference R code for datetime handling
├── airline_problem_canvas.pdf  # Original project specification
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start (New Device Setup)

### Prerequisites
```bash
pip install pandas pytz scikit-learn dash dash-bootstrap-components plotly joblib
```

### Option A: Skip preprocessing (models & processed data already in repo)
```bash
git clone git@github.com:ttsleep/628m3-group6.git
cd 628m3-group6
python app/app.py
# Open http://127.0.0.1:8050 in browser
```

### Option B: Full rebuild from raw data (need the raw CSVs)
```bash
# Step 1: preprocess all raw CSVs → produces processed_flights_utc.csv (~30 min)
python scripts/preprocess_data.py

# Step 2: train all models → produces models/*.joblib (~10 min)
python scripts/train_models.py

# Step 3: launch the web app
python app/app.py
```

---

## 🗂️ Data Scope

| Dimension | Scope |
|-----------|-------|
| **Years** | 2023, 2024, 2025 (Sep–Dec only; 2025 through Nov) |
| **Airlines** | AA (American), DL (Delta), UA (United) |
| **Airports** | ATL, DFW, DEN, ORD, LAX, JFK, LAS, MCO, MIA, CLT, SEA, PHX, EWR, SFO, IAH |
| **Final rows** | 588,993 flights |

---

## 🛠️ How Each Script Works

### `scripts/preprocess_data.py`
- Reads all 11 monthly CSVs, filters to scope (airlines + airports)
- Maps each airport to its IANA timezone using `airports_timezone.csv`
- Converts all departure times from **local time → UTC** using `pytz`
- Computes scheduled arrival UTC using **professor's recommended method**:
  ```
  CRSArrTime_UTC = CRSDepTime_UTC + CRSElapsedTime (minutes)
  ```
  This cleanly handles red-eye flights without date ambiguity
- Exports `processed_flights_utc.csv`

### `scripts/train_models.py`
- Loads processed CSV, encodes categoricals (Origin, Dest, Airline)
- Trains **RandomForestClassifier** → cancellation probability (0–100%)
- Trains **HistGradientBoostingRegressor** with `loss='quantile'` at three levels:
  - q=0.10 (optimistic bound), q=0.50 (median), q=0.90 (pessimistic bound)
  - Applied to both departure and arrival delay
- Computes average TaxiOut per origin airport, TaxiIn per destination
- Exports all models to `models/` as `.joblib` files

### `app/app.py`
- Loads all models into memory at startup
- UI: Origin, Destination (mutually exclusive), Airline, Month, Day of Week, UTC Hour slider
- On "Predict": runs all models and renders:
  - **Gauge chart**: cancellation probability (0–5% scale for sensitivity)
  - **Error bar plot**: median delay + 80% prediction interval (10th–90th percentile)
  - **Taxi insight**: average ground time at origin/destination airports

---

## 📦 Model Features

| Feature | Description |
|---------|-------------|
| `Origin_code` | Encoded origin airport |
| `Dest_code` | Encoded destination airport |
| `Airline_code` | Encoded airline carrier |
| `Month` | Calendar month (9–12) |
| `DayOfWeek` | Day of week (1=Mon … 7=Sun) |
| `CRSDepHour` | Scheduled UTC departure hour (0–23) |

---

## 🔗 Links
- **GitHub Repo**: https://github.com/ttsleep/628m3-group6
- **Local App**: http://127.0.0.1:8050 (after running `python app/app.py`)
