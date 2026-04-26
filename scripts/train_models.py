import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OrdinalEncoder
import joblib

DATA_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'processed_flights_utc.csv')
MODELS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stat628_airplanes')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_pipeline():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    # Fill cancellation NAs
    df['Cancelled'] = df['Cancelled'].fillna(0).astype(int)

    print("Feature Engineering...")

    # Her original: CRSDepHour from UTC — kept for reference but NOT used in model
    df['CRSDepTime_UTC'] = pd.to_datetime(df['CRSDepTime_UTC'])
    df['CRSDepHour'] = df['CRSDepTime_UTC'].dt.hour  # kept, not used in features

    # CRSDepHour_local already extracted in preprocess from local CRSDepTime
    # This is what we use in the model — "18:00 local" means rush hour everywhere

    # Taxi averages — computed here so they can be used as features AND saved
    print("Computing Taxi averages...")
    df_comp      = df[df['Cancelled'] == 0].copy()
    taxi_out_avg = df_comp.groupby('Origin')['TaxiOut'].mean().to_dict()
    taxi_in_avg  = df_comp.groupby('Dest')['TaxiIn'].mean().to_dict()
    df['TaxiOut_avg_origin'] = df['Origin'].map(taxi_out_avg)
    df['TaxiIn_avg_dest']    = df['Dest'].map(taxi_in_avg)

    # OD median elapsed time — saved in feature_meta for app auto-fill
    od_elapsed      = df.groupby(['Origin', 'Dest'])['CRSElapsedTime'].median()
    od_elapsed_dict = {(o, d): float(v) for (o, d), v in od_elapsed.items()}

    # Select categorical inputs that app users can change
    cat_cols = ['Origin', 'Dest', 'Reporting_Airline']

    # Feature list — her original features kept, new ones added
    features = [
        'Origin_code', 'Dest_code', 'Airline_code',
        'Month', 'DayOfWeek',
        'CRSDepHour_local',     # local time hour (replaces UTC CRSDepHour in model)
        'WeekOfMonth',          # her feature: which week of the month
        'days_to_holiday',      # her feature: days to nearest holiday
        'is_christmas_window',  # new: fixed Dec20-Jan3 high-travel window
        'CRSElapsedTime',       # new: flight duration (route proxy)
        'elev_diff',            # new: dest_elev - origin_elev (professor recommends)
        'TaxiOut_avg_origin',   # new: historical avg taxi-out at origin
        'TaxiIn_avg_dest',      # new: historical avg taxi-in at dest
    ]

    # 1. Label Encoding
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[['Origin_code', 'Dest_code', 'Airline_code']] = encoder.fit_transform(df[cat_cols])

    # Elevation map for feature_meta (app needs this)
    elev_map_app = {}
    for airport in df['Origin'].unique():
        vals = df[df['Origin'] == airport]['origin_elev']
        if len(vals) > 0 and not pd.isna(vals.iloc[0]):
            elev_map_app[airport] = float(vals.iloc[0])
    for airport in df['Dest'].unique():
        vals = df[df['Dest'] == airport]['dest_elev']
        if len(vals) > 0 and not pd.isna(vals.iloc[0]):
            elev_map_app[airport] = float(vals.iloc[0])

    # Drop rows with missing features
    df_model = df.dropna(subset=features).copy()
    X        = df_model[features]
    print(f"Rows for modelling: {len(df_model)}")

    # ── Cancellation Classifier ───────────────────────────────────────────────
    # HistGBT + CalibratedClassifierCV:
    # - HistGBT replaces RF: no inflated probabilities from class imbalance
    # - CalibratedClassifierCV (isotonic, cv=3): maps raw scores to true
    #   probabilities, preventing the 30-40% false highs we saw before
    print("Training Cancellation Classifier (HistGBT + Calibration)...")
    y_cancelled = df_model['Cancelled']

    base_clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, min_samples_leaf=50, random_state=42
    )
    clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)
    clf.fit(X, y_cancelled)

    sample_probs = clf.predict_proba(X.iloc[:2000])[:, 1]
    print(f"  Cancel rate in data:           {y_cancelled.mean():.4f}")
    print(f"  Predicted prob range (sample): {sample_probs.min():.4f} – {sample_probs.max():.4f}")
    print(f"  Mean predicted prob (sample):  {sample_probs.mean():.4f}")

    # ── Delay models — completed flights only ─────────────────────────────────
    df_completed = df_model[df_model['Cancelled'] == 0].copy()
    df_completed = df_completed.dropna(subset=['DepDelay', 'ArrDelay'])

    # Cap extreme outliers: beyond ±300 min distorts quantile models
    # (raw data has delays up to 3500 min from rare multi-day groundings)
    df_completed['DepDelay_capped'] = df_completed['DepDelay'].clip(-60, 300)
    df_completed['ArrDelay_capped'] = df_completed['ArrDelay'].clip(-90, 300)

    X_completed = df_completed[features]
    print(f"Rows for delay models: {len(df_completed)}")

    # ── Binary late classifiers ───────────────────────────────────────────────
    print("Training is_delayed classifiers (dep/arr > 15 min)...")
    y_dep_late = (df_completed['DepDelay'] > 15).astype(int)
    y_arr_late = (df_completed['ArrDelay'] > 15).astype(int)
    print(f"  Dep >15min rate: {y_dep_late.mean():.4f}")
    print(f"  Arr >15min rate: {y_arr_late.mean():.4f}")

    clf_dep_late = HistGradientBoostingClassifier(
        max_iter=150, max_depth=5, random_state=42
    )
    clf_dep_late.fit(X_completed, y_dep_late)

    clf_arr_late = HistGradientBoostingClassifier(
        max_iter=150, max_depth=5, random_state=42
    )
    clf_arr_late.fit(X_completed, y_arr_late)

    # ── Departure Delay Quantile Regressors ───────────────────────────────────
    print("Training Departure Delay Quantile Regressors...")
    y_dep = df_completed['DepDelay_capped']

    dep_models = {}
    for q in [0.10, 0.50, 0.90]:
        print(f"  -> Training quantile q={q:.2f}")
        hgb = HistGradientBoostingRegressor(
            loss='quantile', quantile=q,
            max_iter=200, max_depth=5,
            min_samples_leaf=30, random_state=42
        )
        hgb.fit(X_completed, y_dep)
        dep_models[f'q_{int(q*100)}'] = hgb

    # ── Arrival Delay Quantile Regressors ─────────────────────────────────────
    print("Training Arrival Delay Quantile Regressors...")
    y_arr = df_completed['ArrDelay_capped']

    arr_models = {}
    for q in [0.10, 0.50, 0.90]:
        print(f"  -> Training quantile q={q:.2f}")
        hgb = HistGradientBoostingRegressor(
            loss='quantile', quantile=q,
            max_iter=200, max_depth=5,
            min_samples_leaf=30, random_state=42
        )
        hgb.fit(X_completed, y_arr)
        arr_models[f'q_{int(q*100)}'] = hgb

    # ── Save all models ───────────────────────────────────────────────────────
    print("Exporting models...")
    joblib.dump(encoder,      os.path.join(MODELS_DIR, 'encoder.joblib'))
    joblib.dump(clf,          os.path.join(MODELS_DIR, 'clf_cancelled.joblib'))
    joblib.dump(clf_dep_late, os.path.join(MODELS_DIR, 'clf_dep_late.joblib'))
    joblib.dump(clf_arr_late, os.path.join(MODELS_DIR, 'clf_arr_late.joblib'))
    joblib.dump(dep_models,   os.path.join(MODELS_DIR, 'reg_dep_delay.joblib'))
    joblib.dump(arr_models,   os.path.join(MODELS_DIR, 'reg_arr_delay.joblib'))
    joblib.dump(
        {'taxi_out': taxi_out_avg, 'taxi_in': taxi_in_avg},
        os.path.join(MODELS_DIR, 'taxi_stats.joblib')
    )
    joblib.dump(
        {
            'feature_cols': features,
            'cat_cols':     cat_cols,
            'airports':     sorted(df['Origin'].unique().tolist()),
            'airlines':     sorted(df['Reporting_Airline'].unique().tolist()),
            'taxi_out_avg': taxi_out_avg,
            'taxi_in_avg':  taxi_in_avg,
            'elev_map':     elev_map_app,
            'od_elapsed':   od_elapsed_dict,
        },
        os.path.join(MODELS_DIR, 'feature_meta.joblib')
    )

    print("Model Training Complete!")
    print("\nModels saved:")
    for f in sorted(os.listdir(MODELS_DIR)):
        size = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024 / 1024
        print(f"  {f:35s} {size:.1f} MB")


if __name__ == "__main__":
    train_pipeline()