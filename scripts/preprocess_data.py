import os
import glob
import datetime
import pandas as pd
import pytz
from datetime import timedelta
import numpy as np

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stat628_airplanes')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'processed_flights_utc.csv')

# Set scope parameters
TOP_AIRLINES = ['AA', 'DL', 'UA']
TOP_AIRPORTS = [
    'ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'JFK', 'LAS', 'MCO', 
    'MIA', 'CLT', 'SEA', 'PHX', 'EWR', 'SFO', 'IAH'
]

# Essential columns to load to save memory
COLS_TO_USE = [
    'Year', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
    'Reporting_Airline', 'Origin', 'Dest',
    'CRSDepTime', 'DepTime', 'DepDelay', 
    'CRSArrTime', 'ArrTime', 'ArrDelay', 
    'Cancelled', 'CancellationCode',
    'TaxiOut', 'TaxiIn',
    'CRSElapsedTime', 'Distance'
]

def load_timezone_map():
    tz_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'airports_timezone.csv'))
    return dict(zip(tz_df['iata_code'], tz_df['iana_tz']))

def load_elevation_map():
    elev_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'airports_elevation.csv'))
    return dict(zip(elev_df['iata_code'], elev_df['elevation_ft']))

def get_thanksgiving(year):
    """Return date of US Thanksgiving (4th Thursday of November)."""
    nov1 = datetime.date(year, 11, 1)
    days_to_thu = (3 - nov1.weekday()) % 7
    first_thu = nov1 + timedelta(days=days_to_thu)
    return first_thu + timedelta(weeks=3)

def days_to_nearest_holiday(date):
    """Minimum absolute days to Halloween, Thanksgiving, Christmas, NYE."""
    year = date.year
    holidays = [
        datetime.date(year, 10, 31),
        get_thanksgiving(year),
        datetime.date(year, 12, 25),
        datetime.date(year, 12, 31),
    ]
    return min(abs((date - h).days) for h in holidays)

def is_christmas_window(month, day):
    """Dec 20–31 and Jan 1–3: fixed high-travel window, safe to use every year."""
    if month == 12 and day >= 20:
        return 1
    if month == 1 and day <= 3:
        return 1
    return 0

def format_time_str(time_float):
    """Convert float/int time '1755.0' or '5.0' to HHMM string '1755' or '0005'"""
    if pd.isna(time_float):
        return np.nan
    try:
        t_str = str(int(time_float)).zfill(4)
        if t_str == '2400':
            t_str = '0000'
        return t_str
    except Exception:
        return np.nan

def parse_local_dt(date_series, time_series, tz_series):
    """Creates tz-aware UTC datetime pd.Series from local HHMM strings."""
    datetime_strs = date_series + ' ' + time_series.str.slice(0, 2) + ':' + time_series.str.slice(2, 4) + ':00'
    
    temp_df = pd.DataFrame({'dt_str': datetime_strs, 'tz': tz_series})
    temp_df['utc_dt'] = pd.NaT
    
    for tz_name, group in temp_df.groupby('tz'):
        try:
            local_tz = pytz.timezone(tz_name)
            dt_idx = pd.to_datetime(group['dt_str'], errors='coerce')
            localized = dt_idx.dt.tz_localize(local_tz, ambiguous='NaT', nonexistent='NaT')
            utc_converted = localized.dt.tz_convert('UTC')
            temp_df.loc[group.index, 'utc_dt'] = utc_converted
        except Exception as e:
            pass
            
    return temp_df['utc_dt']


def extract_data():
    tz_map   = load_timezone_map()
    elev_map = load_elevation_map()

    all_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, 'airlines_*.csv')))
    
    chunks = []
    
    for i, file in enumerate(all_files):
        print(f"Processing ({i+1}/{len(all_files)}): {os.path.basename(file)}")
        
        df = pd.read_csv(file, usecols=COLS_TO_USE, dtype={
            'CRSDepTime': float, 'DepTime': float,
            'CRSArrTime': float, 'ArrTime': float
        })
        
        # 1. Filter Scope
        df = df[df['Reporting_Airline'].isin(TOP_AIRLINES)]
        df = df[df['Origin'].isin(TOP_AIRPORTS) & df['Dest'].isin(TOP_AIRPORTS)]
        
        if df.empty:
            continue
            
        # 2. Format flight times to HHMM strings
        for col in ['CRSDepTime', 'DepTime', 'CRSArrTime', 'ArrTime']:
            df[col] = df[col].apply(format_time_str)
            
        df = df.dropna(subset=['CRSDepTime', 'CRSArrTime', 'FlightDate'])
        
        # 3. Timezone mapping
        df['Origin_TZ'] = df['Origin'].map(tz_map).fillna('America/New_York')
        df['Dest_TZ']   = df['Dest'].map(tz_map).fillna('America/New_York')

        # 4. CRSDepHour_local — from local CRSDepTime directly (NOT from UTC)
        #    "18:00 local" means evening rush hour regardless of timezone
        df['CRSDepHour_local'] = df['CRSDepTime'].str.slice(0, 2).astype(int, errors='ignore')
        
        # 5. Scheduled Departure UTC
        df['CRSDepTime_UTC'] = parse_local_dt(df['FlightDate'], df['CRSDepTime'], df['Origin_TZ'])
        df['CRSDepTime_UTC'] = pd.to_datetime(df['CRSDepTime_UTC'], utc=True)

        # 6. Professor's red-eye fix: arr_utc = dep_utc + duration
        df['CRSArrTime_UTC'] = df['CRSDepTime_UTC'] + pd.to_timedelta(df['CRSElapsedTime'], unit='m')
        
        # 7. Drop temp timezone columns
        df.drop(columns=['Origin_TZ', 'Dest_TZ'], inplace=True)

        # 8. Elevation features (professor explicitly recommends using elevation)
        df['origin_elev'] = df['Origin'].map(elev_map)
        df['dest_elev']   = df['Dest'].map(elev_map)
        df['elev_diff']   = df['dest_elev'] - df['origin_elev']

        # 9. Christmas/New Year window — fixed dates, no overfitting risk
        df['is_christmas_window'] = df.apply(
            lambda r: is_christmas_window(r['Month'], r['DayofMonth']), axis=1
        )

        # 10. Her original holiday features — kept as-is
        df['WeekOfMonth'] = (df['DayofMonth'] - 1) // 7 + 1
        flight_dates = pd.to_datetime(df['FlightDate']).dt.date
        df['days_to_holiday'] = flight_dates.apply(days_to_nearest_holiday)

        # 11. CancellationCode: recode to readable labels
        code_map = {'A': 'Carrier', 'B': 'Weather', 'C': 'NAS', 'D': 'Security'}
        df['CancellationCode'] = df['CancellationCode'].map(code_map).fillna('None')

        # 12. Cancelled: ensure int
        df['Cancelled'] = df['Cancelled'].fillna(0).astype(int)
        
        chunks.append(df)
        
    print("Concatenating data...")
    final_df = pd.concat(chunks, ignore_index=True)
    
    final_df = final_df.dropna(subset=['CRSDepTime_UTC', 'CRSArrTime_UTC'])
    
    print(f"Exporting to {OUTPUT_FILE} ...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Total rows: {len(final_df)}")
    print("Columns:", final_df.columns.tolist())


if __name__ == "__main__":
    extract_data()