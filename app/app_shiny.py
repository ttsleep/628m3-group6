import os
import joblib
import datetime
from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go
import shinyswatch
from shiny import App, reactive, render, ui

# ── Holiday helpers ─────────────────────────────────────────────────────────────────
def get_thanksgiving(year):
    nov1 = datetime.date(year, 11, 1)
    days_to_thu = (3 - nov1.weekday()) % 7
    first_thu = nov1 + timedelta(days=days_to_thu)
    return first_thu + timedelta(weeks=3)

def days_to_nearest_holiday(date):
    year = date.year
    holidays = [
        datetime.date(year, 10, 31),
        get_thanksgiving(year),
        datetime.date(year, 12, 25),
        datetime.date(year, 12, 31),
    ]
    return min(abs((date - h).days) for h in holidays)

def is_christmas_window(month, day):
    if month == 12 and day >= 20:
        return 1
    if month == 1 and day <= 3:
        return 1
    return 0

# ── Load models ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

encoder        = joblib.load(os.path.join(MODELS_DIR, 'encoder.joblib'))
clf_cancelled  = joblib.load(os.path.join(MODELS_DIR, 'clf_cancelled.joblib'))
reg_dep_delay  = joblib.load(os.path.join(MODELS_DIR, 'reg_dep_delay.joblib'))
reg_arr_delay  = joblib.load(os.path.join(MODELS_DIR, 'reg_arr_delay.joblib'))
taxi_stats     = joblib.load(os.path.join(MODELS_DIR, 'taxi_stats.joblib'))
meta           = joblib.load(os.path.join(MODELS_DIR, 'feature_meta.joblib'))

# ── Constants ─────────────────────────────────────────────────────────────────
AIRPORTS = ['ATL','DFW','DEN','ORD','LAX','JFK','LAS','MCO',
            'MIA','CLT','SEA','PHX','EWR','SFO','IAH']
AIRLINES = {'AA': 'American Airlines (AA)',
            'DL': 'Delta Air Lines (DL)',
            'UA': 'United Airlines (UA)'}
MONTHS   = {'9':'September','10':'October','11':'November','12':'December'}
DAYS     = {'1':'Monday','2':'Tuesday','3':'Wednesday','4':'Thursday',
            '5':'Friday','6':'Saturday','7':'Sunday'}

# ── UI ────────────────────────────────────────────────────────────────────────
_header = ui.div(
    ui.h1("✈  AeroPredict", style="margin:0; color:#4ea8de; font-weight:700;"),
    ui.p("Flight Delay & Cancelation Intelligence · STAT 628 Group 6",
         style="color:#adb5bd; margin:0;"),
    style="padding:24px 32px 12px; border-bottom:1px solid #2e3338;",
)

_input_card = ui.card(
    ui.card_header("Flight Itinerary"),
    ui.layout_columns(
        ui.input_select("origin", "Origin Airport",
                        choices=AIRPORTS, selected="LAX"),
        ui.input_select("dest",   "Destination Airport",
                        choices=AIRPORTS, selected="JFK"),
        col_widths=[6, 6],
    ),
    ui.layout_columns(
        ui.input_select("airline", "Airline",
                        choices=AIRLINES, selected="DL"),
        ui.input_date("flight_date", "Flight Date",
                      value="2025-12-15",
                      min="2023-09-01", max="2025-12-31"),
        col_widths=[6, 6],
    ),
    ui.input_slider("hour", "Scheduled Departure Hour (local time)",
                    min=0, max=23, value=12, step=1),
    ui.input_action_button("predict", "⚡  Predict Now",
                           class_="btn-primary w-100 mt-2"),
)

_output_row = ui.layout_columns(
    ui.card(
        ui.card_header("Cancellation Risk"),
        ui.output_ui("gauge_plot"),
    ),
    ui.card(
        ui.card_header("Delay Predictions & Uncertainties (min)"),
        ui.output_ui("delay_plot"),
    ),
    col_widths=[5, 7],
)

_taxi_card = ui.card(
    ui.card_header("Runway Taxi Insight"),
    ui.output_text("taxi_text"),
    style="color:#0dcaf0;",
)

app_ui = ui.page_fillable(
    _header,
    ui.layout_columns(
        _input_card,
        ui.layout_columns(
            _output_row,
            _taxi_card,
            col_widths=[12, 12],
        ),
        col_widths=[4, 8],
    ),
    padding="16px",
    theme=shinyswatch.theme.darkly,
)

# ── Server ────────────────────────────────────────────────────────────────────
def server(input, output, session):

    # Reactive: keep dest choices in sync with origin (no A→A)
    @reactive.effect
    def _sync_dest():
        origin = input.origin()
        choices = [a for a in AIRPORTS if a != origin]
        # If current dest == origin, reset
        current = input.dest()
        new_val = current if current != origin else choices[0]
        ui.update_select("dest", choices=choices, selected=new_val)

    # ── Core prediction (computed once per button click) ──────────────────────
    @reactive.calc
    @reactive.event(input.predict, ignore_none=False)
    def predictions():
        origin  = input.origin()
        dest    = input.dest()
        airline = input.airline()
        hour    = int(input.hour())

        # Auto-derive month and day-of-week from the date picker
        import datetime
        d = input.flight_date()          # returns a datetime.date object
        if isinstance(d, str):
            d = datetime.date.fromisoformat(d)
        month = d.month                  # 9-12
        dow   = d.isoweekday()           # 1=Mon … 7=Sun

        # Validate: model only trained on Sep-Dec
        if month not in range(9, 13):
            raise ValueError(f"Month {d.strftime('%B')} is outside our training scope (Sep–Dec only).")

        # Encode categoricals
        cat_df = pd.DataFrame({'Origin': [origin],
                               'Dest':   [dest],
                               'Reporting_Airline': [airline]})
        try:
            enc = encoder.transform(cat_df)[0]
        except Exception:
            enc = [-1, -1, -1]

        elev_map   = meta.get('elev_map', {})
        od_elapsed = meta.get('od_elapsed', {})
        
        elapsed = od_elapsed.get((origin, dest), 150.0)
        elev_diff = elev_map.get(dest, 0) - elev_map.get(origin, 0)
        t_out = taxi_stats['taxi_out'].get(origin, 20.0)
        t_in  = taxi_stats['taxi_in'].get(dest, 10.0)

        X = pd.DataFrame({
            'Origin_code':        [enc[0]],
            'Dest_code':          [enc[1]],
            'Airline_code':       [enc[2]],
            'Month':              [month],
            'DayOfWeek':          [dow],
            'CRSDepHour_local':   [hour],
            'WeekOfMonth':        [(d.day - 1) // 7 + 1],
            'days_to_holiday':    [days_to_nearest_holiday(d)],
            'is_christmas_window':[is_christmas_window(month, d.day)],
            'CRSElapsedTime':     [elapsed],
            'elev_diff':          [elev_diff],
            'TaxiOut_avg_origin': [t_out],
            'TaxiIn_avg_dest':    [t_in],
        })

        cancel_pct = clf_cancelled.predict_proba(X)[0][1] * 100

        dep = [reg_dep_delay[f'q_{q}'].predict(X)[0] for q in [10, 50, 90]]
        arr = [reg_arr_delay[f'q_{q}'].predict(X)[0] for q in [10, 50, 90]]

        t_out = taxi_stats['taxi_out'].get(origin, 0)
        t_in  = taxi_stats['taxi_in'].get(dest, 0)

        return dict(cancel=cancel_pct, dep=dep, arr=arr, t_out=t_out, t_in=t_in,
                    origin=origin, dest=dest)

    # ── Gauge ─────────────────────────────────────────────────────────────────
    @render.ui
    def gauge_plot():
        p = predictions()
        val = p['cancel'] if p else 0

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = val,
            number = {'valueformat': '.2f', 'suffix': "%"},
            title  = {'text': "Cancelation Probability"},
            gauge  = {
                'axis': {'range': [0, 5], 'tickcolor': 'white'},
                'bar':  {'color': 'white', 'thickness': 0.25},
                'bgcolor': 'rgba(0,0,0,0)',
                'steps': [
                    {'range': [0,   1.0], 'color': '#006400'},
                    {'range': [1.0, 2.5], 'color': '#B8860B'},
                    {'range': [2.5, 5.0], 'color': '#8B0000'},
                ],
            },
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          font={'color': 'white'}, height=280)
        html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        return ui.HTML(html)

    # ── Delay error-bar plot ──────────────────────────────────────────────────
    @render.ui
    def delay_plot():
        p = predictions()
        if not p:
            dep = [0, 0, 0]
            arr = [0, 0, 0]
        else:
            dep = p['dep']
            arr = p['arr']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Departure Delay", x=["Departure"], y=[dep[1]],
            error_y=dict(type='data', symmetric=False,
                         array=[dep[2]-dep[1]], arrayminus=[dep[1]-dep[0]],
                         color='rgba(255,159,64,1)', thickness=3, width=20),
            marker=dict(color='rgba(255,159,64,1)', size=14), mode="markers",
        ))
        fig.add_trace(go.Scatter(
            name="Arrival Delay", x=["Arrival"], y=[arr[1]],
            error_y=dict(type='data', symmetric=False,
                         array=[arr[2]-arr[1]], arrayminus=[arr[1]-arr[0]],
                         color='rgba(153,102,255,1)', thickness=3, width=20),
            marker=dict(color='rgba(153,102,255,1)', size=14), mode="markers",
        ))
        fig.update_layout(
            yaxis_title="Minutes (negative = early)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            yaxis=dict(gridcolor='#444', zerolinecolor='white'),
            height=280, legend=dict(orientation='h', y=1.12),
        )
        html = fig.to_html(full_html=False, include_plotlyjs=False)
        return ui.HTML(html)

    # ── Taxi text ─────────────────────────────────────────────────────────────
    @render.text
    def taxi_text():
        p = predictions()
        if not p:
            return "Fill in the form and click Predict to see taxi insights."
        return (
            f"✈  At {p['origin']}, average Taxi-Out is {p['t_out']:.1f} min "
            f"before the plane leaves the gate. "
            f"At {p['dest']}, expect {p['t_in']:.1f} min Taxi-In after landing."
        )


app = App(app_ui, server)
