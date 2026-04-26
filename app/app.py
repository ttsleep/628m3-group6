import os
import joblib
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ── Holiday helpers ───────────────────────────────────────────────────────────
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

def get_holiday_notice(date):
    d2h = days_to_nearest_holiday(date)
    if d2h == 0:
        return "🎄 Today is a major holiday — expect significantly higher delays and cancellation risk."
    elif d2h <= 3:
        return f"⚠️ This date is within {d2h} day(s) of a major US holiday — expect elevated delays and cancellation risk."
    elif d2h <= 7:
        return f"📅 This date is {d2h} days from a major US holiday — some increase in delays is expected."
    return ""

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "AeroPredict: Flight Intelligence"

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

# ── Load Models ───────────────────────────────────────────────────────────────
encoder       = joblib.load(os.path.join(MODELS_DIR, 'encoder.joblib'))
clf_cancelled = joblib.load(os.path.join(MODELS_DIR, 'clf_cancelled.joblib'))
clf_dep_late  = joblib.load(os.path.join(MODELS_DIR, 'clf_dep_late.joblib'))
clf_arr_late  = joblib.load(os.path.join(MODELS_DIR, 'clf_arr_late.joblib'))
reg_dep_delay = joblib.load(os.path.join(MODELS_DIR, 'reg_dep_delay.joblib'))
reg_arr_delay = joblib.load(os.path.join(MODELS_DIR, 'reg_arr_delay.joblib'))
taxi_stats    = joblib.load(os.path.join(MODELS_DIR, 'taxi_stats.joblib'))
meta          = joblib.load(os.path.join(MODELS_DIR, 'feature_meta.joblib'))

ELEV_MAP   = meta.get('elev_map', {})
OD_ELAPSED = meta.get('od_elapsed', {})

def get_elapsed(origin, dest):
    val = OD_ELAPSED.get((origin, dest))
    if val is not None:
        return float(val)
    return 150.0

# ── Options ───────────────────────────────────────────────────────────────────
AIRPORTS = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'JFK', 'LAS', 'MCO',
            'MIA', 'CLT', 'SEA', 'PHX', 'EWR', 'SFO', 'IAH']
AIRLINES = ['AA', 'DL', 'UA']

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # Hidden store to track which slider was last changed
    dcc.Store(id='last-changed-slider', data='dep'),

    # Header
    dbc.Row([dbc.Col([
        html.H1("AeroPredict: Flight Delay & Cancelation Intelligence",
                className="text-center mt-4 mb-2 text-primary font-weight-bold"),
        html.H5("Stat 628 Airplane Project", className="text-center text-muted mb-4")
    ], width=12)]),

    # Input panel
    dbc.Row([dbc.Col([dbc.Card([
        dbc.CardHeader("Flight Itinerary Specification", className="font-weight-bold"),
        dbc.CardBody([

            # Row 1: Origin / Dest
            dbc.Row([
                dbc.Col([
                    html.Label("Origin Airport"),
                    dcc.Dropdown(id='origin-dp',
                        options=[{'label': a, 'value': a} for a in AIRPORTS],
                        value='LAX', clearable=False, className="text-dark mb-3")
                ], width=6),
                dbc.Col([
                    html.Label("Destination Airport"),
                    dcc.Dropdown(id='dest-dp',
                        options=[{'label': a, 'value': a} for a in AIRPORTS],
                        value='JFK', clearable=False, className="text-dark mb-3")
                ], width=6),
            ]),

            # Row 2: Airline / Date
            dbc.Row([
                dbc.Col([
                    html.Label("Airline Carrier"),
                    dcc.Dropdown(id='airline-dp',
                        options=[{'label': a, 'value': a} for a in AIRLINES],
                        value='DL', clearable=False, className="text-dark mb-3")
                ], width=4),
                dbc.Col([
                    html.Label("Flight Date"),
                    dbc.Input(id='date-picker', type='date', value='2025-12-15',
                        min='2023-09-01', max='2025-12-31', className="mb-3")
                ], width=8),
            ]),

            # Holiday notice
            dbc.Row([dbc.Col([
                html.Div(id='holiday-notice', className="text-warning fst-italic mb-2")
            ], width=12)]),

            # Row 3: Dep / Arr sliders (local time, bidirectional)
            dbc.Row([
                dbc.Col([
                    html.Label("Scheduled Departure Hour — local time (0–23)"),
                    dcc.Slider(id='dep-hour-slider', min=0, max=23, step=1, value=10,
                        marks={i: str(i) for i in range(0, 24, 3)},
                        tooltip={"placement": "bottom", "always_visible": False}),
                    html.Div(id='dep-hour-display',
                        className="text-info text-center mt-1 mb-3"),
                ], width=6),
                dbc.Col([
                    html.Label("Scheduled Arrival Hour — local time (0–23)"),
                    dcc.Slider(id='arr-hour-slider', min=0, max=23, step=1, value=13,
                        marks={i: str(i) for i in range(0, 24, 3)},
                        tooltip={"placement": "bottom", "always_visible": False}),
                    html.Div(id='arr-hour-display',
                        className="text-info text-center mt-1 mb-3"),
                ], width=6),
            ]),

            # Predict button
            dbc.Row([dbc.Col([
                dbc.Button("Predict Intelligence", id="predict-btn",
                    color="primary", size="lg", className="w-100")
            ], width=12)])
        ])
    ], className="mb-4 shadow-sm border-0")], width=12)]),

    # Output row 1: Cancellation gauge + Late probability
    dbc.Row([
        dbc.Col([dbc.Card([
            dbc.CardHeader("Cancellation Risk"),
            dbc.CardBody([dcc.Graph(id='cancel-gauge')])
        ], className="h-100 shadow-sm border-0")], width=4),

        dbc.Col([dbc.Card([
            dbc.CardHeader("Probability of Significant Delay (> 15 min)"),
            dbc.CardBody([dcc.Graph(id='late-prob-chart')])
        ], className="h-100 shadow-sm border-0")], width=8),
    ], className="mb-4"),

    # Output row 2: Delay interval chart
    dbc.Row([dbc.Col([dbc.Card([
        dbc.CardHeader("Delay Predictions & Uncertainties (Minutes)"),
        dbc.CardBody([dcc.Graph(id='delay-box')])
    ], className="shadow-sm border-0")], width=12)], className="mb-4"),

    # Output row 3: Taxi insight
    dbc.Row([dbc.Col([dbc.Card([
        dbc.CardHeader("Runway Taxi Insight"),
        dbc.CardBody([html.P(id="taxi-text", className="lead text-info")])
    ], className="shadow-sm border-0")], width=12)], className="mb-4"),

], fluid=True, className="p-5")


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('dest-dp', 'options'),
    Output('dest-dp', 'value'),
    Input('origin-dp', 'value'),
    State('dest-dp', 'value')
)
def disable_same_airport(origin, dest):
    opts = [{'label': a, 'value': a} for a in AIRPORTS if a != origin]
    val  = dest if dest != origin else opts[0]['value']
    return opts, val


@app.callback(
    Output('holiday-notice', 'children'),
    Input('date-picker', 'value')
)
def show_holiday_notice(flight_date):
    if not flight_date:
        return ""
    try:
        d = datetime.date.fromisoformat(flight_date)
        return get_holiday_notice(d)
    except Exception:
        return ""


@app.callback(
    Output('dep-hour-slider', 'value'),
    Output('arr-hour-slider', 'value'),
    Input('dep-hour-slider', 'value'),
    Input('arr-hour-slider', 'value'),
    Input('origin-dp', 'value'),
    Input('dest-dp', 'value'),
    prevent_initial_call=True,
)
def sync_sliders(dep_hour, arr_hour, origin, dest):
    elapsed_hours = round(get_elapsed(origin, dest) / 60)

    ctx = dash.callback_context
    if not ctx.triggered:
        return dep_hour, arr_hour

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'dep-hour-slider':
        # 出发变 → 到达跟着动
        new_arr = int((dep_hour + elapsed_hours) % 24)
        return dep_hour, new_arr

    elif triggered_id == 'arr-hour-slider':
        # 到达变 → 出发跟着动
        new_dep = int((arr_hour - elapsed_hours) % 24)
        return new_dep, arr_hour

    else:
        # 航线变了 → 出发不变，到达重新计算
        new_arr = int((dep_hour + elapsed_hours) % 24)
        return dep_hour, new_arr


@app.callback(
    Output('dep-hour-display', 'children'),
    Input('dep-hour-slider', 'value')
)
def show_dep_hour(h):
    return f"Departure: {h:02d}:00 local time"


@app.callback(
    Output('arr-hour-display', 'children'),
    Input('arr-hour-slider', 'value')
)
def show_arr_hour(h):
    return f"Arrival: {h:02d}:00 local time"


@app.callback(
    Output('cancel-gauge',    'figure'),
    Output('late-prob-chart', 'figure'),
    Output('delay-box',       'figure'),
    Output('taxi-text',       'children'),
    Input('predict-btn', 'n_clicks'),
    State('origin-dp',       'value'),
    State('dest-dp',         'value'),
    State('airline-dp',      'value'),
    State('date-picker',     'value'),
    State('dep-hour-slider', 'value'),
    prevent_initial_call=True,
)
def update_predictions(n_clicks, origin, dest, airline, flight_date, dep_hour):

    d     = datetime.date.fromisoformat(flight_date) if flight_date else datetime.date(2025, 12, 15)
    month = d.month
    day   = d.isoweekday()  # 1=Mon … 7=Sun

    if month not in range(9, 13):
        warn_msg = (f"⚠️ Month {d.strftime('%B')} is outside our training data scope "
                    f"(September – December only).")
        empty_fig = go.Figure().update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={'color': 'white'},
            annotations=[dict(text=warn_msg, showarrow=False,
                              font=dict(color='#ffc107', size=14),
                              xref='paper', yref='paper', x=0.5, y=0.5)]
        )
        return empty_fig, empty_fig, empty_fig, warn_msg

    cat_df = pd.DataFrame({'Origin': [origin], 'Dest': [dest], 'Reporting_Airline': [airline]})
    try:
        encoded = encoder.transform(cat_df)[0]
    except Exception:
        encoded = [-1, -1, -1]

    elapsed   = get_elapsed(origin, dest)
    elev_diff = ELEV_MAP.get(dest, 0) - ELEV_MAP.get(origin, 0)
    t_out_avg = taxi_stats['taxi_out'].get(origin, 20.0)
    t_in_avg  = taxi_stats['taxi_in'].get(dest,   10.0)

    X = pd.DataFrame({
        'Origin_code':        [encoded[0]],
        'Dest_code':          [encoded[1]],
        'Airline_code':       [encoded[2]],
        'Month':              [month],
        'DayOfWeek':          [day],
        'CRSDepHour_local':   [dep_hour],
        'WeekOfMonth':        [(d.day - 1) // 7 + 1],
        'days_to_holiday':    [days_to_nearest_holiday(d)],
        'is_christmas_window':[is_christmas_window(month, d.day)],
        'CRSElapsedTime':     [elapsed],
        'elev_diff':          [elev_diff],
        'TaxiOut_avg_origin': [t_out_avg],
        'TaxiIn_avg_dest':    [t_in_avg],
    })

    # ── 1. Cancellation gauge ─────────────────────────────────────────────────
    cancel_prob = float(clf_cancelled.predict_proba(X)[0][1] * 100)
    gauge_max   = max(2.0, round(cancel_prob * 2.5, 1))

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cancel_prob,
        number={'valueformat': '.2f', 'suffix': "%"},
        title={'text': "Cancelation Probability (%)", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, gauge_max], 'tickwidth': 1, 'tickcolor': 'white'},
            'bar':  {'color': 'white', 'thickness': 0.25},
            'bgcolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [0,               gauge_max * 0.3], 'color': '#006400'},
                {'range': [gauge_max * 0.3, gauge_max * 0.6], 'color': '#B8860B'},
                {'range': [gauge_max * 0.6, gauge_max],       'color': '#8B0000'},
            ]
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'},
        margin=dict(t=60, b=20, l=20, r=20)
    )

    # ── 2. Late probability bars ──────────────────────────────────────────────
    dep_late_pct = float(clf_dep_late.predict_proba(X)[0][1] * 100)
    arr_late_pct = float(clf_arr_late.predict_proba(X)[0][1] * 100)

    fig_late = go.Figure()
    for label, val, color in [
        ("Departure > 15 min late", dep_late_pct, '#FF9F40'),
        ("Arrival > 15 min late",   arr_late_pct, '#9966FF'),
    ]:
        fig_late.add_trace(go.Bar(
            x=[val], y=[label], orientation='h',
            marker_color=color,
            text=[f"{val:.1f}%"], textposition='outside',
            width=0.4,
        ))
    fig_late.update_layout(
        xaxis=dict(range=[0, 100], title="Probability (%)",
                   gridcolor='gray', ticksuffix='%'),
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}, showlegend=False,
        margin=dict(t=20, b=40, l=20, r=80),
        bargap=0.4,
    )

    # ── 3. Quantile delay chart ───────────────────────────────────────────────
    dep_pred = [reg_dep_delay[f'q_{q}'].predict(X)[0] for q in [10, 50, 90]]
    arr_pred = [reg_arr_delay[f'q_{q}'].predict(X)[0] for q in [10, 50, 90]]

    fig_box = go.Figure()
    for label, pred, color in [
        ("Departure Delay", dep_pred, 'rgba(255,159,64,1)'),
        ("Arrival Delay",   arr_pred, 'rgba(153,102,255,1)'),
    ]:
        fig_box.add_trace(go.Scatter(
            name=label, x=[label], y=[pred[1]],
            error_y=dict(
                type='data', symmetric=False,
                array=[max(pred[2] - pred[1], 0)],
                arrayminus=[max(pred[1] - pred[0], 0)],
                color=color, thickness=3, width=20
            ),
            marker=dict(color=color, size=15),
            mode='markers',
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"10th pct: {pred[0]:.1f} min<br>"
                f"Median:   {pred[1]:.1f} min<br>"
                f"90th pct: {pred[2]:.1f} min<extra></extra>"
            )
        ))

    fig_box.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
    fig_box.update_layout(
        yaxis_title="Delay in Minutes (Negative is Early)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        yaxis=dict(gridcolor='gray', zerolinecolor='white'),
        margin=dict(t=20, b=40)
    )

    # ── 4. Taxi insight ───────────────────────────────────────────────────────
    taxi_insight = (
        f"Historical data at {origin} indicates an average Taxi-Out time of "
        f"{t_out_avg:.1f} minutes contributing to possible departure discrepancies. "
        f"Upon landing at {dest}, expect an average Taxi-In time of "
        f"{t_in_avg:.1f} minutes before reaching the gate."
    )

    return fig_gauge, fig_late, fig_box, taxi_insight


if __name__ == '__main__':
    app.run(debug=True, port=8050)