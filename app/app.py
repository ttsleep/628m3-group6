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

DEFAULT_ORIGIN   = 'LAX'
DEFAULT_DEST     = 'JFK'
DEFAULT_AIRLINE  = 'DL'
DEFAULT_DATE     = '2025-12-15'
DEFAULT_DEP_HOUR = 10

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    dbc.Row([dbc.Col([
        html.H1("AeroPredict: Flight Delay & Cancelation Intelligence",
                className="text-center mt-4 mb-2 text-primary font-weight-bold"),
        html.H5("Stat 628 Airplane Project", className="text-center text-muted mb-4")
    ], width=12)]),

    # Input panel
    dbc.Row([dbc.Col([dbc.Card([
        dbc.CardHeader("Flight Itinerary Specification", className="font-weight-bold"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Origin Airport"),
                    dcc.Dropdown(id='origin-dp',
                        options=[{'label': a, 'value': a} for a in AIRPORTS],
                        value=DEFAULT_ORIGIN, clearable=False, className="text-dark mb-3")
                ], width=6),
                dbc.Col([
                    html.Label("Destination Airport"),
                    dcc.Dropdown(id='dest-dp',
                        options=[{'label': a, 'value': a} for a in AIRPORTS],
                        value=DEFAULT_DEST, clearable=False, className="text-dark mb-3")
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Airline Carrier"),
                    dcc.Dropdown(id='airline-dp',
                        options=[{'label': a, 'value': a} for a in AIRLINES],
                        value=DEFAULT_AIRLINE, clearable=False, className="text-dark mb-3")
                ], width=4),
                dbc.Col([
                    html.Label("Flight Date"),
                    dbc.Input(id='date-picker', type='date', value=DEFAULT_DATE,
                        min='2023-09-01', max='2025-12-31', className="mb-3")
                ], width=8),
            ]),
            dbc.Row([dbc.Col([
                html.Div(id='holiday-notice', className="text-warning fst-italic mb-2")
            ], width=12)]),
            dbc.Row([
                dbc.Col([
                    html.Label("Scheduled Departure Hour — local time (0–23)"),
                    dcc.Slider(id='dep-hour-slider', min=0, max=23, step=1,
                        value=DEFAULT_DEP_HOUR,
                        marks={i: str(i) for i in range(0, 24, 3)}),
                    html.Div(id='dep-hour-display',
                        className="text-info text-center mt-1 mb-3"),
                ], width=6),
                dbc.Col([
                    html.Label("Scheduled Arrival Hour — local time (0–23)"),
                    dcc.Slider(id='arr-hour-slider', min=0, max=23, step=1,
                        value=int((DEFAULT_DEP_HOUR +
                            round(get_elapsed(DEFAULT_ORIGIN, DEFAULT_DEST) / 60)) % 24),
                        marks={i: str(i) for i in range(0, 24, 3)}),
                    html.Div(id='arr-hour-display',
                        className="text-info text-center mt-1 mb-3"),
                ], width=6),
            ]),
            dbc.Row([dbc.Col([
                dbc.Button("Predict Intelligence", id="predict-btn",
                    color="primary", size="lg", className="w-100")
            ], width=12)])
        ])
    ], className="mb-4 shadow-sm border-0")], width=12)]),

    # Row 1: Cancellation gauge (5) + Late probability (7)
    dbc.Row([
        dbc.Col([dbc.Card([
            dbc.CardHeader("Cancellation Risk"),
            dbc.CardBody([dcc.Graph(id='cancel-gauge')])
        ], className="h-100 shadow-sm border-0")], width=5),
        dbc.Col([dbc.Card([
            dbc.CardHeader("Probability of Significant Delay (> 15 min)"),
            dbc.CardBody([dcc.Graph(id='late-prob-chart')])
        ], className="h-100 shadow-sm border-0")], width=7),
    ], className="mb-4"),

    # Row 2: Delay interval chart (8) + Taxi insight (4)
    dbc.Row([
        dbc.Col([dbc.Card([
            dbc.CardHeader("Delay Predictions & Uncertainties (Minutes)"),
            dbc.CardBody([dcc.Graph(id='delay-box')])
        ], className="h-100 shadow-sm border-0")], width=8),
        dbc.Col([dbc.Card([
            dbc.CardHeader("Runway Taxi Insight"),
            dbc.CardBody([
                html.P(id="taxi-text", className="lead text-info"),
            ])
        ], className="h-100 shadow-sm border-0")], width=4),
    ], className="mb-4"),

], fluid=True, className="p-5")


# ── Helper: build delay interval figure ──────────────────────────────────────
def make_delay_fig(dep_pred, arr_pred):
    fig = go.Figure()

    labels      = ['Departure Delay', 'Arrival Delay']
    preds       = [dep_pred, arr_pred]
    good_colors = ['rgba(39,174,96,0.7)',  'rgba(39,174,96,0.7)']
    risk_colors = ['rgba(231,76,60,0.7)',  'rgba(231,76,60,0.7)']
    med_colors  = ['rgba(255,159,64,1)',   'rgba(153,102,255,1)']

    for label, pred, gc, rc, mc in zip(
            labels, preds, good_colors, risk_colors, med_colors):

        q10, q50, q90 = pred

        # Good zone: q10 → q50
        fig.add_trace(go.Bar(
            name=f'{label} best→median',
            y=[label], x=[q50 - q10], base=q10,
            orientation='h',
            marker=dict(color=gc, line=dict(width=0)),
            width=0.4, showlegend=False,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Best case (10th pct):  {q10:.1f} min<br>"
                f"Median (50th pct):     {q50:.1f} min<br>"
                f"Worst case (90th pct): {q90:.1f} min"
                "<extra></extra>"
            )
        ))

        # Risk zone: q50 → q90
        fig.add_trace(go.Bar(
            name=f'{label} median→worst',
            y=[label], x=[q90 - q50], base=q50,
            orientation='h',
            marker=dict(color=rc, line=dict(width=0)),
            width=0.4, showlegend=False,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Best case (10th pct):  {q10:.1f} min<br>"
                f"Median (50th pct):     {q50:.1f} min<br>"
                f"Worst case (90th pct): {q90:.1f} min"
                "<extra></extra>"
            )
        ))

        # Median marker
        fig.add_trace(go.Scatter(
            name=label,
            y=[label], x=[q50],
            mode='markers+text',
            marker=dict(color=mc, size=14, symbol='diamond',
                        line=dict(color='white', width=1)),
            text=[f"  {q50:.1f} min"],
            textposition='middle right',
            textfont=dict(color=mc, size=11),
            showlegend=True,
            hoverinfo='skip',
        ))

        # q10 and q90 end labels
        fig.add_annotation(
            x=q10, y=label, text=f"{q10:.1f}",
            showarrow=False, xanchor='right',
            font=dict(color='#adb5bd', size=10), xshift=-6,
        )
        fig.add_annotation(
            x=q90, y=label, text=f"{q90:.1f}",
            showarrow=False, xanchor='left',
            font=dict(color='#adb5bd', size=10), xshift=6,
        )

    fig.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.5)
    fig.add_annotation(
        x=0, y=1.08, yref='paper',
        text="← Early  |  On Time  |  Late →",
        showarrow=False,
        font=dict(color='#adb5bd', size=11),
        xanchor='center',
    )

    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            title="Minutes (negative = early)",
            gridcolor='#444',
            zerolinecolor='white',
            zerolinewidth=1,
        ),
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        legend=dict(orientation='h', y=1.15, x=0),
        margin=dict(t=50, b=40, l=20, r=40),
        height=280,
    )
    return fig


# ── Helper: run prediction & build all figures ────────────────────────────────
def make_predictions(origin, dest, airline, flight_date, dep_hour):
    d     = datetime.date.fromisoformat(flight_date) if flight_date else datetime.date(2025, 12, 15)
    month = d.month
    day   = d.isoweekday()

    cat_df = pd.DataFrame({'Origin': [origin], 'Dest': [dest],
                           'Reporting_Airline': [airline]})
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

    cancel_prob  = float(clf_cancelled.predict_proba(X)[0][1] * 100)
    dep_late_pct = float(clf_dep_late.predict_proba(X)[0][1] * 100)
    arr_late_pct = float(clf_arr_late.predict_proba(X)[0][1] * 100)
    dep_pred     = [reg_dep_delay[f'q_{q}'].predict(X)[0] for q in [10, 50, 90]]
    arr_pred     = [reg_arr_delay[f'q_{q}'].predict(X)[0] for q in [10, 50, 90]]

    # Gauge
    gauge_max = max(2.0, round(cancel_prob * 2.5, 1))
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=cancel_prob,
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

    # Late prob bars — dynamic x axis
    x_max = max(10.0, round(max(dep_late_pct, arr_late_pct) * 1.4))
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
        xaxis=dict(range=[0, x_max], title="Probability (%)",
                   gridcolor='gray', ticksuffix='%'),
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}, showlegend=False,
        margin=dict(t=20, b=40, l=20, r=80), bargap=0.4,
    )

    # Delay interval chart
    fig_box = make_delay_fig(dep_pred, arr_pred)

    # Taxi text
    taxi_insight = (
        f"📍 At {origin}, the average taxi-out time is {t_out_avg:.1f} min. "
        f"This is the time from gate push-back to wheels-off, and contributes to "
        f"departure delay but is not counted in the official DepDelay figure.\n\n"
        f"🛬 At {dest}, the average taxi-in time is {t_in_avg:.1f} min. "
        f"This is the time from wheels-on to gate arrival, and can contribute to "
        f"arrival delay at the gate."
    )

    return fig_gauge, fig_late, fig_box, taxi_insight


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
        return dep_hour, int((dep_hour + elapsed_hours) % 24)
    elif triggered_id == 'arr-hour-slider':
        return int((arr_hour - elapsed_hours) % 24), arr_hour
    else:
        return dep_hour, int((dep_hour + elapsed_hours) % 24)


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
    Input('origin-dp',   'value'),
    State('dest-dp',         'value'),
    State('airline-dp',      'value'),
    State('date-picker',     'value'),
    State('dep-hour-slider', 'value'),
)
def update_predictions(n_clicks, origin, dest, airline, flight_date, dep_hour):
    d     = datetime.date.fromisoformat(flight_date) if flight_date else datetime.date(2025, 12, 15)
    month = d.month

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

    return make_predictions(origin, dest, airline, flight_date, dep_hour)


if __name__ == '__main__':
    app.run(debug=True, port=8050)