from __future__ import annotations
import os
from typing import Tuple, List, Dict, Any

import flask
import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import httpx

from config import settings

# Accessible color palette (high-contrast)
COLORS = {
    "bg": "#0f172a",
    "panel": "#111827",
    "accent": "#10b981",
    "accent2": "#60a5fa",
    "text": "#f3f4f6",
    "muted": "#9ca3af",
    "danger": "#ef4444",
}

MISSIONS = ["Kepler", "TESS", "K2"]
STATUSES = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]


def _api_base_url() -> str:
    base = os.getenv("EXO_API_BASE_URL")
    if base:
        return base.rstrip("/")
    return ""


def _build_lightcurve_figure(time: List[float], flux: List[float], show_model: bool, normalize: bool) -> go.Figure:
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    if normalize and f.size:
        med = np.median(f)
        iqr = np.subtract(*np.percentile(f, [75, 25]))
        scale = iqr if iqr > 0 else 1.0
        f = (f - med) / scale

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=f, mode="markers", name="Flux",
        marker=dict(size=4, color=COLORS["accent2"], opacity=0.8),
        hovertemplate="t=%{x:.3f}, f=%{y:.5f}<extra></extra>",
    ))

    if show_model and f.size >= 11:
        # Simple moving average as a proxy for model fit
        win = max(11, (int(len(f) * 0.01) | 1))
        cumsum = np.cumsum(np.insert(f, 0, 0.0))
        ma = (cumsum[win:] - cumsum[:-win]) / float(win)
        tt = t[win - 1:]
        if len(tt) > len(ma):
            tt = tt[:len(ma)]
        fig.add_trace(go.Scatter(
            x=tt, y=ma, mode="lines", name="Model (MA)",
            line=dict(color=COLORS["accent"], width=2),
            hoverinfo="skip",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
        margin=dict(l=40, r=10, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Tempo (dias)", gridcolor="#1f2937")
    fig.update_yaxes(title_text="Brilho relativo", gridcolor="#1f2937")
    return fig


def create_dash_app(requests_pathname_prefix: str = "/ui/") -> Tuple[flask.Flask, Dash]:
    server = flask.Flask(__name__)

    external_stylesheets = [dbc.themes.SLATE]
    app = dash.Dash(
        __name__,
        server=server,
        suppress_callback_exceptions=True,
        # Register routes at '/' inside the mounted WSGI app, but make the client request under the provided prefix
        routes_pathname_prefix="/",
        requests_pathname_prefix=requests_pathname_prefix,
        external_stylesheets=external_stylesheets,
        title="Exoplanet Transit Explorer",
        update_title=None,
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    )

    # Layout
    app.layout = html.Div(
        id="root",
        children=[
            html.Header(
                className="app-header",
                children=[
                    html.H1("Exoplanet Transit Explorer", className="app-title"),
                    html.Div(
                        className="search-bar",
                        children=[
                            dcc.Input(
                                id="search-input",
                                type="text",
                                placeholder="Search for a star or exoplanet...",
                                className="search-input",
                            ),
                            dbc.Button("Search", id="search-btn", color="success", className="ml-2", n_clicks=0),
                        ],
                    ),
                ],
            ),
            html.Main(
                className="app-main container-fluid",
                children=[
                    dbc.Row([
                        dbc.Col([
                            html.Section([
                                html.H2("Filters", className="section-title"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Mission", html_for="filter-mission"),
                                        dcc.Dropdown(
                                            id="filter-mission",
                                            options=[{"label": m, "value": m} for m in MISSIONS],
                                            value=["Kepler", "TESS", "K2"],
                                            multi=True,
                                            clearable=False,
                                        ),
                                    ], md=6),
                                    dbc.Col([
                                        dbc.Label("Status"),
                                        dcc.Dropdown(
                                            id="filter-status",
                                            options=[{"label": s.replace("_", " "), "value": s} for s in STATUSES],
                                            value=[],
                                            multi=True,
                                        ),
                                    ], md=6),
                                ]),
                                # Optional filters (placeholders)
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Star Type"),
                                        dcc.Input(id="filter-star-type", type="text", placeholder="Ex.: G2V"),
                                    ], md=6),
                                    dbc.Col([
                                        dbc.Label("Detection Method"),
                                        dcc.Dropdown(
                                            id="filter-method",
                                            options=[{"label": "Trânsito", "value": "TRANSIT"}],
                                            value="TRANSIT",
                                            clearable=False,
                                        ),
                                    ], md=6),
                                ]),
                            ], className="filters-panel"),
                            html.Section([
                                html.H2("Results", className="section-title"),
                                dash_table.DataTable(
                                    id="results-table",
                                    columns=[
                                        {"name": "Planet", "id": "target_name"},
                                        {"name": "Star", "id": "star_name"},
                                        {"name": "Period (d)", "id": "orbital_period"},
                                        {"name": "Radius (R_earth)", "id": "planet_radius"},
                                        {"name": "Distance (ly)", "id": "distance_ly"},
                                        {"name": "Mission", "id": "mission"},
                                        {"name": "Status", "id": "status"},
                                    ],
                                    data=[],
                                    page_current=0,
                                    page_size=10,
                                    page_action="custom",
                                    sort_action="custom",
                                    sort_mode="multi",
                                    sort_by=[],
                                    style_table={"overflowX": "auto"},
                                    style_cell={"color": COLORS["text"], "backgroundColor": COLORS["panel"]},
                                    style_header={"backgroundColor": "#1f2937", "fontWeight": "bold"},
                                    row_selectable="single",
                                ),
                            ], className="results-panel"),
                        ], md=5),
                        dbc.Col([
                            html.Section([
                                html.H2("Exoplanet Details", className="section-title"),
                                html.Div(id="planet-details", className="details-card"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Checklist(
                                            id="lc-options",
                                            options=[
                                                {"label": "Normalize", "value": "normalize"},
                                                {"label": "Show model", "value": "model"},
                                            ],
                                            value=["normalize", "model"],
                                            inline=True,
                                            switch=True,
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.Checklist(
                                            id="lc-download",
                                            options=[{"label": "Allow remote download", "value": "allow"}],
                                            value=[],
                                            inline=True,
                                            switch=True,
                                        ),
                                    ], md=4),
                                    dbc.Col([
                                        dbc.Checklist(
                                            id="lc-fold",
                                            options=[{"label": "Fold by period", "value": "fold"}],
                                            value=[],
                                            inline=True,
                                            switch=True,
                                        ),
                                    ], md=4),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Period (days)"),
                                        dcc.Input(id="lc-period", type="number", min=0.01, step=0.01, placeholder="Ex.: 3.14"),
                                    ], md=3),
                                    dbc.Col(
                                        dbc.Button("Classify with AI", id="predict-btn", color="primary", n_clicks=0),
                                        className="mb-2",
                                        md=3,
                                    ),
                                    dbc.Col(html.Div(id="predict-result"), md=6),
                                ], align="center"),
                                dcc.Loading(
                                    dcc.Graph(id="lightcurve-graph", config={"displaylogo": False}),
                                    type="dot",
                                ),
                                html.Div(id="lc-alt-text", className="sr-only", role="img", **{"aria-live": "polite"}),
                            ], className="details-panel"),
                        ], md=7),
                    ], className="g-3"),
                ],
            ),
            html.Footer(
                className="app-footer",
                children=[html.Small("Made with FastAPI + Dash | High-contrast palette | WCAG 2.1 AA accessible")],
            ),
            dcc.Store(id="store-total", storage_type="memory"),
            dcc.Store(id="store-ids", storage_type="memory"),
            dcc.Store(id="store-lc", storage_type="memory"),
        ],
    )

    # Add a simple route to confirm the server is running
    @server.route("/test")
    def test():
        return "Dash server is running!"

    # Callbacks
    @app.callback(
        Output("results-table", "data"),
        Output("store-total", "data"),
        Input("search-btn", "n_clicks"),
        Input("search-input", "n_submit"),
        State("search-input", "value"),
        State("filter-mission", "value"),
        State("filter-status", "value"),
        Input("results-table", "page_current"),
        Input("results-table", "page_size"),
        Input("results-table", "sort_by"),
        prevent_initial_call=False,
    )
    def update_table(n_clicks, n_submit, query, missions, status, page_current, page_size, sort_by):
        base = _api_base_url()
        params = {
            "query": query or "",
            "page": (page_current or 0) + 1,
            "page_size": page_size or 10,
        }
        if missions:
            for m in missions:
                params.setdefault("mission", []).append(m)
        if status:
            for s in status:
                params.setdefault("status", []).append(s)
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.get(f"{base}/catalog/search", params=params)
                r.raise_for_status()
                payload = r.json()
        except Exception:
            payload = {"total": 0, "items": []}
        items = payload.get("items", [])
        # Apply sorting on current page
        if sort_by:
            for sort in reversed(sort_by):  # apply in reverse for stable multi-sort
                col = sort.get("column_id")
                direction = sort.get("direction", "asc")
                try:
                    items.sort(key=lambda x: (x.get(col) is None, x.get(col)), reverse=(direction == "desc"))
                except Exception:
                    pass
        # Placeholders readable
        for it in items:
            for k in ["star_name", "orbital_period", "planet_radius", "distance_ly"]:
                if it.get(k) in (None, "", "null"):
                    it[k] = "—"
        return items, payload.get("total", 0)

    @app.callback(
        Output("planet-details", "children"),
        Output("lightcurve-graph", "figure"),
        Output("lc-alt-text", "children"),
        Output("store-lc", "data"),
        Input("results-table", "selected_rows"),
        State("results-table", "data"),
        State("lc-options", "value"),
        State("lc-download", "value"),
        State("lc-fold", "value"),
        State("lc-period", "value"),
        prevent_initial_call=True,
    )
    def update_details(selected_rows, rows, options, dl_opts, fold_opts, period):
        if not rows or not selected_rows:
            return html.Div("Selecione um alvo na tabela."), go.Figure(), "", None
        row = rows[selected_rows[0]]
        mission = row.get("mission")
        ids = row.get("ids") or {}
        allow_download = "allow" in (dl_opts or [])

        base = _api_base_url()
        params = {"mission": mission, "download": str(allow_download).lower()}
        for key in ("kepid", "tic", "epic"):
            if key in ids and ids[key] is not None:
                params[key] = ids[key]
                break
        # Request lightcurve
        time, flux = [], []
        target_name = row.get("target_name")
        try:
            with httpx.Client(timeout=60.0) as client:
                r = client.get(f"{base}/lightcurve", params=params)
                if r.status_code == 404:
                    fig = go.Figure()
                    fig.update_layout(
                        paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
                        font=dict(color=COLORS["text"]),
                    )
                    details = _details_card(row)
                    return details, fig, "Curva de luz não encontrada para o alvo selecionado.", None
                r.raise_for_status()
                data = r.json()
                time = data.get("time") or []
                flux = data.get("flux") or []
        except Exception:
            fig = go.Figure()
            fig.update_layout(
                paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
                font=dict(color=COLORS["text"]),
            )
            details = _details_card(row)
            return details, fig, "Erro ao carregar curva de luz.", None

        # Folding
        fold = "fold" in (fold_opts or [])
        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)
        if fold and period and period > 0 and t.size:
            phase = ((t % period) / period) - 0.5
            order = np.argsort(phase)
            t = phase[order]
            f = f[order]
            time = t.tolist()
            flux = f.tolist()

        show_model = "model" in (options or [])
        normalize = "normalize" in (options or [])
        fig = _build_lightcurve_figure(time, flux, show_model=show_model, normalize=normalize)
        axis_x = "Fase (períodos)" if (fold and period and period > 0) else "Tempo (dias)"
        fig.update_xaxes(title_text=axis_x)
        alt_text = (
            f"Curva de luz para {target_name} com {len(time)} pontos. Eixo X: {axis_x.lower()}; "
            f"Eixo Y: brilho relativo. Modelo {'exibido' if show_model else 'oculto'}."
        )
        details = _details_card(row)
        store = {"target_name": target_name, "mission": mission, "time": time, "flux": flux}
        return details, fig, alt_text, store

    @app.callback(
        Output("predict-result", "children"),
        Input("predict-btn", "n_clicks"),
        State("store-lc", "data"),
        prevent_initial_call=True,
    )
    def run_prediction(n_clicks, store):
        if not n_clicks or not store or not store.get("flux"):
            return dash.no_update
        base = _api_base_url()
        candidate = {
            "target_name": store.get("target_name") or "Alvo",
            "light_curve": {
                "flux": store.get("flux"),
                "mission": store.get("mission") or "Kepler",
            },
            "stellar_params": None,
            "transit_params": None,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(f"{base}/predict", json=candidate)
                r.raise_for_status()
                res = r.json()
        except Exception:
            return dbc.Alert("Falha ao classificar o alvo.", color="danger", dismissable=True)
        pred = res.get("prediction")
        conf = float(res.get("confidence") or 0.0)
        q = res.get("quality_score")
        probs = res.get("probabilities") or {}
        prog = dbc.Progress(value=int(conf * 100), label=f"Confiança: {conf*100:.1f}%", color="success", striped=True, animated=True)
        prob_list = html.Ul([
            html.Li(f"{k}: {v*100:.1f}%") for k, v in probs.items()
        ])
        items = [
            html.Div([html.Strong("Classe:"), html.Span(f" {pred}")]),
            prog,
            html.Div([html.Strong("Qualidade do sinal:"), html.Span(f" {q:.2f}" if q is not None else " —")]),
            html.Div([html.Strong("Probabilidades:"), prob_list]),
        ]
        return dbc.Alert(items, color="dark")

    def _details_card(row: Dict[str, Any]):
        return html.Div(
            className="details-grid",
            children=[
                _kv("Planeta", row.get("target_name")),
                _kv("Missão", row.get("mission")),
                _kv("Status", (row.get("status") or "").replace("_", " ")),
                _kv("Estrela", row.get("star_name") or "—"),
                _kv("Período", row.get("orbital_period") or "—"),
                _kv("Raio", row.get("planet_radius") or "—"),
                _kv("Distância", row.get("distance_ly") or "—"),
            ],
        )

    def _kv(k: str, v: Any):
        return html.Div([
            html.Div(k, className="kv-key"),
            html.Div(str(v) if v not in (None, "") else "—", className="kv-val"),
        ], className="kv-row")

    return server, app
