"""Fermentation Length Optimizer — Streamlit dashboard.

Run from the repository root:

    streamlit run dashboard/fermentation_dashboard.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fermentation import (  # noqa: E402
    CONDITION_RANGES,
    FEATURES,
    Constraints,
    Economics,
    FermentationModel,
    evaluate_harvest,
    find_optimal,
    generate_history,
)
from fermentation.model import calibrate  # noqa: E402
from fermentation.optimizer import OBJECTIVES  # noqa: E402
from fermentation.simulator import CONDITIONS, TARGETS  # noqa: E402
from fermentation.validation import backtest, split_batches  # noqa: E402

# Reference data-viz palette: one hue per measure, neutral ink for annotations.
BLUE, BLUE_BAND = "#2a78d6", "rgba(42, 120, 214, 0.15)"
ORANGE, AQUA = "#eb6834", "#1baf7a"
MUTED = "#8a8984"
WINDOW_BAND = "rgba(27, 175, 122, 0.12)"

LABELS = {
    "temperature": "Temperature (°C)",
    "ph": "pH",
    "dissolved_oxygen": "Dissolved O₂ (%)",
    "inoculum_density": "Inoculum density (g/L)",
    "feed_rate": "Glucose feed (g/L/day)",
    "hour": "Fermentation time (h)",
}
OBJECTIVE_UNITS = {"profit_rate": "$/h", "productivity": "g/L/h", "titer": "g/L"}

st.set_page_config(page_title="Fermentation Length Optimizer", page_icon="🧫", layout="wide")


# --------------------------------------------------------------------------- data & model
@st.cache_data(show_spinner="Generating historical batches…")
def load_synthetic(n_batches: int, seed: int) -> pd.DataFrame:
    return generate_history(n_batches=n_batches, seed=seed)[1]


@st.cache_resource(show_spinner="Training fermentation model…")
def train(timeseries: pd.DataFrame):
    train_ts, test_ts = split_batches(timeseries)
    model = FermentationModel().fit(train_ts)
    return model, test_ts


@st.cache_data(show_spinner="Back-testing on held-out batches…")
def run_backtest(_model, test_ts, econ: Economics, cons: Constraints, objective: str, key: str):
    return backtest(_model, test_ts, econ, cons, objective)


@st.cache_data(show_spinner="Computing feature importance…")
def importance(_model, timeseries, key: str) -> pd.Series:
    return _model.feature_importance(timeseries)


def base_layout(fig: go.Figure, title: str, y_title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        xaxis_title=LABELS["hour"],
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig


def mark_harvest(fig: go.Figure, hour: float, window: tuple[float, float] | None = None):
    if window:
        fig.add_vrect(x0=window[0], x1=window[1], fillcolor=WINDOW_BAND, line_width=0, layer="below")
    fig.add_vline(x=hour, line=dict(color=MUTED, width=1.5, dash="dash"),
                  annotation_text=f"Harvest {hour:.0f} h", annotation_position="top left")


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🧫 Batch set-up")

with st.sidebar.expander("Historical data", expanded=False):
    source = st.radio("Source", ["Synthetic history", "Upload CSV"], label_visibility="collapsed")
    if source == "Upload CSV":
        st.caption("Long format, one row per sample: `batch_id`, "
                   + ", ".join(f"`{c}`" for c in FEATURES + TARGETS))
        upload = st.file_uploader("Batch history CSV", type="csv")
        if upload is None:
            st.info("Using synthetic history until a file is uploaded.")
            history = load_synthetic(400, 42)
        else:
            history = pd.read_csv(upload)
            missing = set(["batch_id"] + FEATURES + TARGETS) - set(history.columns)
            if missing:
                st.error(f"Missing columns: {', '.join(sorted(missing))}")
                st.stop()
    else:
        n_batches = st.slider("Batches", 100, 800, 400, 50)
        history = load_synthetic(n_batches, 42)
    st.caption(f"{history['batch_id'].nunique()} batches · {len(history):,} samples")

model, test_ts = train(history)
max_hour = float(history["hour"].max())

st.sidebar.subheader("Process set-points")
conditions = {}
for name, rng in CONDITION_RANGES.items():
    step = 0.05 if name in ("ph", "inoculum_density") else 0.5
    conditions[name] = st.sidebar.slider(
        LABELS[name], float(rng["min"]), float(rng["max"]), float(rng["default"]), step
    )

st.sidebar.subheader("Optimization goal")
objective = st.sidebar.selectbox("Maximize", list(OBJECTIVES), format_func=OBJECTIVES.get)

with st.sidebar.expander("Economics", expanded=False):
    econ = Economics(
        reactor_volume_l=st.number_input("Working volume (L)", 10.0, 50000.0, 2000.0, 100.0),
        product_price_per_g=st.number_input("Product value ($/g purified)", 1.0, 5000.0, 100.0, 10.0),
        operating_cost_per_h=st.number_input("Operating cost ($/h)", 0.0, 20000.0, 800.0, 50.0),
        batch_fixed_cost=st.number_input("Fixed cost per batch ($)", 0.0, 1e6, 30000.0, 1000.0),
        turnaround_h=st.number_input("Turnaround between batches (h)", 0.0, 168.0, 24.0, 2.0),
        resin_cost_per_g_hcp=st.number_input("Gel wear cost ($/g HCP)", 0.0, 500.0, 20.0, 1.0,
                                              help="Chromatography resin wear and extra "
                                                   "purification per gram of host-cell protein."),
    )

with st.sidebar.expander("Harvest constraints", expanded=True):
    hours_window = st.slider("Allowed harvest window (h)", 0.0, max_hour, (48.0, max_hour), 2.0)
    cons = Constraints(
        min_viability=st.slider("Min. viability at harvest (%)", 0.0, 100.0, 70.0, 1.0),
        max_hcp=st.slider("Max. HCP at harvest (mg/L)", 100.0, 3000.0, 1500.0, 50.0),
        min_hours=hours_window[0],
        max_hours=hours_window[1],
    )

with st.sidebar.expander("Running batch? Calibrate with a sample", expanded=False):
    use_calibration = st.checkbox("Use in-process measurement")
    sample_hour = st.number_input("Elapsed time (h)", 1.0, max_hour, 72.0, 1.0)
    sample_titer = st.number_input("Measured titer (g/L)", 0.0, 50.0, 2.0, 0.05)
    sample_viab = st.number_input("Measured viability (%)", 0.0, 100.0, 95.0, 0.5)

# --------------------------------------------------------------------------- prediction
hours = np.arange(0, max_hour + 1, 1.0)
curves = model.predict_curves(conditions, hours)
if use_calibration:
    curves = calibrate(curves, sample_hour, sample_titer, sample_viab)
evaluated = evaluate_harvest(curves, econ)
best = find_optimal(evaluated, cons, objective)
opt_h = best["hour"]

# Near-optimal harvest window: hours reaching ≥ 98 % of the best objective, inside constraints.
feasible = (
    evaluated["hour"].between(cons.min_hours, cons.max_hours)
    & (evaluated["viability"] >= cons.min_viability)
    & (evaluated["hcp"] <= cons.max_hcp)
)
tol = 0.02 * abs(best[objective])
near = evaluated[feasible & (evaluated[objective] >= best[objective] - tol)]
window = (near["hour"].min(), near["hour"].max()) if not near.empty else None

# Pessimistic / optimistic optimum from the titer uncertainty band.
opt_lo = find_optimal(evaluate_harvest(curves, econ, "titer_lo"), cons, objective)
opt_hi = find_optimal(evaluate_harvest(curves, econ, "titer_hi"), cons, objective)

# --------------------------------------------------------------------------- header & KPIs
st.title("Fermentation Length Optimizer")
st.caption(
    "Predicts titer, viability and host-cell protein (HCP) over time for the chosen set-points, "
    "then picks the harvest time that maximizes the selected goal. HCP drives downstream yield "
    "and chromatography gel wear."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Optimal fermentation length", f"{opt_h:.0f} h", f"{opt_h / 24:.1f} days",
          delta_color="off", delta_arrow="off")
if window:
    k2.metric("Harvest window (≥ 98 % of best)", f"{window[0]:.0f}–{window[1]:.0f} h")
else:
    k2.metric("Harvest window", "—")
k3.metric("Titer at harvest", f"{best['titer']:.2f} g/L",
          f"80 % band {best['titer_lo']:.2f}–{best['titer_hi']:.2f}", delta_color="off", delta_arrow="off")
k4.metric("Viability / HCP at harvest", f"{best['viability']:.0f} %", f"{best['hcp']:.0f} mg/L HCP",
          delta_color="off", delta_arrow="off")
k5.metric("Profit rate", f"${best['profit_rate']:,.0f}/h", f"${best['profit']:,.0f} per batch",
          delta_color="off", delta_arrow="off")

if not best["feasible"]:
    st.error("⛔ No harvest time satisfies every constraint for these set-points — showing the best "
             "time inside the harvest window. Relax the constraints or change the set-points.")
elif best["limited_by"] == "viability":
    st.warning(f"⚠️ Harvest pulled in by the viability limit ({cons.min_viability:.0f} %). "
               "Without it, the goal would peak later.")
elif best["limited_by"] == "hcp":
    st.warning(f"⚠️ Harvest pulled in by the HCP limit ({cons.max_hcp:.0f} mg/L). "
               "Without it, the goal would peak later.")
if use_calibration:
    st.info(f"ℹ️ Trajectory calibrated to the sample at {sample_hour:.0f} h "
            f"({sample_titer:.2f} g/L, {sample_viab:.0f} % viable).")

tab_pred, tab_sens, tab_hist, tab_val = st.tabs(
    ["📈 Prediction", "🎛️ Sensitivity", "🗂️ Similar batches", "✅ Model validation"]
)

# --------------------------------------------------------------------------- tab: prediction
with tab_pred:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves["hour"], y=curves["titer_hi"], line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=curves["hour"], y=curves["titer_lo"], line=dict(width=0),
                             fill="tonexty", fillcolor=BLUE_BAND, name="80 % interval",
                             hovertemplate="%{y:.2f} g/L (P10)<extra></extra>"))
    fig.add_trace(go.Scatter(x=curves["hour"], y=curves["titer"], line=dict(color=BLUE, width=2),
                             name="Predicted titer", hovertemplate="%{y:.2f} g/L<extra></extra>"))
    if use_calibration:
        fig.add_trace(go.Scatter(x=[sample_hour], y=[sample_titer], mode="markers", name="Sample",
                                 marker=dict(size=10, color=BLUE, line=dict(width=2, color="white"))))
    mark_harvest(fig, opt_h, window)
    st.plotly_chart(base_layout(fig, "Product titer", "Titer (g/L)", 380), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=curves["hour"], y=curves["viability"], name="Viability",
                                   line=dict(color=ORANGE, width=2),
                                   hovertemplate="%{y:.1f} %<extra></extra>"))
        fig.add_hline(y=cons.min_viability, line=dict(color=MUTED, width=1, dash="dot"),
                      annotation_text=f"Min {cons.min_viability:.0f} %", annotation_position="bottom left")
        mark_harvest(fig, opt_h)
        st.plotly_chart(base_layout(fig, "Cell viability", "Viability (%)"), width="stretch")
    with c2:
        fig = go.Figure(go.Scatter(x=curves["hour"], y=curves["hcp"], name="HCP",
                                   line=dict(color=AQUA, width=2),
                                   hovertemplate="%{y:.0f} mg/L<extra></extra>"))
        fig.add_hline(y=cons.max_hcp, line=dict(color=MUTED, width=1, dash="dot"),
                      annotation_text=f"Max {cons.max_hcp:.0f} mg/L", annotation_position="top left")
        mark_harvest(fig, opt_h)
        st.plotly_chart(base_layout(fig, "Host-cell protein (gel load)", "HCP (mg/L)"),
                        width="stretch")

    unit = OBJECTIVE_UNITS[objective]
    fig = go.Figure(go.Scatter(x=evaluated["hour"], y=evaluated[objective], name=OBJECTIVES[objective],
                               line=dict(color=BLUE, width=2),
                               hovertemplate=f"%{{y:,.2f}} {unit}<extra></extra>"))
    infeasible = evaluated[~feasible]
    fig.add_trace(go.Scatter(x=infeasible["hour"], y=infeasible[objective], mode="markers",
                             name="Violates a constraint", marker=dict(size=4, color=MUTED),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[opt_h], y=[best[objective]], mode="markers", name="Optimum",
                             marker=dict(size=12, color=BLUE, line=dict(width=2, color="white")),
                             hovertemplate=f"Optimum %{{x:.0f}} h: %{{y:,.2f}} {unit}<extra></extra>"))
    mark_harvest(fig, opt_h, window)
    st.plotly_chart(base_layout(fig, f"Goal if harvested at each hour — {OBJECTIVES[objective]}", unit),
                    width="stretch")
    st.caption(
        f"Uncertainty: using the pessimistic (P10) titer curve the optimum is **{opt_lo['hour']:.0f} h**, "
        f"with the optimistic (P90) curve **{opt_hi['hour']:.0f} h**. The shaded green band marks harvest "
        "times within 2 % of the best achievable value."
    )

    with st.expander("Hour-by-hour table"):
        cols = ["hour", "titer", "titer_lo", "titer_hi", "viability", "hcp", "downstream_yield",
                "purified_kg", "profit", "profit_rate", "productivity"]
        st.dataframe(evaluated[cols].iloc[::6].round(3), width="stretch", hide_index=True)
        st.download_button("Download full prediction (CSV)", evaluated[cols].to_csv(index=False),
                           "fermentation_prediction.csv", "text/csv")

# --------------------------------------------------------------------------- tab: sensitivity
with tab_sens:
    st.markdown("How does the optimal length change if you move one set-point, all else held?")
    param = st.selectbox("Set-point to sweep", CONDITIONS, format_func=LABELS.get)
    rng = CONDITION_RANGES[param]
    sweep = []
    for value in np.linspace(rng["min"], rng["max"], 17):
        cond = {**conditions, param: float(value)}
        ev = evaluate_harvest(model.predict_curves(cond, hours), econ)
        opt = find_optimal(ev, cons, objective)
        sweep.append({param: value, "optimal_hours": opt["hour"], objective: opt[objective],
                      "feasible": opt["feasible"]})
    sweep = pd.DataFrame(sweep)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=sweep[param], y=sweep["optimal_hours"], mode="lines+markers",
                                   line=dict(color=BLUE, width=2), marker=dict(size=8),
                                   name="Optimal length",
                                   hovertemplate="%{x:.2f} → %{y:.0f} h<extra></extra>"))
        fig.add_vline(x=conditions[param], line=dict(color=MUTED, dash="dash"),
                      annotation_text="Current", annotation_position="top left")
        fig = base_layout(fig, "Optimal fermentation length", "Hours")
        fig.update_layout(hovermode="closest", xaxis_title=LABELS[param])
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure(go.Scatter(x=sweep[param], y=sweep[objective], mode="lines+markers",
                                   line=dict(color=BLUE, width=2), marker=dict(size=8),
                                   name=OBJECTIVES[objective],
                                   hovertemplate=f"%{{x:.2f}} → %{{y:,.2f}} {unit}<extra></extra>"))
        fig.add_vline(x=conditions[param], line=dict(color=MUTED, dash="dash"),
                      annotation_text="Current", annotation_position="top left")
        fig = base_layout(fig, f"Best achievable — {OBJECTIVES[objective]}", unit)
        fig.update_layout(hovermode="closest", xaxis_title=LABELS[param])
        st.plotly_chart(fig, width="stretch")
    if not sweep["feasible"].all():
        st.caption("Some sweep points cannot meet every constraint; their best in-window hour is shown.")

# --------------------------------------------------------------------------- tab: similar batches
with tab_hist:
    batch_cond = history.groupby("batch_id")[CONDITIONS].first()
    spans = pd.Series({c: CONDITION_RANGES[c]["max"] - CONDITION_RANGES[c]["min"] for c in CONDITIONS})
    target = pd.Series(conditions)[CONDITIONS]
    distance = (((batch_cond - target) / spans) ** 2).sum(axis=1) ** 0.5
    n_similar = st.slider("Number of most similar historical batches", 3, 15, 5)
    nearest = distance.nsmallest(n_similar)

    fig = go.Figure()
    for i, batch_id in enumerate(nearest.index):
        b = history[history["batch_id"] == batch_id]
        fig.add_trace(go.Scatter(x=b["hour"], y=b["titer"], mode="lines", name="Historical batches",
                                 legendgroup="hist", showlegend=i == 0,
                                 line=dict(color=MUTED, width=1), opacity=0.7,
                                 hovertemplate=f"Batch {batch_id}: %{{y:.2f}} g/L<extra></extra>"))
    fig.add_trace(go.Scatter(x=curves["hour"], y=curves["titer"], name="Prediction",
                             line=dict(color=BLUE, width=2.5),
                             hovertemplate="Prediction: %{y:.2f} g/L<extra></extra>"))
    mark_harvest(fig, opt_h)
    fig = base_layout(fig, "Predicted titer vs. the most similar past batches", "Titer (g/L)", 400)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")

    table = batch_cond.loc[nearest.index].copy()
    table["similarity_distance"] = nearest
    peak = history[history["batch_id"].isin(nearest.index)].loc[
        lambda d: d.groupby("batch_id")["titer"].idxmax()].set_index("batch_id")
    table["peak_titer_g_l"] = peak["titer"]
    table["hour_of_peak"] = peak["hour"]
    st.dataframe(table.round(2).rename(columns=LABELS), width="stretch")

# --------------------------------------------------------------------------- tab: validation
with tab_val:
    key = f"{len(history)}-{history['batch_id'].nunique()}"
    bt = run_backtest(model, test_ts, econ, cons, objective, key)
    mae = bt["error_hours"].abs().mean()
    median_err = bt["error_hours"].abs().median()
    within_12 = (bt["error_hours"].abs() <= 12).mean()
    captured = bt["captured_objective"].sum() / bt["best_objective"].sum()

    st.markdown(
        f"Back-test on **{len(bt)} held-out batches** the model never saw: predict the optimal length "
        "from set-points only, then compare with the optimum found in that batch's actual measurements."
    )
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Mean abs. error", f"{mae:.0f} h")
    v2.metric("Median abs. error", f"{median_err:.0f} h")
    v3.metric("Within ±12 h", f"{within_12:.0%}")
    v4.metric("Goal captured", f"{captured:.0%}",
              help="Sum of the goal reached by harvesting at the predicted time, divided by the sum "
                   "reached at each batch's true optimum.")

    c1, c2 = st.columns([3, 2])
    with c1:
        lim = [bt[["actual_hours", "predicted_hours"]].min().min() - 5,
               bt[["actual_hours", "predicted_hours"]].max().max() + 5]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction",
                                 line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bt["actual_hours"], y=bt["predicted_hours"], mode="markers",
                                 name="Held-out batch",
                                 marker=dict(size=8, color=BLUE, opacity=0.75,
                                             line=dict(width=1, color="white")),
                                 customdata=bt[["batch_id", "error_hours"]],
                                 hovertemplate="Batch %{customdata[0]}<br>Actual %{x:.0f} h · "
                                               "Predicted %{y:.0f} h<br>Error %{customdata[1]:+.0f} h"
                                               "<extra></extra>"))
        fig = base_layout(fig, "Predicted vs. actual optimal length", "Predicted (h)", 400)
        fig.update_layout(hovermode="closest", xaxis_title="Actual optimum (h)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        imp = importance(model, history, key)
        imp = imp[imp > 0].sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[LABELS[i] for i in imp.index], orientation="h",
                               marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig = base_layout(fig, "What drives titer (permutation importance)", "", 400)
        fig.update_layout(hovermode="closest", xaxis_title="Drop in R² when shuffled", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.caption(
        "Most remaining error comes from batch-to-batch variation that set-points alone cannot "
        "explain (seed-train quality, raw-material lots). For a running batch, use "
        "*Calibrate with a sample* in the sidebar to correct the forecast with a real measurement."
    )
    with st.expander("Back-test details"):
        st.dataframe(bt.round(2), width="stretch", hide_index=True)
