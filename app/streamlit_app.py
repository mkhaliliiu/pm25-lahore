"""Streamlit dashboard: Lahore PM2.5 forecasts and severe-smog warnings.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deploy import HORIZONS, Forecaster, aqi_category  # noqa: E402

DATA, RES = ROOT / "app" / "data", ROOT / "results"
FIG = RES / "figures"
THRESHOLD = 125.5
SERIES = {  # name in the snapshot -> (label, colour); fixed order so colours never shift
    "y": ("Observed", "#52514e"),
    "Ridge + CAMS": ("Forecast (Ridge + CAMS)", "#2a78d6"),
    "Raw CAMS": ("Raw CAMS", "#eb6834"),
    "Persistence": ("Persistence", "#1baf7a"),
}
BASELINES = {"Persistence", "Weekly naive", "Climatology", "Raw CAMS", "No-event (always 0)"}

st.set_page_config(page_title="Lahore PM2.5 forecast", page_icon="🌫️", layout="wide")


@st.cache_data
def load_backtest():
    bt = pd.read_csv(DATA / "backtest.csv.gz", parse_dates=["time"])
    bt["target_time"] = bt["time"] + pd.to_timedelta(bt["h"], unit="h")
    bt["month"] = bt["target_time"].dt.to_period("M").astype(str)
    return bt


@st.cache_data
def load_result(name):
    path = RES / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_recent():
    return pd.read_csv(DATA / "recent_features.csv.gz", parse_dates=["time"], index_col="time")


@st.cache_resource
def load_model(meta_name="meta.json"):
    return Forecaster(meta_name)


@st.cache_data(ttl=1800, show_spinner="Fetching today's weather and CAMS forecast from Open-Meteo...")
def get_live():
    """Live inputs are refreshed at most every 30 minutes."""
    from src.live import fetch_live
    row, cams, missing = fetch_live()
    return row, cams, missing, load_model("meta_live.json").predict(row)


def show_figure(name, caption):
    if (FIG / name).exists():
        st.image(str(FIG / name), caption=caption, use_container_width=True)


st.title("Lahore PM2.5 forecasts and severe-smog warnings")
st.caption("US Diplomatic Post monitor (OpenAQ) · ERA5 weather and CAMS forecasts (Open-Meteo) · "
           "accuracy figures are out-of-sample, from a 10-fold rolling-origin backtest (Aug 2023 – Mar 2025).")

tab_live, tab_explore, tab_results, tab_shap, tab_api, tab_about = st.tabs(
    ["Live forecast", "Forecast explorer", "Model results", "Explainability", "Deployed model & API", "About"])

# ----------------------------------------------------------------------------- live
with tab_live:
    try:
        row, cams, missing, pred = get_live()
    except Exception as e:
        st.warning(f"Open-Meteo could not be reached just now, so there is no live forecast. ({e})")
    else:
        live_meta = load_model("meta_live.json").meta
        t = row.index[0]
        p = pred.iloc[0]
        st.subheader(f"Lahore, forecast issued {t:%d %b %Y, %H:%M} (Pakistan time)")
        cols = st.columns(len(HORIZONS))
        for col, hh in zip(cols, HORIZONS):
            pm = p[f"pm25_h{hh}"]
            col.metric(f"{(t + pd.Timedelta(hours=hh)):%a %d %b, %H:%M} (+{hh} h)", f"{pm:.0f} µg/m³")
            col.caption(f"US EPA category: **{aqi_category(pm)}** · raw CAMS {row[f'cams_h{hh}'].iloc[0]:.0f} µg/m³")
            prob = p[f"event_prob_h{hh}"]
            if p[f"event_warning_h{hh}"]:
                col.error(f"⚠ Severe-smog warning · probability {prob:.2f}")
            else:
                col.success(f"✓ No severe-smog warning · probability {prob:.2f}")
        st.caption("A warning means the model sees a real chance that the **24-hour mean** around that time "
                   "exceeds 125.5 µg/m³. Warnings are set so that missing a smog day counts 10 times worse than "
                   "a false alarm, so they lean towards caution and can appear even when the single-hour "
                   "forecast is lower.")

        window = cams[(cams.index >= t - pd.Timedelta(hours=24)) & (cams.index <= t + pd.Timedelta(hours=72))]
        cams_line = window.reset_index().rename(columns={"cams_pm25": "pm25"}).assign(series="Raw CAMS forecast")
        points = pd.DataFrame({"time": [t + pd.Timedelta(hours=hh) for hh in HORIZONS],
                               "pm25": [p[f"pm25_h{hh}"] for hh in HORIZONS], "series": "Our forecast"})
        domain, colours = ["Raw CAMS forecast", "Our forecast"], ["#eb6834", "#2a78d6"]
        enc = dict(x=alt.X("time:T", title=None, axis=alt.Axis(format="%a %d %b %H:%M", tickCount=8)),
                   y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)"),
                   color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=colours),
                                   legend=alt.Legend(orient="top", title=None)),
                   tooltip=[alt.Tooltip("time:T", format="%d %b %H:%M"), "series:N",
                            alt.Tooltip("pm25:Q", title="PM2.5", format=".0f")])
        chart = (alt.Chart(cams_line).mark_line(strokeWidth=2).encode(**enc)
                 + alt.Chart(points).mark_circle(size=140, stroke="white", strokeWidth=2, opacity=1).encode(**enc)
                 + alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(strokeDash=[4, 4], color="#8a8985").encode(y="y:Q")
                 + alt.Chart(pd.DataFrame({"time": [t]})).mark_rule(color="#8a8985").encode(x="time:T"))
        st.altair_chart(chart.properties(height=340), use_container_width=True)
        st.caption("Solid vertical line: now. Dashed line: 125.5 µg/m³ (US EPA 'Very Unhealthy'). "
                   "Our forecast corrects the CAMS forecast for Lahore using the current weather.")

        lr = load_result("live_model_regression.csv")
        if lr is not None:
            name = live_meta["regression_model"]
            acc = lr[lr.model.isin(["Raw CAMS", "Persistence", name, "Ridge + CAMS"])].pivot(
                index="model", columns="h", values="MAE").round(1)
            acc.columns = [f"MAE {c} h" for c in acc.columns]
            st.markdown("**How accurate is the live model?** Backtest MAE (µg/m³), same test hours as the main results:")
            st.dataframe(acc.rename(index={"Ridge + CAMS": "Full model (needs measured PM2.5)"}))
        st.info("The live model uses only Open-Meteo inputs (current weather, current CAMS value and the CAMS "
                "forecast), because the US Embassy monitor's public feed ended in March 2025. "
                + (f"Not provided by Open-Meteo right now: {', '.join(missing)} (filled automatically). " if missing else "")
                + "Live weather comes from a forecast model, not the ERA5 reanalysis used in training.")

# ----------------------------------------------------------------------------- explorer
with tab_explore:
    bt = load_backtest()
    months = sorted(bt["month"].unique())
    worst = bt[bt.h == 24].groupby("month")["y"].mean().idxmax()
    c1, c2 = st.columns([1, 1])
    h = c1.radio("Forecast horizon", HORIZONS, index=0, horizontal=True, format_func=lambda x: f"{x} h ahead")
    month = c2.selectbox("Month (target time)", months, index=months.index(worst))
    g = bt[(bt.h == h) & (bt.month == month)].sort_values("target_time")

    cols = st.columns(len(SERIES) - 1)
    for col, key in zip(cols, list(SERIES)[1:]):
        ok = g[[key, "y"]].dropna()
        mae = (ok[key] - ok["y"]).abs().mean() if len(ok) else float("nan")
        col.metric(f"MAE · {SERIES[key][0]}", f"{mae:.1f} µg/m³")

    long = g.melt(id_vars="target_time", value_vars=list(SERIES), var_name="key", value_name="pm25").dropna()
    long["series"] = long["key"].map(lambda k: SERIES[k][0])
    labels, colours = [v[0] for v in SERIES.values()], [v[1] for v in SERIES.values()]
    hover = alt.selection_point(fields=["target_time"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(long).encode(
        x=alt.X("target_time:T", title=None, axis=alt.Axis(format="%d %b", tickCount=10)),
        y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)"),
        color=alt.Color("series:N", scale=alt.Scale(domain=labels, range=colours),
                        legend=alt.Legend(orient="top", title=None)))
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_circle(size=60).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("target_time:T", title="Time", format="%d %b %H:%M"), "series:N",
                 alt.Tooltip("pm25:Q", title="PM2.5", format=".0f")]).add_params(hover)
    rule = alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(strokeDash=[4, 4], color="#8a8985").encode(y="y:Q")
    st.altair_chart((lines + points + rule).properties(height=380), use_container_width=True)
    st.caption(f"Dashed line: 125.5 µg/m³ (US EPA 'Very Unhealthy'). Each forecast was made {h} hours earlier "
               "by a model that had not seen this period.")

    ev = g.dropna(subset=["event_observed", "event_warning"])
    if len(ev):
        tp = int(((ev.event_warning == 1) & (ev.event_observed == 1)).sum())
        fn = int(((ev.event_warning == 0) & (ev.event_observed == 1)).sum())
        fp = int(((ev.event_warning == 1) & (ev.event_observed == 0)).sum())
        st.subheader("Smog warnings this month (24-h mean > 125.5)")
        w1, w2, w3 = st.columns(3)
        w1.metric("Event hours warned in time", f"{tp}")
        w2.metric("Event hours missed", f"{fn}")
        w3.metric("False-alarm hours", f"{fp}")

    with st.expander("Show the data as a table"):
        st.dataframe(g[["time", "target_time", "y", "Ridge + CAMS", "Raw CAMS", "Persistence",
                        "event_prob", "event_warning", "event_observed"]].round(1), hide_index=True)

# ----------------------------------------------------------------------------- results
with tab_results:
    reg = load_result("regression_summary.csv")
    hr = st.radio("Horizon", HORIZONS, horizontal=True, key="h_res", format_func=lambda x: f"{x} h")
    if reg is not None:
        t = reg[reg.h == hr].sort_values("MAE").copy()
        t["type"] = t["model"].map(lambda m: "Baseline" if m in BASELINES else "Machine learning")
        bar = alt.Chart(t).mark_bar(cornerRadiusEnd=4, height=14).encode(
            x=alt.X("MAE:Q", title="MAE (µg/m³), lower is better"),
            y=alt.Y("model:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("type:N", scale=alt.Scale(domain=["Machine learning", "Baseline"],
                                                       range=["#2a78d6", "#a3a29d"]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["model", alt.Tooltip("MAE:Q", format=".1f"), alt.Tooltip("RMSE:Q", format=".1f"),
                     alt.Tooltip("R2:Q", format=".2f")])
        st.altair_chart(bar.properties(height=34 * len(t)), use_container_width=True)
        st.dataframe(t[["model", "MAE", "RMSE", "R2", "MAPE", "MAE_fold_std", "skill_vs_persistence",
                        "skill_vs_cams"]].round(3), hide_index=True)

    st.subheader("Is the difference real? Diebold-Mariano and bootstrap")
    dm, boot = load_result("dm_tests.csv"), load_result("bootstrap_ci.csv")
    if dm is not None:
        st.dataframe(dm[dm.h == hr].round(3), hide_index=True)
    if boot is not None:
        st.dataframe(boot[boot.h == hr].round(2), hide_index=True)

    st.subheader("RQ2: a random split makes the model look better than it is")
    rq2 = load_result("rq2_random_vs_rolling.csv")
    if rq2 is not None:
        st.dataframe(rq2.round(2), hide_index=True)

    st.subheader("RQ3: severe-smog warnings (cost of a miss = 10 false alarms)")
    cls = load_result("classification_summary.csv")
    if cls is not None:
        ev_names = list(cls.event.unique())
        evn = st.selectbox("Event", ev_names, index=next((i for i, e in enumerate(ev_names) if "125.5" in e), 0))
        sub = cls[(cls.event == evn) & (cls.cost_ratio == 10)]
        hc = st.radio("Horizon", sorted(sub.h.unique()), horizontal=True, key="h_cls", format_func=lambda x: f"{x} h")
        st.dataframe(sub[sub.h == hc][["model", "recall", "precision", "F1", "PR_AUC", "PR_AUC_random",
                                       "false_alarm_hours_per_30d", "FN", "FP", "cost"]].round(3), hide_index=True)
    lead = load_result("lead_time.csv")
    if lead is not None:
        st.markdown("**Lead time at the start of smog episodes**")
        st.dataframe(lead.round(1), hide_index=True)

    with st.expander("Figures used in the paper"):
        for name, cap in [("mae_by_horizon.png", "MAE by horizon"), ("mae_by_fold.png", "MAE per test fold"),
                          ("example_forecast.png", "Example month"), ("pr_curves_h48.png", "Precision-recall, 48 h"),
                          ("recall_by_season.png", "Recall by season"), ("lstm_comparison.png", "LSTM comparison")]:
            show_figure(name, cap)

# ----------------------------------------------------------------------------- SHAP
with tab_shap:
    st.markdown("What drives the LightGBM + CAMS forecast? SHAP values show how much each input "
                "pushes a prediction up or down (µg/m³).")
    c1, c2 = st.columns(2)
    with c1:
        show_figure("shap_summary_h48.png", "SHAP summary, 48 h ahead")
    with c2:
        show_figure("shap_importance_by_horizon.png", "Mean |SHAP| by horizon")
        show_figure("shap_dependence_cams_h48.png", "How the CAMS forecast is corrected")
    imp = load_result("shap_importance.csv")
    if imp is not None:
        st.dataframe(imp.head(15).round(2), hide_index=True)

# ----------------------------------------------------------------------------- deployed model
with tab_api:
    fc, recent = load_model(), load_recent()
    st.markdown(f"The deployed models (**{fc.meta['regression_model']}** for PM2.5, "
                f"**{fc.meta['event_model']}** for warnings) were trained on "
                f"{fc.meta['trained_on'][0][:10]} to {fc.meta['trained_on'][1][:10]}. The monitor's public "
                "feed ends in March 2025, so this tab runs them on the last 60 days of stored data.")
    times = list(recent.index)
    t = st.select_slider("Issue time", options=times, value=times[-1],
                         format_func=lambda x: x.strftime("%d %b %Y %H:%M"))
    p = fc.predict(recent.loc[[t], fc.features]).iloc[0]
    cols = st.columns(len(HORIZONS))
    for col, hh in zip(cols, HORIZONS):
        pm = p[f"pm25_h{hh}"]
        col.metric(f"{(t + pd.Timedelta(hours=hh)):%d %b %H:%M} (+{hh} h)", f"{pm:.0f} µg/m³")
        col.caption(f"US EPA category: **{aqi_category(pm)}**")
        prob = p[f"event_prob_h{hh}"]
        if p[f"event_warning_h{hh}"]:
            col.error(f"⚠ Smog warning · probability {prob:.2f}")
        else:
            col.success(f"✓ No warning · probability {prob:.2f}")
    st.markdown("The same models are served by the FastAPI app (`uvicorn api.main:app`):")
    st.code("GET  /forecast?time=2025-03-01T08:00\n"
            "GET  /features/example\n"
            "POST /predict   {\"issued_at\": \"2025-03-01T08:00\", \"features\": {...}}", language="text")

# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Project.** Forecasting PM2.5 and predicting severe smog events in Lahore using machine learning on
imbalanced data. DSTT semester project, FAST NUCES Islamabad, Fall 2026 (Muhammad Khalil, Muhammad Shakil).

**Data.** Hourly PM2.5 from the US Diplomatic Post, Lahore (OpenAQ, May 2019 – Mar 2025); ERA5 weather and
the CAMS global PM2.5 forecast through Open-Meteo (CAMS from Aug 2022).

**Method.** Direct forecasts at 24, 48 and 72 h; lagged PM2.5, weather, ventilation and calendar features plus
the CAMS forecast for the target hour. Evaluation is a 10-fold rolling-origin backtest with 2-month test blocks
and a horizon-length gap, compared with persistence, weekly naive, climatology and raw CAMS, with
Diebold-Mariano tests and day-block bootstrap intervals.

**Limitations.** One monitor; archived CAMS values may not match the lead time of a real-time CAMS forecast;
the station feed ended in March 2025, so live forecasting needs a new data source.
""")
