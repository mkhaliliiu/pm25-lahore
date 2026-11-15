"""Streamlit dashboard: Lahore PM2.5 forecasts and severe-smog warnings.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deploy import HORIZONS, Forecaster, HourlyRidge, aqi_category, us_aqi  # noqa: E402

DATA, RES = ROOT / "app" / "data", ROOT / "results"
FIG = RES / "figures"
THRESHOLD = 125.5
SERIES = {  # name in the snapshot -> (label, colour); fixed order so colours never shift
    "y": ("Observed", "#8f8e8a"),
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


@st.cache_resource
def load_hourly(kind):
    return HourlyRidge(kind)


def openaq_key():
    """OpenAQ key from Streamlit secrets (never stored in the repository)."""
    try:
        return st.secrets["OPENAQ_API_KEY"]
    except Exception:
        return os.getenv("OPENAQ_API_KEY")


@st.cache_data(ttl=1800, show_spinner="Reading Lahore's live sensors (OpenAQ) and weather + CAMS (Open-Meteo)...")
def get_live():
    """Live inputs and forecasts, refreshed at most every 30 minutes."""
    from src.live import build_live, verification
    res = build_live(openaq_key())
    hourly = load_hourly("full" if res["anchored"] else "live").predict(res["row"]).iloc[0]
    warn = load_model("meta.json" if res["anchored"] else "meta_live.json").predict(res["row"]).iloc[0]
    ver = verification(res["frame"], load_hourly("full"), res["issue"]) if res["anchored"] else pd.DataFrame()
    return res, hourly, warn, ver


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
        res, hourly, warn, ver = get_live()
    except Exception as e:
        st.warning(f"Live data could not be fetched just now, so there is no live forecast. ({e})")
    else:
        df_l, row, t = res["frame"], res["row"], res["issue"]
        anchored = res["anchored"]
        cal = res["calibration"]
        calibrated = not (cal["slope"] == 1 and cal["intercept"] == 0)
        st.subheader(f"Lahore now · latest measured hour {t:%a %d %b %Y, %H:%M} (Pakistan time)")
        if anchored:
            st.caption(f"Forecast mode: **full model**, anchored on the live sensor network (OpenAQ) + weather and "
                       f"CAMS (Open-Meteo). Sensors publish with a delay of about 1-2 hours.")
        else:
            st.caption("Forecast mode: **weather + CAMS only** (live sensor data unavailable right now).")

        # ---- current status, reading by reading
        c1, c2, c3, c4 = st.columns(4)
        if anchored:
            pm_now = row["pm25_obs"].iloc[0]
            c1.metric("PM2.5 measured now", f"{pm_now:.0f} µg/m³")
            c1.caption(f"US AQI **{us_aqi(pm_now)}** · {aqi_category(pm_now)}")
            c2.metric("Sensors at this hour", f"{int(row['n_sensors'].iloc[0])} of {res['sensors_total']}")
            c2.caption(("median of low-cost sensors, converted to the US Embassy reference scale through a "
                        "Clarity sensor that ran alongside both"
                        if calibrated else "median of low-cost sensors (AirGradient, Hawanama); not calibrated to "
                        "a reference monitor, because these sensors started after the US Embassy feed ended"))
        else:
            c1.metric("PM2.5 measured now", "n/a")
            c1.caption("no live sensor data")
            c2.metric("Sensors reporting", f"{res['sensors_responding']}")
        cams_now = row["cams_pm25"].iloc[0]
        c3.metric("CAMS model now", f"{cams_now:.0f} µg/m³")
        c3.caption(f"US AQI {us_aqi(cams_now)} · the global model we correct")
        c4.metric("Weather now", f"{row['temperature_2m'].iloc[0]:.0f} °C")
        c4.caption(f"humidity {row['relative_humidity_2m'].iloc[0]:.0f}% · wind {row['wind_speed_10m'].iloc[0]:.0f} km/h"
                   + (f" · mixing height {row['boundary_layer_height'].iloc[0]:.0f} m"
                      if pd.notna(row['boundary_layer_height'].iloc[0]) else ""))

        # ---- 24 / 48 / 72 h cards with warnings
        cols = st.columns(len(HORIZONS))
        for col, hh in zip(cols, HORIZONS):
            pm = hourly[hh]
            col.metric(f"{(t + pd.Timedelta(hours=hh)):%a %d %b, %H:%M} (+{hh} h)", f"{pm:.0f} µg/m³")
            col.caption(f"US AQI **{us_aqi(pm)}** · {aqi_category(pm)} · raw CAMS {row[f'cams_h{hh}'].iloc[0]:.0f}")
            prob = warn[f"event_prob_h{hh}"]
            if warn[f"event_warning_h{hh}"]:
                col.error(f"⚠ Severe-smog warning · probability {prob:.2f}")
            else:
                col.success(f"✓ No severe-smog warning · probability {prob:.2f}")
        st.caption("A warning means a real chance that the **24-hour mean** around that time exceeds 125.5 µg/m³. "
                   "Missing a smog day counts 10 times worse than a false alarm, so warnings lean towards caution.")

        # ---- chart: measured past, hourly forecast, CAMS
        past = df_l[(df_l.index >= t - pd.Timedelta(hours=72)) & (df_l.index <= t)]
        fut = df_l[(df_l.index > t) & (df_l.index <= t + pd.Timedelta(hours=72))]
        fc_times = [t + pd.Timedelta(hours=int(h)) for h in hourly.index]
        parts = [pd.DataFrame({"time": fc_times, "pm25": hourly.to_numpy(), "series": "Our forecast"}),
                 pd.concat([past, fut])[["cams_pm25"]].reset_index().rename(columns={"cams_pm25": "pm25"})
                 .assign(series="Raw CAMS")]
        if anchored:
            parts.append(past[["pm25_obs"]].reset_index().rename(columns={"pm25_obs": "pm25"}).assign(series="Measured"))
        line = pd.concat(parts, ignore_index=True).dropna()
        line["us_aqi"] = line["pm25"].map(us_aqi)
        domain, colours = ["Measured", "Our forecast", "Raw CAMS"], ["#8f8e8a", "#2a78d6", "#eb6834"]
        hover = alt.selection_point(fields=["time"], nearest=True, on="pointerover", empty=False)
        base = alt.Chart(line).encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%a %d %b %H:%M", tickCount=8)),
            y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)"),
            color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=colours),
                            legend=alt.Legend(orient="top", title=None)))
        chart = (base.mark_line(strokeWidth=2)
                 + base.mark_circle(size=60).encode(
                     opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                     tooltip=[alt.Tooltip("time:T", format="%a %d %b %H:%M"), "series:N",
                              alt.Tooltip("pm25:Q", title="PM2.5 µg/m³", format=".0f"),
                              alt.Tooltip("us_aqi:Q", title="US AQI")]).add_params(hover)
                 + alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(strokeDash=[4, 4], color="#8a8985").encode(y="y:Q")
                 + alt.Chart(pd.DataFrame({"time": [t]})).mark_rule(color="#8a8985").encode(x="time:T"))
        st.altair_chart(chart.properties(height=360), use_container_width=True)
        st.caption("Left of the vertical line: last 3 days. Right: hour-by-hour forecast for the next 72 hours. "
                   "Dashed line: 125.5 µg/m³ (US EPA 'Very Unhealthy').")

        with st.expander("Hour-by-hour forecast table (µg/m³ and US AQI, comparable with IQAir / Punjab AQI apps)"):
            table = pd.DataFrame({"Time (PKT)": [x.strftime("%a %d %b %H:%M") for x in fc_times],
                                  "PM2.5 forecast (µg/m³)": hourly.round(0).astype(int).to_numpy(),
                                  "US AQI": [us_aqi(v) for v in hourly],
                                  "Category": [aqi_category(v) for v in hourly],
                                  "Raw CAMS (µg/m³)": [round(row[f"cams_h{int(h)}"].iloc[0]) for h in hourly.index]})
            st.dataframe(table, hide_index=True, height=420)
            st.caption("Note: IQAir and the Punjab AQI site show the AQI index, not µg/m³. "
                       "For example AQI 227 = 152 µg/m³.")

        # ---- verification over the last 7 days
        if len(ver):
            st.subheader("How did our 24-hour forecasts do over the last 7 days?")
            mae_ours = (ver.forecast - ver.measured).abs().mean()
            mae_cams = (ver.raw_cams - ver.measured).abs().mean()
            v1, v2, v3 = st.columns(3)
            v1.metric("Our MAE", f"{mae_ours:.0f} µg/m³")
            v2.metric("Raw CAMS MAE", f"{mae_cams:.0f} µg/m³")
            v3.metric("Hours checked", f"{len(ver)}")
            vl = ver.melt(id_vars="target_time", value_vars=["measured", "forecast", "raw_cams"],
                          var_name="series", value_name="pm25")
            vl["series"] = vl["series"].map({"measured": "Measured", "forecast": "Our forecast", "raw_cams": "Raw CAMS"})
            vchart = alt.Chart(vl).mark_line(strokeWidth=2).encode(
                x=alt.X("target_time:T", title=None, axis=alt.Axis(format="%a %d %b")),
                y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)"),
                color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=colours),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("target_time:T", format="%d %b %H:%M"), "series:N",
                         alt.Tooltip("pm25:Q", format=".0f")])
            st.altair_chart(vchart.properties(height=280), use_container_width=True)
            st.caption("Each forecast was made 24 hours before the measurement, from the live inputs available "
                       "then. CAMS values for past days come from Open-Meteo's latest runs, so this check slightly "
                       "favours CAMS.")

        with st.expander("How accurate is this, and where do the numbers come from?"):
            nt = load_result("network_pipeline_test.csv")
            if nt is not None:
                st.markdown("**Deployment test, Nov 2024 – Mar 2025** (live sensors as inputs, US Embassy "
                            "monitor as truth), MAE in µg/m³:")
                st.dataframe(nt.pivot(index="model", columns="h", values="MAE").round(1), use_container_width=True)
            lr = load_result("live_model_regression.csv")
            if lr is not None:
                st.markdown("**Backtest, Aug 2023 – Mar 2025** (10 folds), MAE in µg/m³:")
                st.dataframe(lr.pivot(index="model", columns="h", values="MAE").round(1), use_container_width=True)
            st.markdown("Live sensors: 15 AirGradient and Hawanama low-cost monitors via OpenAQ, hourly median. "
                        + (f"Converted to the US Embassy reference scale as {cal['intercept']} + {cal['slope']} × median, "
                           "using a Clarity sensor that overlapped the Embassy monitor (2023-2025) and the live "
                           "sensors (2025-2026) as a bridge. "
                           if calibrated else "No calibration to the reference monitor was possible (no period in "
                           "which both ran), so live readings can differ in level from the training data. ")
                        + "Weather and CAMS: Open-Meteo. "
                        + (f"Not provided by Open-Meteo right now: {', '.join(res['missing_weather'])}. "
                           if res["missing_weather"] else ""))

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

**Live forecast.** The US Embassy feed ended in March 2025. The live tab therefore reads Lahore's low-cost sensor
network on OpenAQ (AirGradient, Hawanama; hourly median of 15 sensors) together with today's Open-Meteo weather
and CAMS forecast, and checks its own 24-hour forecasts against the sensors over the last 7 days.

**Limitations.** Training used one reference monitor; the live sensors are low-cost and could not be calibrated
against it (they started after it stopped); archived CAMS values may not match the lead time of a real-time CAMS
forecast; live weather comes from a forecast model rather than ERA5.
""")
