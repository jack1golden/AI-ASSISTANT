import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="OBW AI Safety Assistant", layout="wide", page_icon="🛡️")
SAFE, WARN, DANGER = "safe", "warn", "danger"

ROOM_NAMES = {
    "boiler": "Boiler Room",
    "lab": "Process Lab",
    "corr_north": "Corridor North",
    "corr_south": "Corridor South",
    "warehouse": "Cylinder Store",
    "control": "Control Room",
}

DETECTORS = {
    "boiler": {"id":"boiler_xnx", "model":"Honeywell XNX", "gas":"CH₄", "warn":45.0, "danger":55.0, "pos":(190,170)},
    "lab": {"id":"lab_midas", "model":"Honeywell Midas", "gas":"NH₃", "warn":35.0, "danger":45.0, "pos":(190,500)},
    "corr_north": {"id":"cn_sensepoint", "model":"Honeywell Sensepoint", "gas":"O₂", "warn":18.0, "danger":17.0, "pos":(640,170), "oxygen_mode": True},
    "corr_south": {"id":"cs_searchpoint", "model":"Honeywell Searchpoint", "gas":"CO", "warn":30.0, "danger":40.0, "pos":(640,500)},
    "warehouse": {"id":"store_searchline", "model":"Honeywell Searchline", "gas":"H₂", "warn":35.0, "danger":50.0, "pos":(1040,170)},
    "control": {"id":"control_xnx", "model":"Honeywell XNX", "gas":"H₂S", "warn":5.0, "danger":10.0, "pos":(1040,500)},
}

if "view" not in st.session_state: st.session_state.view = "facility"
if "selected_room" not in st.session_state: st.session_state.selected_room = None
if "data" not in st.session_state: st.session_state.data = pd.read_csv("demo_data.csv", parse_dates=["timestamp"])
if "demo_index" not in st.session_state: st.session_state.demo_index = 0
if "timers" not in st.session_state: st.session_state.timers = {k: {"state": SAFE, "danger_start": None, "warn_start": None, "danger_longest": 0} for k in DETECTORS.keys()}
if "incident_log" not in st.session_state: st.session_state.incident_log = []
if "free_play" not in st.session_state: st.session_state.free_play = False
if "audio" not in st.session_state: st.session_state.audio = False

def gas_state(room_key, ppm):
    d = DETECTORS[room_key]
    if d.get("oxygen_mode"):
        if ppm <= d["danger"]: return DANGER
        elif ppm <= d["warn"]: return WARN
        return SAFE
    else:
        if ppm >= d["danger"]: return DANGER
        elif ppm >= d["warn"]: return WARN
        return SAFE

def update_timers(room_key, state, ts):
    t = st.session_state.timers[room_key]
    if state != t["state"]:
        st.session_state.incident_log.append({"timestamp": ts, "room": room_key, "detector": DETECTORS[room_key]["model"], "from": t["state"], "to": state})
    if state == DANGER:
        if t["danger_start"] is None: t["danger_start"] = ts
        dur = (ts - t["danger_start"]).total_seconds()
        if dur > t["danger_longest"]: t["danger_longest"] = dur
        t["warn_start"] = None
    elif state == WARN:
        if t["warn_start"] is None: t["warn_start"] = ts
        t["danger_start"] = None
    else:
        t["danger_start"] = None; t["warn_start"] = None
    t["state"] = state

def time_in_state_str(room_key, state, now_ts):
    t = st.session_state.timers[room_key]
    if state == DANGER and t["danger_start"] is not None:
        s = int((now_ts - t["danger_start"]).total_seconds()); m, s = divmod(s, 60); return f"{m}m {s}s"
    if state == WARN and t["warn_start"] is not None:
        s = int((now_ts - t["warn_start"]).total_seconds()); m, s = divmod(s, 60); return f"{m}m {s}s"
    return "—"

def last_ppm(room_key):
    df = st.session_state.data; sub = df[df["room"] == room_key]
    if sub.empty: return None, None
    last = sub.iloc[min(len(sub)-1, st.session_state.demo_index)]
    return float(last["ppm"]), pd.to_datetime(last["timestamp"])

def prediction_curve(room_key, horizon=15):
    df = st.session_state.data[st.session_state.data["room"] == room_key].iloc[:st.session_state.demo_index+1]
    if len(df) < 3: return pd.DataFrame(columns=["timestamp","ppm"])
    recent = df.tail(5)["ppm"].values
    deltas = np.diff(recent); slope = deltas.mean() if len(deltas) else 0.0
    start_ppm = df["ppm"].iloc[-1]; start_ts = df["timestamp"].iloc[-1]
    return pd.DataFrame({"timestamp":[start_ts + timedelta(minutes=i) for i in range(1,horizon+1)],
                         "ppm":[max(0.0, start_ppm + slope*i) for i in range(1,horizon+1)]})

def state_color(state): return {"safe":"#10b981","warn":"#f59e0b","danger":"#ef4444"}[state]

c1,c2,c3 = st.columns([1,3,1])
with c1: st.image("assets/obw_logo.png")
with c2: st.markdown("<h2 style='text-align:center;margin-top:10px;'>OBW AI Safety Assistant</h2>", unsafe_allow_html=True)
with c3:
    if st.button("⚙️", help="Developer Controls", key="gear"): st.session_state.free_play = not st.session_state.free_play

if st.session_state.free_play:
    with st.sidebar:
        st.markdown("### Developer Controls (Free Play)")
        st.toggle("Enable Audio Alerts", key="audio")
        if st.button("Replay Incident"):
            st.session_state.demo_index = 0; st.session_state.incident_log = []
            for k in st.session_state.timers: st.session_state.timers[k] = {"state": SAFE, "danger_start": None, "warn_start": None, "danger_longest": 0}
            st.session_state.view = "facility"; st.rerun()
        st.markdown("---"); st.markdown("**Manual Triggers**")
        for rk in DETECTORS.keys():
            if st.button(f"Trigger Danger: {ROOM_NAMES[rk]}"):
                idx = st.session_state.demo_index; df = st.session_state.data
                mask = (df["room"] == rk) & (df.index >= idx) & (df.index < idx+5)
                df.loc[mask, "ppm"] = df.loc[mask, "ppm"] + 50.0; st.session_state.data = df; st.toast(f"Forced danger for {ROOM_NAMES[rk]}")
        st.markdown("---"); st.markdown("**Mode**: Free Play (click ⚙️ to hide)")

# rest of your render_facility, render_room, render_evac, demo_tick functions stay the same

