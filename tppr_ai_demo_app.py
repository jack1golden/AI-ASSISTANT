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
with c1: st.image("assets/obw_logo.png", use_container_width=True)
with c2: st.markdown("<h2 style='text-align:center;margin-top:10px;'>OBW AI Safety Assistant</h2>", unsafe_allow_html=True)
with c3:
    if st.button("⚙️", help="Developer Controls", key="gear", use_container_width=True): st.session_state.free_play = not st.session_state.free_play

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

def render_facility():
    st.markdown("#### Facility Overview")
    col_map, col_side = st.columns([3,1])
    with col_map:
        st.image("assets/facility.svg", use_container_width=True)
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", xaxis=dict(visible=False, range=[0,1200]), yaxis=dict(visible=False, range=[700,0]),
                          margin=dict(l=0,r=0,t=0,b=0), height=520)
        xs, ys, texts, colors, custom = [], [], [], [], []
        now_ts = None
        for rk, d in DETECTORS.items():
            ppm, ts = last_ppm(rk); 
            if ts is not None: now_ts = ts
            stt = gas_state(rk, ppm if ppm is not None else 0); update_timers(rk, stt, ts if ts is not None else datetime.utcnow())
            xs.append(d["pos"][0]); ys.append(d["pos"][1])
            label = f"{ROOM_NAMES[rk]} — {d['model']}"
            tstr = time_in_state_str(rk, DANGER, ts if ts is not None else datetime.utcnow())
            if stt == DANGER: label += f"  ⏱ {tstr}"
            texts.append(label); colors.append(state_color(stt)); custom.append(rk)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", marker=dict(size=16, color=colors, line=dict(color="#111", width=1)),
                                 text=[t.split(' — ')[1] for t in texts], textposition="top center", customdata=custom,
                                 hovertemplate="%{text}<extra></extra>"))
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="facility_click")
        if clicked:
            room_clicked = clicked[0]["customdata"]; st.session_state.selected_room = room_clicked; st.session_state.view = "room"; st.rerun()
    with col_side:
        st.markdown("### Danger Leaderboard")
        rows = []
        for rk, t in st.session_state.timers.items():
            rows.append({"Detector": DETECTORS[rk]["model"], "Room": ROOM_NAMES[rk], "State": t["state"].upper(), "Longest Danger (s)": int(t["danger_longest"]) })
        df_lead = pd.DataFrame(rows).sort_values("Longest Danger (s)", ascending=False)
        st.dataframe(df_lead, use_container_width=True, hide_index=True)
        if st.button("Replay Incident", use_container_width=True):
            st.session_state.demo_index = 0; st.session_state.incident_log = []
            for k in st.session_state.timers: st.session_state.timers[k] = {"state": SAFE, "danger_start": None, "warn_start": None, "danger_longest": 0}
            st.session_state.view = "facility"; st.rerun()

def render_room():
    rk = st.session_state.selected_room
    if rk is None or rk not in ROOM_NAMES: st.session_state.view = "facility"; st.rerun()
    st.markdown(f"### {ROOM_NAMES[rk]}")
    col_map, col_ai = st.columns([2,1])
    with col_map:
        room_svg_path = f"assets/room_{rk}.svg" if rk in ["boiler","lab","corr_north"] else "assets/room_boiler.svg"
        st.image(room_svg_path, use_container_width=True)
        c1,c2,c3 = st.columns([1,1,2])
        if c1.button("⬅️ Door to Corridor"): st.session_state.view = "facility"; st.rerun()
        if c2.button("Evacuation"): st.session_state.view = "evac"; st.rerun()
    with col_ai:
        ppm, ts = last_ppm(rk); stt = gas_state(rk, ppm if ppm is not None else 0)
        danger_time = time_in_state_str(rk, DANGER, ts if ts is not None else datetime.utcnow())
        warn_time = time_in_state_str(rk, WARN, ts if ts is not None else datetime.utcnow())
        st.metric(f"{DETECTORS[rk]['model']} ({DETECTORS[rk]['gas']})", f"{ppm:.1f} ppm" if ppm is not None else "—")
        st.markdown(f"**State:** `{stt.upper()}`  •  **Time in Danger:** `{danger_time}`  •  **Time in Warning:** `{warn_time}`")
        st.markdown("#### AI Room Summary")
        if stt == DANGER: st.error("Danger detected. Advise immediate evacuation along nearest safe route. Isolate energy sources and stop process feeds if possible.")
        elif stt == WARN: st.warning("Warning. Levels are trending upward. Increase ventilation and prepare for evacuation if rise continues.")
        else: st.success("All clear. Monitoring normal.")
    st.markdown("### Live Readings & 15‑min Prediction")
    df_room = st.session_state.data[st.session_state.data["room"] == rk].iloc[:st.session_state.demo_index+1]
    pred = prediction_curve(rk, 15)
    fig = go.Figure()
    if not df_room.empty: fig.add_trace(go.Scatter(x=df_room["timestamp"], y=df_room["ppm"], mode="lines+markers", name="Live", line=dict(width=3)))
    if not pred.empty: fig.add_trace(go.Scatter(x=pred["timestamp"], y=pred["ppm"], mode="lines", name="Prediction", line=dict(dash="dash")))
    d = DETECTORS[rk]; warn = d["warn"]; danger = d["danger"]
    if d.get("oxygen_mode"):
        fig.add_hrect(y0=-1e6, y1=danger, fillcolor="#ef4444", opacity=0.1, line_width=0)
        fig.add_hrect(y0=danger, y1=warn, fillcolor="#f59e0b", opacity=0.08, line_width=0)
    else:
        fig.add_hrect(y0=danger, y1=1e6, fillcolor="#ef4444", opacity=0.1, line_width=0)
        fig.add_hrect(y0=warn, y1=danger, fillcolor="#f59e0b", opacity=0.08, line_width=0)
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    if stt == DANGER:
        st.session_state.view = "evac"; st.rerun()

def render_evac():
    st.markdown("### Evacuation Mode")
    st.info("AI is guiding evacuation. Follow green exit markers.")
    cols = st.columns([3,1])
    with cols[0]: st.image("assets/facility.svg", use_container_width=True)
    with cols[1]:
        if st.button("Return to Room"): st.session_state.view = "room"; st.rerun()
        if st.button("Back to Facility"): st.session_state.view = "facility"; st.rerun()
    st.markdown("#### AI Evacuation Guidance")
    rk = st.session_state.selected_room or "boiler"
    st.write(f"Starting from **{ROOM_NAMES[rk]}**. Nearest exit is **East Exit**. Avoid Corridor South if alarms are active.")

def demo_tick():
    if st.session_state.demo_index < len(st.session_state.data)-1: st.session_state.demo_index += 1

demo_tick()
view = st.session_state.view
if view == "facility": render_facility()
elif view == "room": render_room()
elif view == "evac": render_evac()
else: render_facility()
