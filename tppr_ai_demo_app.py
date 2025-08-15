
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="OBW AI Safety Assistant — Floorplan", layout="wide", page_icon="🧭")

SAFE, WARN, DANGER = "safe", "warn", "danger"
STATE_COLORS = {SAFE: "#22c55e", WARN:"#f59e0b", DANGER:"#ef4444", "fault":"#f97316", "over":"#e5e7eb"}

# --- Rooms (floorplan coords) ---
ROOMS = {
    "entry":    {"name":"Entry",        "poly":[(2,2),(10,2),(10,8),(2,8)], "center":(6,5)},
    "room1":    {"name":"Room 1",       "poly":[(22,2),(36,2),(36,12),(22,12)], "center":(29,7)},
    "room2":    {"name":"Room 2",       "poly":[(36,12),(52,12),(52,28),(36,28)], "center":(44,20)},
    "room3":    {"name":"Room 3",       "poly":[(22,12),(36,12),(36,28),(22,28)], "center":(29,20)},
    "room12":   {"name":"Room 12",      "poly":[(10,2),(22,2),(22,12),(10,12)], "center":(16,7)},
    "prod1":    {"name":"Production 1", "poly":[(10,12),(22,12),(22,28),(10,28)], "center":(16,20)},
    "prod2":    {"name":"Production 2", "poly":[(2,12),(10,12),(10,28),(2,28)], "center":(6,20)},
}

ROOM_DISPLAY = {k:v["name"] for k,v in ROOMS.items()}

# --- Session ---
if "data" not in st.session_state:
    # Build demo data
    rng = pd.date_range(end=pd.Timestamp.utcnow(), periods=600, freq="min")
    rows = []
    keys = list(ROOMS.keys())
    for i, ts in enumerate(rng):
        rk = keys[i % len(keys)]
        base = 10 + (i % 30) * 0.12 + np.random.normal(0,0.3)
        rows.append({"timestamp": ts, "room": rk, "ppm": round(float(base),2)})
    st.session_state.data = pd.DataFrame(rows)

for k,v in [("view","facility"),("room_key",None),("detector_key",None),
            ("demo_index",0),("free_play",False),("show_route",False)]:
    if k not in st.session_state: st.session_state[k]=v

# --- Helpers ---
def last_ppm(room_key: str):
    df = st.session_state.data
    sub = df[df["room"]==room_key]
    if sub.empty: return None, None
    i = min(len(sub)-1, st.session_state.demo_index)
    row = sub.iloc[i]
    return float(row["ppm"]), pd.to_datetime(row["timestamp"])

def status_for_room(room_key: str):
    val,_ = last_ppm(room_key)
    if val is None: return "fault"
    if val >= 70: return "over"
    if val >= 55: return DANGER
    if val >= 45: return WARN
    return SAFE

def demo_tick():
    if st.session_state.demo_index < len(st.session_state.data)-1:
        st.session_state.demo_index += 1

# --- Top status bar ---
top1, top2, top3 = st.columns([2,4,2])
with top1:
    st.markdown("### OBW Technologies")
with top2:
    st.caption(f"Date: {pd.Timestamp.utcnow().date()} | Time: {pd.Timestamp.utcnow().strftime('%H:%M:%S')} | Lang: en")
with top3:
    st.caption("Heartbeat: ON  |  Event: OK")

# --- Quick room buttons (unique keys) ---
btn_cols = st.columns(len(ROOMS))
for idx, (rk, rv) in enumerate(ROOMS.items()):
    with btn_cols[idx]:
        if st.button(ROOM_DISPLAY[rk], key=f"roombtn_{rk}"):
            st.session_state.room_key = rk
            st.session_state.view = "room"

# --- Sidebar Dev (unique keys) ---
if st.toggle("Dev", key="dev_toggle_global"):
    st.session_state.free_play = True
else:
    st.session_state.free_play = False

if st.session_state.free_play:
    with st.sidebar:
        st.markdown("### Simulation Center")
        room_choice = st.selectbox("Room", list(ROOMS.keys()), key="sim_room")
        mode = st.selectbox("Mode", ["Spike","Ramp","O₂ drop","CO spike"], key="sim_mode")
        intensity = st.slider("Intensity", 5, 100, 40, step=5, key="sim_intensity")
        duration = st.slider("Duration (ticks)", 3, 30, 10, key="sim_duration")
        if st.button("Run Simulation", key="sim_run"):
            apply_simulation(room_choice, mode, intensity, duration)
            st.toast(f"Simulated {mode} in {ROOM_DISPLAY[room_choice]}")
        if st.button("Replay Incident", key="sim_replay"):
            st.session_state.demo_index = 0

def apply_simulation(room_key: str, mode: str, intensity: int, duration: int):
    df = st.session_state.data.copy()
    idx = st.session_state.demo_index
    if mode == "Spike":
        mask = (df["room"] == room_key) & (df.index >= idx) & (df.index < idx + duration)
        df.loc[mask, "ppm"] = df.loc[mask, "ppm"] + float(intensity)
    elif mode == "Ramp":
        for i in range(duration):
            mask = (df["room"] == room_key) & (df.index == idx + i)
            df.loc[mask, "ppm"] = df.loc[mask, "ppm"] + float(intensity) * (i + 1) / duration
    elif mode == "O₂ drop":
        for i in range(duration):
            mask = (df["room"] == room_key) & (df.index == idx + i)
            df.loc[mask, "ppm"] = df.loc[mask, "ppm"] - float(intensity) * (i + 1) / duration
    elif mode == "CO spike":
        for i in range(duration):
            mask = (df["room"] == room_key) & (df.index == idx + i)
            df.loc[mask, "ppm"] = df.loc[mask, "ppm"] + float(intensity) * 0.4
    st.session_state.data = df

# --- Facility floorplan ---
def build_floorplan():
    fig = go.Figure()
    # background
    fig.add_shape(type="rect", x0=0, y0=0, x1=56, y1=30, fillcolor="#224a2c", line=dict(color="#224a2c"))
    for rk, rv in ROOMS.items():
        col = STATE_COLORS[status_for_room(rk)]
        poly = rv["poly"]
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=col,
                                 line=dict(color="#111827"), hoverinfo="skip", name=rk,
                                 showlegend=False))
        cx, cy = rv["center"]
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers+text",
                                 marker=dict(size=10, color="#111"), text=[rv["name"]],
                                 textposition="top center",
                                 name=f"label_{rk}", showlegend=False,
                                 customdata=[rk], hovertemplate="%{text}<extra></extra>"))
    # Legend
    leg = [("Healthy","#22c55e"),("Inhibit","#f59e0b"),("Alarm","#ef4444"),("Fault","#f97316"),("Activated / Over Range","#e5e7eb")]
    lx, ly = 46, 3
    fig.add_shape(type="rect", x0=lx-1, y0=ly-1, x1=lx+10, y1=ly+9, line=dict(color="#9ca3af"), fillcolor="#1f2937", opacity=0.9)
    for i,(label,color) in enumerate(leg):
        fig.add_shape(type="rect", x0=lx, y0=ly+i*1.6, x1=lx+1.2, y1=ly+1.2+i*1.6, fillcolor=color, line=dict(color=color))
        fig.add_annotation(x=lx+6.2, y=ly+0.6+i*1.6, text=label, showarrow=False, font=dict(color="#e5e7eb", size=11))
    fig.update_layout(template="plotly_white", xaxis=dict(visible=False, range=[0,56]), yaxis=dict(visible=False, range=[30,0]),
                      margin=dict(l=0,r=0,t=0,b=0), height=560, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    return fig

# --- Router ---
def render_facility():
    st.subheader("Facility Overview")
    fig = build_floorplan()
    clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="facility_click_floorplan", override_height=560)
    if clicked:
        payload = clicked[0]
        rk = payload.get("customdata")
        if isinstance(rk, str) and rk in ROOMS:
            st.session_state.room_key = rk
            st.session_state.view = "room"
    # table
    st.markdown("### Status")
    rows = []
    for rk in ROOMS.keys():
        ppm,_ = last_ppm(rk)
        rows.append({"Room":ROOM_DISPLAY[rk], "Reading": f"{ppm:.1f} ppm" if ppm is not None else "—", "State": status_for_room(rk).upper()})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_room():
    rk = st.session_state.room_key
    if rk is None or rk not in ROOMS:
        st.session_state.view="facility"; return
    st.subheader(ROOM_DISPLAY[rk])
    # AI mini analyst
    df_room = st.session_state.data[st.session_state.data["room"]==rk].iloc[: st.session_state.demo_index+1].copy()
    if not df_room.empty:
        df_room["roll_mean"] = df_room["ppm"].rolling(12, min_periods=3).mean()
    col_plot, col_act = st.columns([3,1])
    with col_plot:
        fig = go.Figure()
        if not df_room.empty:
            fig.add_trace(go.Scatter(x=df_room["timestamp"], y=df_room["ppm"], mode="lines+markers", name="Live", line=dict(width=3)))
            if "roll_mean" in df_room.columns:
                fig.add_trace(go.Scatter(x=df_room["timestamp"], y=df_room["roll_mean"], mode="lines", name="Rolling mean", line=dict(dash="dot")))
        fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True, key=f"room_timeseries_{rk}")
    with col_act:
        st.write("**Actions**")
        if st.button("Evacuation", key=f"btn_evac_{rk}"):
            st.session_state.view = "evac"
        if st.button("Back to Facility", key=f"btn_backfac_{rk}"):
            st.session_state.view = "facility"

def render_evac():
    st.subheader("Evacuation Mode")
    fig = build_floorplan()  # route overlay could be added here under shapes
    st.plotly_chart(fig, use_container_width=True, key="evac_plot")
    if st.button("Back to Facility", key="evac_back"):
        st.session_state.view = "facility"

# --- Tick and route ---
demo_tick()

# --- View routing ---
if st.session_state.view == "facility":
    render_facility()
elif st.session_state.view == "room":
    render_room()
elif st.session_state.view == "evac":
    render_evac()
else:
    render_facility()
