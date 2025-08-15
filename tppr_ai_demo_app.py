
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="OBW — Next‑Gen Safety (Floorplan Skin)", layout="wide", page_icon="🧭")

# ---------- Constants & Colors ----------
SAFE, WARN, DANGER = "safe", "warn", "danger"
STATUS_COLORS = {
    "Healthy": "#22c55e",
    "Inhibit": "#fbbf24",
    "Alarm": "#ef4444",
    "Fault": "#f97316",
    "Activated": "#f3f4f6",
}
STATE_TO_STATUS = {
    SAFE: "Healthy",
    WARN: "Inhibit",
    DANGER: "Alarm",
}
ROOM_DISPLAY = {
    "reactor": "Room 1",
    "granulation": "Production 1",
    "cleanroom": "Room 3",
    "packaging": "Production 2",
    "warehouse": "Room 12",
    "utilities": "Entry",
}

# ---------- Facility model (2D floorplan) ----------
# Coordinates in a 1200x700 canvas to mimic the ref look
ROOMS = {
    "reactor":    {"name": "Reactor Suite", "center": (350, 190), "poly": [(240,120),(460,120),(460,260),(240,260)]},
    "granulation":{"name": "Granulation",    "center": (640, 190), "poly": [(540,120),(740,120),(740,260),(540,260)]},
    "cleanroom":  {"name": "Cleanroom",      "center": (860, 420), "poly": [(760,320),(960,320),(960,520),(760,520)]},
    "packaging":  {"name": "Packaging",      "center": (560, 420), "poly": [(440,320),(680,320),(680,520),(440,520)]},
    "warehouse":  {"name": "Warehouse",      "center": (340, 420), "poly": [(240,320),(440,320),(440,520),(240,520)]},
    "utilities":  {"name": "Utilities",      "center": (150, 560), "poly": [(80,520),(220,520),(220,640),(80,640)]},
}
# Exits approx
EXITS = {"East": (1160, 360), "West": (40, 360)}

# ---------- Session ----------
if "data" not in st.session_state:
    try:
        st.session_state.data = pd.read_csv("demo_data.csv", parse_dates=["timestamp"])
    except Exception:
        rng = pd.date_range(end=pd.Timestamp.utcnow(), periods=600, freq="min")
        keys = list(ROOMS.keys())
        rows = []
        for i, ts in enumerate(rng):
            rk = keys[i % len(keys)]
            base = 10 + (i % 30) * 0.1
            rows.append({"timestamp": ts, "room": rk, "ppm": base})
        st.session_state.data = pd.DataFrame(rows)
if "view" not in st.session_state:
    st.session_state.view = "facility"  # facility | room | detector | evac
if "room_key" not in st.session_state:
    st.session_state.room_key = None
if "detector_key" not in st.session_state:
    st.session_state.detector_key = None
if "demo_index" not in st.session_state:
    st.session_state.demo_index = 0
if "timers" not in st.session_state:
    st.session_state.timers = {rk: {"state": SAFE, "danger_start": None, "warn_start": None, "danger_longest": 0} for rk in ROOMS.keys()}
if "incident_log" not in st.session_state:
    st.session_state.incident_log = []
if "free_play" not in st.session_state:
    st.session_state.free_play = False
if "show_route" not in st.session_state:
    st.session_state.show_route = True
if "last_event" not in st.session_state:
    st.session_state.last_event = None

# ---------- Helper logic ----------
def room_state(room_key: str, ppm: float) -> str:
    # simple thresholds per demo; customize per room/gas later
    # Treat >70 as Activated, >55 Alarm, >45 Inhibit
    if ppm >= 70: return "Activated"
    if ppm >= 55: return "Alarm"
    if ppm >= 45: return "Inhibit"
    return "Healthy"

def gas_state_basic(ppm: float) -> str:
    return DANGER if ppm >= 55 else (WARN if ppm >= 45 else SAFE)

def last_ppm(room_key: str):
    df = st.session_state.data
    sub = df[df["room"] == room_key]
    if sub.empty:
        return None, None
    last = sub.iloc[min(len(sub) - 1, st.session_state.demo_index)]
    return float(last["ppm"]), pd.to_datetime(last["timestamp"])

def compute_route(start_room_key: str):
    start = ROOMS[start_room_key]["center"]
    east, west = EXITS["East"], EXITS["West"]
    target = east if abs(east[0]-start[0]) < abs(start[0]-west[0]) else west
    path = [start, (start[0], 360), (target[0], 360), target]
    # dedupe
    dedup=[path[0]]
    for p in path[1:]:
        if p != dedup[-1]: dedup.append(p)
    return dedup

# ---------- Header (like SCADA bar) ----------
def render_header():
    with st.container():
        col_l, col_c, col_r = st.columns([2,5,3])
        with col_l:
            st.markdown("### OBW Technologies")
            st.caption("next generation")
        with col_c:
            # event ticker
            if st.session_state.incident_log:
                last = st.session_state.incident_log[-1]
                st.write(f"**{last['timestamp'].strftime('%H:%M:%S')}** — {last['room']} → {last['to'].upper()}")
            else:
                st.write("System OK • No events")
        with col_r:
            now = datetime.utcnow()
            st.write(f"**Date:** {now.strftime('%d.%m.%Y')}  \n**Time:** {now.strftime('%H:%M:%S')}  \n**User:** demo  \n**Lang:** en")

# ---------- Legend ----------
def render_legend():
    st.markdown("### Legend")
    for label, color in STATUS_COLORS.items():
        st.markdown(f"<div style='display:flex;align-items:center;margin-bottom:4px'><div style='width:16px;height:16px;background:{color};margin-right:8px;border:1px solid #111'></div>{label}</div>", unsafe_allow_html=True)

# ---------- Facility map ----------
def build_floorplan(show_route=False, start_room=None):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1200]),
        yaxis=dict(visible=False, range=[700, 0]),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        paper_bgcolor="#e5e7eb",
        plot_bgcolor="#c7d2fe"
    )
    # Outer green field
    fig.add_shape(type="rect", x0=60, y0=60, x1=1140, y1=640, fillcolor="#6ee7b7", line=dict(color="#065f46", width=3))
    # Building footprint (light)
    fig.add_shape(type="rect", x0=120, y0=120, x1=1080, y1=600, fillcolor="#e5e7eb", line=dict(color="#111827", width=2))

    # Draw rooms
    pins_x, pins_y, labels, statuses = [], [], [], []
    for rk, r in ROOMS.items():
        poly = r["poly"]
        ppm, _ = last_ppm(rk)
        status = room_state(rk, ppm or 0)
        color = STATUS_COLORS[status]
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                                 fillcolor=color, line=dict(color="#374151", width=2),
                                 hovertemplate=f"{ROOM_DISPLAY[rk]}<extra></extra>", name=ROOM_DISPLAY[rk]))
        cx, cy = r["center"]
        pins_x.append(cx); pins_y.append(cy); labels.append(ROOM_DISPLAY[rk]); statuses.append(status)

    # Optional evac route (draw under pins)
    if show_route and start_room in ROOMS:
        path = compute_route(start_room)
        fig.add_trace(go.Scatter(x=[p[0] for p in path], y=[p[1] for p in path],
                                 mode="lines", line=dict(width=6, color="#10b981"), hoverinfo="skip", name="Route"))

    # Pins on top
    fig.add_trace(go.Scatter(x=pins_x, y=pins_y, mode="markers+text",
                             marker=dict(size=16, color="#111827"),
                             text=labels, textposition="top center",
                             customdata=list(ROOMS.keys()),
                             hovertemplate="%{text}<extra></extra>", name="pins"))

    # Exits
    for name,(ex,ey) in EXITS.items():
        fig.add_trace(go.Scatter(x=[ex], y=[ey], mode="markers+text",
                                 marker=dict(size=10, color="#10b981", symbol="square"),
                                 text=[name+" Exit"], textposition="bottom center",
                                 hoverinfo="skip", name=name))
    return fig

def get_clicked_room(payload: dict):
    if not isinstance(payload, dict):
        return None
    # Prefer customdata (we attached it)
    cd = payload.get("customdata")
    if isinstance(cd, str) and cd in ROOMS:
        return cd
    # Fallback: nearest center
    x, y = payload.get("x"), payload.get("y")
    if isinstance(x, (int,float)) and isinstance(y, (int,float)):
        best=None; bestd=1e18
        for rk,r in ROOMS.items():
            cx,cy = r["center"]
            d=(cx-x)**2 + (cy-y)**2
            if d<bestd: bestd=d; best=rk
        return best
    return None

# ---------- Simulation ----------
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

# ---------- Header ----------
render_header()

# ---------- Toolbar (room shortcuts) ----------
short_cols = st.columns(len(ROOMS)+2)
i=0
with short_cols[i]:
    if st.button("Entry"): st.session_state.room_key="utilities"; st.session_state.view="room"
i+=1
for rk in ROOMS.keys():
    with short_cols[i]:
        if st.button(ROOM_DISPLAY[rk]):
            st.session_state.room_key=rk; st.session_state.view="room"
    i+=1
with short_cols[i]:
    if st.button("Evac"):
        st.session_state.view="evac"

# ---------- Dev sidebar ----------
with st.sidebar:
    st.markdown("### Controls")
    st.toggle("Show Evacuation Route", key="show_route")
    st.markdown("---")
    st.markdown("#### Simulation Center")
    room_choice = st.selectbox("Room", list(ROOMS.keys()), index=0, format_func=lambda k: ROOM_DISPLAY[k])
    mode = st.selectbox("Mode", ["Spike","Ramp","O₂ drop","CO spike"])
    intensity = st.slider("Intensity", 5, 100, 40, step=5)
    duration = st.slider("Duration (ticks)", 3, 30, 8)
    if st.button("Run Simulation", use_container_width=True):
        apply_simulation(room_choice, mode, intensity, duration)
        st.toast(f"Simulated {mode} in {ROOM_DISPLAY[room_choice]}")

# ---------- Main views ----------
def render_breadcrumbs():
    crumbs=["Facility"]
    if st.session_state.room_key:
        crumbs.append(ROOM_DISPLAY[st.session_state.room_key])
    st.caption(" › ".join(crumbs))

def render_facility():
    render_breadcrumbs()
    col_map, col_leg = st.columns([4,1])
    with col_map:
        fig = build_floorplan(show_route=st.session_state.show_route, start_room=st.session_state.room_key or "utilities")
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="fac_click_skin", override_height=600)
        if clicked:
            payload = clicked[0]
            st.session_state.last_event = payload
            rk = get_clicked_room(payload)
            if rk:
                st.session_state.room_key = rk
                st.session_state.view = "room"
    with col_leg:
        render_legend()

def render_room():
    rk = st.session_state.room_key
    if rk is None or rk not in ROOMS:
        st.session_state.view="facility"; return
    render_breadcrumbs()
    st.markdown(f"### {ROOM_DISPLAY[rk]}")
    ppm, ts = last_ppm(rk)
    status = room_state(rk, ppm or 0)
    st.metric("Status", status)
    st.metric("Live", f"{ppm:.1f} ppm" if ppm is not None else "—")
    # simple trend + projection
    df_room = st.session_state.data[st.session_state.data['room']==rk].iloc[:st.session_state.demo_index+1]
    future = df_room.tail(5)['ppm'].values
    slope = np.diff(future).mean() if len(future)>=2 else 0.0
    pred_x = [df_room['timestamp'].iloc[-1] + timedelta(minutes=i) for i in range(1,16)] if not df_room.empty else []
    pred_y = [max(0.0, df_room['ppm'].iloc[-1] + slope*i) for i in range(1,16)] if not df_room.empty else []
    fig = go.Figure()
    if not df_room.empty:
        fig.add_trace(go.Scatter(x=df_room['timestamp'], y=df_room['ppm'], mode="lines+markers", name="Live", line=dict(width=3)))
    if pred_x:
        fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode="lines", name="Prediction", line=dict(dash="dash")))
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    c1,c2 = st.columns(2)
    if c1.button("⬅️ Facility"): st.session_state.view="facility"
    if c2.button("Evacuation"): st.session_state.view="evac"

def render_evac():
    render_breadcrumbs()
    st.markdown("### Evacuation")
    fig = build_floorplan(show_route=True, start_room=st.session_state.room_key or "utilities")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Demo tick / timers ----------
def demo_tick():
    if st.session_state.demo_index < len(st.session_state.data) - 1:
        st.session_state.demo_index += 1
demo_tick()

# update incident log if thresholds change
now_ts = datetime.utcnow()
for rk in ROOMS.keys():
    ppm, ts = last_ppm(rk)
    new_state = gas_state_basic(ppm or 0)
    t = st.session_state.timers[rk]
    if new_state != t["state"]:
        st.session_state.incident_log.append({
            "timestamp": now_ts, "room": ROOM_DISPLAY[rk], "from": t["state"], "to": new_state
        })
        t["state"]=new_state

# Router
if st.session_state.view == "facility":
    render_facility()
elif st.session_state.view == "room":
    render_room()
elif st.session_state.view == "evac":
    render_evac()
else:
    render_facility()
