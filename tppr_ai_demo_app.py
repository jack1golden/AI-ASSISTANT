
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="OBW AI Safety Assistant — Pharma 3D", layout="wide", page_icon="🧪")

SAFE, WARN, DANGER = "safe", "warn", "danger"
STATE_COLORS = {SAFE: "#10b981", WARN: "#f59e0b", DANGER: "#ef4444"}

ROOMS = {
    "reactor": {
        "name": "Reactor Suite",
        "center": (15, 10, 0),
        "size": (18, 12, 6),
        "detectors": [
            {"key": "rxr_h2s", "model": "Honeywell XNX", "gas": "H₂S", "warn": 5.0, "danger": 10.0, "oxygen_mode": False, "room_pos": (6, 4)},
        ],
    },
    "granulation": {
        "name": "Granulation",
        "center": (45, 10, 0),
        "size": (16, 12, 5),
        "detectors": [
            {"key": "gran_co", "model": "Sensepoint", "gas": "CO", "warn": 30.0, "danger": 40.0, "oxygen_mode": False, "room_pos": (5, 5)},
        ],
    },
    "cleanroom": {
        "name": "Cleanroom (Grade C)",
        "center": (15, 28, 0),
        "size": (20, 12, 4),
        "detectors": [
            {"key": "cr_o2", "model": "Sensepoint", "gas": "O₂", "warn": 18.0, "danger": 17.0, "oxygen_mode": True, "room_pos": (8, 5)},
        ],
    },
    "packaging": {
        "name": "Packaging Hall",
        "center": (45, 28, 0),
        "size": (22, 12, 5),
        "detectors": [
            {"key": "pack_ch4", "model": "XNX", "gas": "CH₄", "warn": 45.0, "danger": 55.0, "oxygen_mode": False, "room_pos": (9, 4)},
        ],
    },
    "warehouse": {
        "name": "Warehouse (API)",
        "center": (75, 10, 0),
        "size": (18, 12, 6),
        "detectors": [
            {"key": "wh_h2", "model": "Searchline", "gas": "H₂", "warn": 35.0, "danger": 50.0, "oxygen_mode": False, "room_pos": (6, 4)},
        ],
    },
    "utilities": {
        "name": "Utilities / Boilers",
        "center": (75, 28, 0),
        "size": (18, 12, 6),
        "detectors": [
            {"key": "util_ch4", "model": "XNX", "gas": "CH₄", "warn": 45.0, "danger": 55.0, "oxygen_mode": False, "room_pos": (6, 4)},
        ],
    },
}
DETECTOR_INDEX = {d["key"]: (room_key, d) for room_key, room in ROOMS.items() for d in room["detectors"]}
EXITS = {"west": (0, 19, 0), "east": (90, 19, 0)}

# ---- Session ----
if "data" not in st.session_state:
    try:
        st.session_state.data = pd.read_csv("demo_data.csv", parse_dates=["timestamp"])
    except Exception:
        rng = pd.date_range(end=pd.Timestamp.utcnow(), periods=300, freq="min")
        rows = []
        keys = list(ROOMS.keys())
        for i, ts in enumerate(rng):
            rk = keys[i % len(keys)]
            base = 10 + (i % 30) * 0.1
            rows.append({"timestamp": ts, "room": rk, "ppm": base})
        st.session_state.data = pd.DataFrame(rows)
if "view" not in st.session_state:
    st.session_state.view = "facility3d"  # facility3d | room | detector | evac
if "room_key" not in st.session_state:
    st.session_state.room_key = None
if "detector_key" not in st.session_state:
    st.session_state.detector_key = None
if "demo_index" not in st.session_state:
    st.session_state.demo_index = 0
if "timers" not in st.session_state:
    st.session_state.timers = {k: {"state": SAFE, "danger_start": None, "warn_start": None, "danger_longest": 0} for k in DETECTOR_INDEX.keys()}
if "incident_log" not in st.session_state:
    st.session_state.incident_log = []
if "free_play" not in st.session_state:
    st.session_state.free_play = False
if "show_route" not in st.session_state:
    st.session_state.show_route = False
if "last_event" not in st.session_state:
    st.session_state.last_event = None

# ---- Helpers ----
def gas_state(detector_cfg, ppm: float) -> str:
    if detector_cfg.get("oxygen_mode"):
        if ppm <= detector_cfg["danger"]:
            return DANGER
        elif ppm <= detector_cfg["warn"]:
            return WARN
        return SAFE
    else:
        if ppm >= detector_cfg["danger"]:
            return DANGER
        elif ppm >= detector_cfg["warn"]:
            return WARN
        return SAFE

def update_timers(detector_key: str, state: str, ts: datetime):
    t = st.session_state.timers[detector_key]
    if state != t["state"]:
        room_key, d = DETECTOR_INDEX[detector_key]
        st.session_state.incident_log.append({
            "timestamp": ts,
            "room": ROOMS[room_key]["name"],
            "detector": d["model"],
            "from": t["state"],
            "to": state,
        })
    if state == DANGER:
        if t["danger_start"] is None:
            t["danger_start"] = ts
        dur = (ts - t["danger_start"]).total_seconds()
        if dur > t["danger_longest"]:
            t["danger_longest"] = dur
        t["warn_start"] = None
    elif state == WARN:
        if t["warn_start"] is None:
            t["warn_start"] = ts
        t["danger_start"] = None
    else:
        t["danger_start"] = None
        t["warn_start"] = None
    t["state"] = state

def time_in_state_str(detector_key: str, state: str, now_ts: datetime) -> str:
    t = st.session_state.timers[detector_key]
    if state == DANGER and t["danger_start"] is not None:
        s = int((now_ts - t["danger_start"]).total_seconds())
        m, s = divmod(s, 60)
        return f"{m}m {s}s"
    if state == WARN and t["warn_start"] is not None:
        s = int((now_ts - t["warn_start"]).total_seconds())
        m, s = divmod(s, 60)
        return f"{m}m {s}s"
    return "—"

def last_ppm(room_key: str):
    df = st.session_state.data
    sub = df[df["room"] == room_key]
    if sub.empty:
        return None, None
    last = sub.iloc[min(len(sub) - 1, st.session_state.demo_index)]
    return float(last["ppm"]), pd.to_datetime(last["timestamp"])  # type: ignore

def prediction_curve(room_key: str, horizon: int = 15) -> pd.DataFrame:
    df = st.session_state.data[st.session_state.data["room"] == room_key].iloc[: st.session_state.demo_index + 1]
    if len(df) < 3:
        return pd.DataFrame(columns=["timestamp", "ppm"])
    recent = df.tail(5)["ppm"].values
    deltas = np.diff(recent)
    slope = deltas.mean() if len(deltas) else 0.0
    start_ppm = df["ppm"].iloc[-1]
    start_ts = df["timestamp"].iloc[-1]
    return pd.DataFrame({
        "timestamp": [start_ts + timedelta(minutes=i) for i in range(1, horizon + 1)],
        "ppm": [max(0.0, start_ppm + slope * i) for i in range(1, horizon + 1)],
    })

def df_table(df: pd.DataFrame):
    return st.dataframe(df, use_container_width=True, hide_index=True)

# ---- 3D Facility (Plotly) ----
def _room_mesh(room_key: str):
    cx, cy, cz = ROOMS[room_key]["center"]
    sx, sy, sz = ROOMS[room_key]["size"]
    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy - sy/2, cy + sy/2
    z0, z1 = cz, cz + sz
    X = [x0,x1,x1,x0, x0,x1,x1,x0]
    Y = [y0,y0,y1,y1, y0,y0,y1,y1]
    Z = [z0,z0,z0,z0, z1,z1,z1,z1]
    I = [0,1,2, 0,2,3,  4,5,6, 4,6,7,  0,1,5, 0,5,4,  2,3,7, 2,7,6,  1,2,6, 1,6,5,  3,0,4, 3,4,7]
    J = [1,2,3, 1,3,0,  5,6,7, 5,7,4,  1,5,4, 1,4,0,  3,7,4, 3,4,0,  2,6,5, 2,5,1,  0,4,7, 0,7,3]
    K = [2,3,0, 2,0,1,  6,7,4, 6,4,5,  5,4,0, 5,0,1,  7,4,0, 7,0,3,  6,5,1, 6,1,2,  4,7,3, 4,3,0]
    return X, Y, Z, I, J, K

def worst_state_for_room(room_key: str):
    ppm, _ = last_ppm(room_key)
    worst = SAFE
    for det in ROOMS[room_key]["detectors"]:
        s = gas_state(det, ppm or 0)
        if s == DANGER:
            return DANGER
        elif s == WARN:
            worst = WARN
    return worst

def build_facility_3d(show_route=False, start_room=None):
    fig = go.Figure()
    # Floor border
    fig.add_trace(go.Scatter3d(x=[0, 90, 90, 0, 0], y=[0, 0, 38, 38, 0], z=[0,0,0,0,0],
                               mode="lines", line=dict(width=4), hoverinfo="skip", name="Floor"))
    centers_x, centers_y, centers_z, labels, colors = [], [], [], [], []
    for rk in ROOMS.keys():
        X, Y, Z, I, J, K = _room_mesh(rk)
        state = worst_state_for_room(rk)
        fig.add_trace(go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color=STATE_COLORS[state], opacity=0.35, hoverinfo="skip"))
        cx, cy, cz = ROOMS[rk]["center"]
        centers_x.append(cx); centers_y.append(cy); centers_z.append(cz + ROOMS[rk]["size"][2] + 1.2)
        labels.append(ROOMS[rk]["name"]); colors.append(STATE_COLORS[state])
    fig.add_trace(go.Scatter3d(x=centers_x, y=centers_y, z=centers_z, mode="markers+text",
                               marker=dict(size=8, color=colors), text=labels, textposition="top center",
                               hovertemplate="%{text}<extra></extra>", name="Rooms"))
    # Exits
    for name, (ex, ey, ez) in {"west": (0,19,0), "east": (90,19,0)}.items():
        fig.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[ez], mode="markers+text",
                                   marker=dict(size=5, color="#10b981", symbol="square"),
                                   text=[name.title()+" Exit"], textposition="bottom center",
                                   hoverinfo="skip"))
    # Route
    if show_route and start_room in ROOMS:
        xs, ys, zs = zip(*compute_route_3d(start_room))
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(width=8), hoverinfo="skip"))
    fig.update_layout(height=640, margin=dict(l=0,r=0,t=0,b=0),
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 zaxis=dict(visible=False, range=[0, 12]),
                                 camera=dict(eye=dict(x=1.6, y=1.4, z=1.2)), aspectmode="data"),
                      showlegend=False, paper_bgcolor="#0f172a")
    return fig

def compute_route_3d(start_room_key: str):
    cx, cy, cz = ROOMS[start_room_key]["center"]
    east = EXITS["east"]; west = EXITS["west"]
    target = east if abs(east[0]-cx) < abs(cx-west[0]) else west
    mid = (target[0], cy, 0)
    return [(cx, cy, 0), mid, target]

def get_clicked_room_from_payload(payload: dict):
    if not isinstance(payload, dict): return None
    x, y, z = payload.get("x"), payload.get("y"), payload.get("z")
    if not all(isinstance(v, (int,float)) for v in (x,y,z)): return None
    best, bestd = None, 1e18
    for rk, r in ROOMS.items():
        cx, cy, cz = r["center"]
        d = (cx-x)**2 + (cy-y)**2 + (cz-0)**2  # floor proj
        if d < bestd: bestd, best = d, rk
    return best

# ---- Top bar ----
col_logo, col_title, col_gear = st.columns([1, 3, 1])
with col_logo: st.write("🧪")
with col_title: st.markdown("<h2 style='text-align:center;margin-top:10px;'>Pharma Facility — 3D Interactive</h2>", unsafe_allow_html=True)
with col_gear:
    if st.button("⚙️", key="gear_btn", use_container_width=True):
        st.session_state.free_play = not st.session_state.get("free_play", False)
    new_val = st.checkbox("Dev", key="dev_hdr", value=st.session_state.get("free_play", False))
    if new_val != st.session_state.get("free_play", False): st.session_state.free_play = new_val

# ---- Dev sidebar ----
if st.session_state.free_play:
    with st.sidebar:
        st.markdown("### Developer Controls")
        st.toggle("Show Evacuation Route", key="show_route")
        st.markdown("#### Simulation Center")
        room_choice = st.selectbox("Room ID", list(ROOMS.keys()), index=0)
        mode = st.selectbox("Mode", ["Spike", "Ramp", "O₂ drop", "CO spike"])
        intensity = st.slider("Intensity", 5, 100, 50, step=5)
        duration = st.slider("Duration (ticks)", 3, 30, 10)
        if st.button("Run Simulation", use_container_width=True):
            apply_simulation(room_choice, mode, intensity, duration)
            st.toast(f"Simulated {mode} in {ROOMS[room_choice]['name']}")
        if st.button("Replay Incident", use_container_width=True):
            st.session_state.demo_index = 0
            st.session_state.incident_log = []
            for k in st.session_state.timers:
                st.session_state.timers[k] = {"state": SAFE, "danger_start": None, "warn_start": None, "danger_longest": 0}
            st.session_state.view = "facility3d"

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

# ---- Debug ----
with st.expander("Debug", expanded=False):
    st.write({
        "view": st.session_state.get("view"),
        "room_key": st.session_state.get("room_key"),
        "detector_key": st.session_state.get("detector_key"),
        "demo_index": st.session_state.get("demo_index"),
        "free_play": st.session_state.get("free_play"),
        "show_route": st.session_state.get("show_route"),
    })
    st.json(st.session_state.get("last_event") or {})

# ---- Views ----
def render_breadcrumbs():
    crumbs = ["Facility 3D"]
    if st.session_state.room_key: crumbs.append(ROOMS[st.session_state.room_key]["name"])
    if st.session_state.detector_key:
        _, d = DETECTOR_INDEX[st.session_state.detector_key]; crumbs.append(d["model"])
    st.caption(" › ".join(crumbs))

def render_facility_3d():
    render_breadcrumbs()
    st.markdown("#### Facility — 3D Overview")
    col_map, col_side = st.columns([3, 1])
    with col_map:
        fig = build_facility_3d(show_route=st.session_state.show_route, start_room=st.session_state.room_key or "reactor")
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="facility3d_click", override_height=640)
        if clicked:
            payload = clicked[0]
            st.session_state.last_event = payload
            rk = get_clicked_room_from_payload(payload)
            if rk:
                st.session_state.room_key = rk
                st.session_state.detector_key = None
                st.session_state.view = "room"
    with col_side:
        st.markdown("### Danger Leaderboard")
        rows = []
        for det_key, t in st.session_state.timers.items():
            room_key, d = DETECTOR_INDEX[det_key]
            rows.append({"Detector": d["model"], "Room": ROOMS[room_key]["name"], "State": t["state"].upper(), "Longest Danger (s)": int(t["danger_longest"])})
        df_lead = pd.DataFrame(rows).sort_values("Longest Danger (s)", ascending=False)
        df_table(df_lead)

def render_room():
    render_breadcrumbs()
    rk = st.session_state.room_key
    if rk is None or rk not in ROOMS: st.session_state.view = "facility3d"; return
    room = ROOMS[rk]
    st.markdown(f"### {room['name']}")

    col_map, col_ai = st.columns([2, 1])
    with col_map:
        sx, sy, _ = room["size"]
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", xaxis=dict(visible=False, range=[0, sx]), yaxis=dict(visible=False, range=[sy, 0]), margin=dict(l=0, r=0, t=0, b=0), height=380)
        fig.add_shape(type="rect", x0=0, y0=0, x1=sx, y1=sy, line=dict(color="#475569"), fillcolor="#111827")
        xs, ys, labels, colors = [], [], [], []
        ppm, ts = last_ppm(rk)
        for det in room["detectors"]:
            stt = gas_state(det, ppm or 0)
            dx, dy = det["room_pos"]
            xs.append(dx); ys.append(dy); labels.append(f"{det['model']} ({det['gas']})"); colors.append(STATE_COLORS[stt])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", marker=dict(size=20, color=colors, line=dict(color="#0b0b0b", width=1)), text=[l.split(" (")[0] for l in labels], textposition="top center", hovertemplate="%{text}<extra></extra>", showlegend=False))
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=f"room_click_{rk}", override_height=380)
        c1, c2, _ = st.columns([1,1,2])
        if c1.button("⬅️ Back to Facility 3D"): st.session_state.view = "facility3d"
        if c2.button("Evacuation"): st.session_state.view = "evac"
        if clicked:
            payload = clicked[0]; x = payload.get("x"); y = payload.get("y")
            if isinstance(x,(int,float)) and isinstance(y,(int,float)):
                det = min(room["detectors"], key=lambda d: (d["room_pos"][0]-x)**2 + (d["room_pos"][1]-y)**2)
                st.session_state.detector_key = det["key"]; st.session_state.view = "detector"

    with col_ai:
        ppm, _ = last_ppm(rk)
        worst = SAFE
        for det in room["detectors"]:
            s = gas_state(det, ppm or 0)
            if s == DANGER: worst = DANGER; break
            elif s == WARN: worst = WARN
        if worst == DANGER: st.error("Danger detected. Consider evacuation.")
        elif worst == WARN: st.warning("Warning. Levels trending upward.")
        else: st.success("All clear.")
        st.markdown("#### Detectors")
        for det in room["detectors"]:
            if st.button(f"Open: {det['model']} — {det['gas']}", key=f"open_{det['key']}", use_container_width=True):
                st.session_state.detector_key = det["key"]; st.session_state.view = "detector"

def render_detector():
    render_breadcrumbs()
    dk = st.session_state.detector_key
    if dk is None or dk not in DETECTOR_INDEX: st.session_state.view = "room"; return
    rk, det = DETECTOR_INDEX[dk]
    st.markdown(f"### {ROOMS[rk]['name']} • {det['model']} ({det['gas']})")
    ppm, ts = last_ppm(rk)
    stt = gas_state(det, ppm or 0)
    danger_time = time_in_state_str(dk, DANGER, ts if ts is not None else datetime.utcnow())
    warn_time = time_in_state_str(dk, WARN, ts if ts is not None else datetime.utcnow())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Live Reading", f"{ppm:.1f} ppm" if ppm is not None else "—")
    c2.metric("State", stt.upper())
    c3.metric("Time in Danger", danger_time)
    c4.metric("Time in Warning", warn_time)
    st.markdown("### AI Analyst")
    df_room_full = st.session_state.data[st.session_state.data["room"] == rk].iloc[: st.session_state.demo_index + 1].copy()
    if not df_room_full.empty:
        df_room_full["roll_mean"] = df_room_full["ppm"].rolling(12, min_periods=3).mean()
        df_room_full["roll_std"] = df_room_full["ppm"].rolling(12, min_periods=3).std().replace(0, np.nan)
        last_val = df_room_full["ppm"].iloc[-1]
        mean = df_room_full["roll_mean"].iloc[-1] if not np.isnan(df_room_full["roll_mean"].iloc[-1]) else df_room_full["ppm"].mean()
        std = df_room_full["roll_std"].iloc[-1] if not np.isnan(df_room_full["roll_std"].iloc[-1]) else (df_room_full["ppm"].std() or 1.0)
        z = (last_val - mean) / std if std else 0.0
        slope = float(np.polyfit(np.arange(min(10, len(df_room_full))), df_room_full["ppm"].tail(10), 1)[0]) if len(df_room_full) >= 2 else 0.0
        warn_thr, danger_thr = det["warn"], det["danger"]
        eta = None
        if det.get("oxygen_mode"):
            if slope < 0 and last_val > danger_thr: eta = (last_val - danger_thr) / (-slope)
        else:
            if slope > 0 and last_val < danger_thr: eta = (danger_thr - last_val) / slope
        c1,c2,c3 = st.columns(3)
        c1.metric("Anomaly (z-score)", f"{z:.2f}")
        c2.metric("Slope (Δppm/tick)", f"{slope:.2f}")
        c3.metric("ETA to Danger (ticks)", "≤1" if eta is not None and eta <= 1 else (f"{eta:.1f}" if eta is not None else "—"))
        with st.expander("What‑if: Ventilation & Mitigation"):
            reduction = st.slider("Reduce slope (%)", 0, 90, 35, step=5)
            adj_slope = slope * (1 - reduction/100.0)
            horizon = 30
            start_ppm = last_val
            future = [max(0.0, start_ppm + adj_slope * i) for i in range(1, horizon+1)]
            risk = "LOW"
            projected = future[9] if len(future) >= 10 else future[-1]
            if det.get("oxygen_mode"):
                if projected <= danger_thr: risk = "CRITICAL"
                elif projected <= warn_thr: risk = "ELEVATED"
            else:
                if projected >= danger_thr: risk = "CRITICAL"
                elif projected >= warn_thr: risk = "ELEVATED"
            st.write(f"**Projected 10‑tick risk:** `{risk}`")
    st.markdown("### Live Readings & 15‑min Prediction")
    df_room = df_room_full; pred = prediction_curve(rk, 15)
    fig = go.Figure()
    if not df_room.empty:
        fig.add_trace(go.Scatter(x=df_room["timestamp"], y=df_room["ppm"], mode="lines+markers", name="Live", line=dict(width=3)))
        if "roll_mean" in df_room.columns:
            fig.add_trace(go.Scatter(x=df_room["timestamp"], y=df_room["roll_mean"], mode="lines", name="Rolling mean", line=dict(dash="dot")))
    if not pred.empty:
        fig.add_trace(go.Scatter(x=pred["timestamp"], y=pred["ppm"], mode="lines", name="Prediction", line=dict(dash="dash")))
    warn = det["warn"]; danger = det["danger"]
    if det.get("oxygen_mode"):
        fig.add_hrect(y0=-1e6, y1=danger, fillcolor=STATE_COLORS[DANGER], opacity=0.10, line_width=0)
        fig.add_hrect(y0=danger, y1=warn, fillcolor=STATE_COLORS[WARN], opacity=0.08, line_width=0)
    else:
        fig.add_hrect(y0=danger, y1=1e6, fillcolor=STATE_COLORS[DANGER], opacity=0.10, line_width=0)
        fig.add_hrect(y0=warn, y1=danger, fillcolor=STATE_COLORS[WARN], opacity=0.08, line_width=0)
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    c1,c2,_ = st.columns([1,1,2])
    if c1.button("⬅️ Room"): st.session_state.view = "room"
    if c2.button("Evacuation"): st.session_state.view = "evac"

def render_evac():
    st.markdown("### Evacuation Mode")
    st.info("3D route to safe exit. Follow green labels.")
    fig = build_facility_3d(show_route=True, start_room=st.session_state.room_key or "reactor")
    st.plotly_chart(fig, use_container_width=True)
    c1,c2,_ = st.columns([1,1,2])
    if c1.button("Return to Room"): st.session_state.view = "room"
    if c2.button("Back to Facility 3D"): st.session_state.view = "facility3d"

# ---- Demo tick ----
def demo_tick():
    if st.session_state.demo_index < len(st.session_state.data) - 1:
        st.session_state.demo_index += 1
demo_tick()

# ---- Update timers & route ----
now_ts = datetime.utcnow()
for room_key, room in ROOMS.items():
    ppm, ts = last_ppm(room_key)
    ts_use = ts if ts is not None else now_ts
    for det in room["detectors"]:
        stt = gas_state(det, ppm if ppm is not None else 0)
        update_timers(det["key"], stt, ts_use)

# ---- Router ----
def render_breadcrumbs():
    crumbs = ["Facility 3D"]
    if st.session_state.room_key: crumbs.append(ROOMS[st.session_state.room_key]["name"])
    if st.session_state.detector_key:
        _, d = DETECTOR_INDEX[st.session_state.detector_key]; crumbs.append(d["model"])
    st.caption(" › ".join(crumbs))

view = st.session_state.view
if view == "facility3d":
    render_breadcrumbs(); render_facility_3d()
elif view == "room":
    render_breadcrumbs(); render_room()
elif view == "detector":
    render_breadcrumbs(); render_detector()
elif view == "evac":
    render_breadcrumbs(); render_evac()
else:
    render_breadcrumbs(); render_facility_3d()
