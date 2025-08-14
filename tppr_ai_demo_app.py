
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events

# ──────────────────────────────────────────────────────────────────────────────
# App config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="OBW AI Safety Assistant", layout="wide", page_icon="🛡️")

SAFE, WARN, DANGER = "safe", "warn", "danger"
STATE_COLORS = {SAFE: "#10b981", WARN: "#f59e0b", DANGER: "#ef4444"}

# ──────────────────────────────────────────────────────────────────────────────
# Domain model (rooms & detectors)
# facility canvas = 1200 x 700; room canvas = 1000 x 600
# ──────────────────────────────────────────────────────────────────────────────
ROOMS = {
    "boiler": {
        "name": "Boiler Room",
        "facility_pos": (190, 170),
        "room_img": "assets/room_boiler.svg",
        "detectors": [
            {
                "key": "boiler_xnx",
                "model": "Honeywell XNX",
                "gas": "CH₄",
                "warn": 45.0,
                "danger": 55.0,
                "oxygen_mode": False,
                "facility_pos": (190, 170),
                "room_pos": (350, 280),
            },
        ],
    },
    "lab": {
        "name": "Process Lab",
        "facility_pos": (190, 500),
        "room_img": "assets/room_lab.svg",
        "detectors": [
            {
                "key": "lab_midas",
                "model": "Honeywell Midas",
                "gas": "NH₃",
                "warn": 35.0,
                "danger": 45.0,
                "oxygen_mode": False,
                "facility_pos": (190, 500),
                "room_pos": (520, 220),
            },
        ],
    },
    "corr_north": {
        "name": "Corridor North",
        "facility_pos": (640, 170),
        "room_img": "assets/room_corr_north.svg",
        "detectors": [
            {
                "key": "cn_sensepoint",
                "model": "Honeywell Sensepoint",
                "gas": "O₂",
                "warn": 18.0,
                "danger": 17.0,
                "oxygen_mode": True,  # lower is worse
                "facility_pos": (640, 170),
                "room_pos": (480, 160),
            },
        ],
    },
    "corr_south": {
        "name": "Corridor South",
        "facility_pos": (640, 500),
        "room_img": "assets/room_boiler.svg",  # placeholder image
        "detectors": [
            {
                "key": "cs_searchpoint",
                "model": "Honeywell Searchpoint",
                "gas": "CO",
                "warn": 30.0,
                "danger": 40.0,
                "oxygen_mode": False,
                "facility_pos": (640, 500),
                "room_pos": (500, 300),
            },
        ],
    },
    "warehouse": {
        "name": "Cylinder Store",
        "facility_pos": (1040, 170),
        "room_img": "assets/room_boiler.svg",  # placeholder image
        "detectors": [
            {
                "key": "store_searchline",
                "model": "Honeywell Searchline",
                "gas": "H₂",
                "warn": 35.0,
                "danger": 50.0,
                "oxygen_mode": False,
                "facility_pos": (1040, 170),
                "room_pos": (420, 260),
            },
        ],
    },
    "control": {
        "name": "Control Room",
        "facility_pos": (1040, 500),
        "room_img": "assets/room_boiler.svg",  # placeholder image
        "detectors": [
            {
                "key": "control_xnx",
                "model": "Honeywell XNX",
                "gas": "H₂S",
                "warn": 5.0,
                "danger": 10.0,
                "oxygen_mode": False,
                "facility_pos": (1040, 500),
                "room_pos": (560, 260),
            },
        ],
    },
}
DETECTOR_INDEX = {d["key"]: (room_key, d) for room_key, room in ROOMS.items() for d in room["detectors"]}

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data = pd.read_csv("demo_data.csv", parse_dates=["timestamp"])
if "view" not in st.session_state:
    st.session_state.view = "facility"  # facility | room | detector | evac
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
if "audio" not in st.session_state:
    st.session_state.audio = False
if "diagram_mode" not in st.session_state:
    st.session_state.diagram_mode = True  # use blueprint-style diagram
if "last_event" not in st.session_state:
    st.session_state.last_event = None
if "show_route" not in st.session_state:
    st.session_state.show_route = False

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
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

# Robust click helpers
def get_clicked_room(payload: dict):
    """Return a room_key from a plotly click payload."""
    if not isinstance(payload, dict):
        return None
    # Preferred: customdata (if provided)
    if "customdata" in payload and payload["customdata"] in ROOMS:
        return payload["customdata"]
    # Fallback: pointIndex / pointNumber
    idx = payload.get("pointIndex", payload.get("pointNumber"))
    if isinstance(idx, int):
        room_keys = list(ROOMS.keys())
        if 0 <= idx < len(room_keys):
            return room_keys[idx]
    # Last resort: nearest room by (x, y)
    x, y = payload.get("x"), payload.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return min(
            ROOMS.items(),
            key=lambda kv: (kv[1]["facility_pos"][0] - x) ** 2 + (kv[1]["facility_pos"][1] - y) ** 2,
        )[0]
    return None

def get_clicked_detector_key(room_key: str, payload: dict):
    """Return a detector key within a room from a plotly click payload."""
    if not isinstance(payload, dict) or room_key not in ROOMS:
        return None
    room = ROOMS[room_key]
    # Preferred
    if "customdata" in payload:
        return payload["customdata"]
    # Fallback by point index
    idx = payload.get("pointIndex", payload.get("pointNumber"))
    if isinstance(idx, int) and 0 <= idx < len(room["detectors"]):
        return room["detectors"][idx]["key"]
    # Nearest by (x, y)
    x, y = payload.get("x"), payload.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return min(
            room["detectors"],
            key=lambda d: (d["room_pos"][0] - x) ** 2 + (d["room_pos"][1] - y) ** 2,
        )["key"]
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Industrial facility diagram (with pipes/icons + alternate exits + route)
# ──────────────────────────────────────────────────────────────────────────────
def build_facility_diagram(show_route=False, start_room=None):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False, range=[0, 1200]),
        yaxis=dict(visible=False, range=[700, 0]),
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        paper_bgcolor="#0f172a",  # slate-900
        plot_bgcolor="#111827",   # gray-900
    )
    # Outer walls
    fig.add_shape(type="rect", x0=30, y0=30, x1=1170, y1=670, line=dict(color="#9ca3af", width=4))

    # Rooms (blocks), aligned roughly to facility_pos coordinates
    rooms_rects = {
        "boiler": (80, 80, 320, 260),
        "lab": (80, 420, 320, 640),
        "corr_north": (520, 80, 680, 260),
        "corr_south": (520, 420, 680, 640),
        "warehouse": (900, 80, 1140, 260),
        "control": (900, 420, 1140, 640),
    }
    for key, (x0, y0, x1, y1) in rooms_rects.items():
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor="#1f2937", opacity=0.95, line=dict(color="#4b5563", width=2))
        cx = (x0 + x1) / 2; cy = y0 - 10
        fig.add_annotation(x=cx, y=cy, text=ROOMS[key]["name"], showarrow=False, font=dict(color="#e5e7eb", size=12))

    # Corridors
    fig.add_shape(type="rect", x0=480, y0=80, x1=720, y1=640, fillcolor="#0b1220", line=dict(color="#374151", width=1))
    fig.add_shape(type="rect", x0=320, y0=300, x1=900, y1=420, fillcolor="#0b1220", line=dict(color="#374151", width=1))

    # Equipment icons (simple glyphs)
    # Boiler (circle + valve symbol)
    fig.add_shape(type="circle", x0=140, y0=120, x1=180, y1=160, line=dict(color="#94a3b8"))
    fig.add_annotation(x=160, y=110, text="Valve", showarrow=False, font=dict(color="#9ca3af", size=10))
    fig.add_shape(type="line", x0=180, y0=140, x1=320, y1=140, line=dict(color="#64748b", width=3))  # pipe
    # Cylinder rack in warehouse (stack of circles)
    for i in range(6):
        cx = 955 + (i%3)*20; cy = 120 + (i//3)*20
        fig.add_shape(type="circle", x0=cx-6, y0=cy-6, x1=cx+6, y1=cy+6, line=dict(color="#93c5fd"))
    fig.add_annotation(x=1000, y=100, text="Cylinders", showarrow=False, font=dict(color="#93c5fd", size=10))

    # Exits (East & West)
    fig.add_shape(type="rect", x0=1160, y0=330, x1=1170, y1=370, fillcolor="#10b981", line=dict(color="#10b981"))
    fig.add_annotation(x=1165, y=320, text="East Exit", showarrow=False, font=dict(color="#10b981"))
    fig.add_shape(type="rect", x0=30, y0=330, x1=40, y1=370, fillcolor="#10b981", line=dict(color="#10b981"))
    fig.add_annotation(x=35, y=320, text="West Exit", showarrow=False, font=dict(color="#10b981"))

    # Room pins (click targets), colored by worst detector state
    xs, ys, labels, colors = [], [], [], []
    for room_key, room in ROOMS.items():
        x, y = room["facility_pos"]
        xs.append(x); ys.append(y); labels.append(room["name"])
        ppm, _ = last_ppm(room_key)
        worst = SAFE
        for det in room["detectors"]:
            s = gas_state(det, ppm or 0)
            if s == DANGER:
                worst = DANGER; break
            elif s == WARN:
                worst = WARN
        colors.append(STATE_COLORS[worst])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=26, color=colors, line=dict(color="#111", width=1)),
        text=labels, textposition="top center",
        hoverlabel=dict(namelength=-1),
        hovertemplate="%{text}<extra></extra>",
    ))

    # Optional evacuation route overlay
    if show_route and start_room in ROOMS:
        path = compute_route(start_room)
        if path:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in path], y=[p[1] for p in path],
                mode="lines+markers", name="Evac Route",
                line=dict(width=4, dash="solid"), marker=dict(size=8),
                hovertemplate="Route point<extra></extra>"
            ))
    return fig

def compute_route(start_room_key: str):
    # Very simple routing: choose nearest exit unless that path crosses a DANGER room (corridors)
    east_exit = (1165, 350); west_exit = (35, 350)
    start = ROOMS[start_room_key]["facility_pos"]
    # Check corridor states to bias route
    def worst_state(room_key):
        ppm, _ = last_ppm(room_key)
        worst = SAFE
        for det in ROOMS[room_key]["detectors"]:
            s = gas_state(det, ppm or 0)
            if s == DANGER: return DANGER
            if s == WARN: worst = WARN
        return worst
    avoid_north = worst_state("corr_north") == DANGER
    avoid_south = worst_state("corr_south") == DANGER

    # Waypoints: go to horizontal corridor (y=360), then to exit
    mid_y = 360
    waypoints = [start, (start[0], mid_y)]
    # If starting in top/bottom, go through appropriate vertical corridor unless avoided
    if start_room_key in ("boiler", "corr_north", "warehouse"):
        # top
        if avoid_north:  # reroute via south corridor
            waypoints.append((start[0], 540))
            waypoints.append((640, 540))
        else:
            waypoints.append((640, 200))
    else:
        # bottom
        if avoid_south:
            waypoints.append((start[0], 180))
            waypoints.append((640, 180))
        else:
            waypoints.append((640, 540))

    # Choose exit by shorter distance from corridor center (640, mid_y)
    east_dist = abs(1165 - 640); west_dist = abs(35 - 640)
    exit_pt = east_exit if east_dist <= west_dist else west_exit
    waypoints.append((exit_pt[0], mid_y))
    waypoints.append(exit_pt)
    # Deduplicate consecutive points
    dedup = [waypoints[0]]
    for p in waypoints[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    return dedup

# ──────────────────────────────────────────────────────────────────────────────
# Top bar & dev controls (incl. Simulation Center)
# ──────────────────────────────────────────────────────────────────────────────
col_logo, col_title, col_gear = st.columns([1, 3, 1])
with col_logo:
    try:
        st.image("assets/obw_logo.png", use_container_width=True)
    except Exception:
        st.write("🛡️")
with col_title:
    st.markdown("<h2 style='text-align:center;margin-top:10px;'>OBW AI Safety Assistant</h2>", unsafe_allow_html=True)
with col_gear:
    if st.button("⚙️", help="Developer Controls", key="gear", use_container_width=True):
        st.session_state.free_play = not st.session_state.free_play

if st.session_state.free_play:
    with st.sidebar:
        st.markdown("### Developer Controls")
        st.toggle("Enable Audio Alerts", key="audio")
        st.toggle("Show Evacuation Route (facility)", key="show_route")
        st.markdown("---")
        # Simulation Center
        st.markdown("#### Simulation Center")
        room_choice = st.selectbox("Room ID", list(ROOMS.keys()), index=0)
        scenario = st.selectbox("Scenario", [
            "Spike: +50 for 5 ticks",
            "Ramp: +5 per tick for 15",
            "O₂ drop: -0.5 per tick (20)",
            "CO spike: +20 for 10",
        ])
        if st.button("Run Scenario"):
            apply_simulation(room_choice, scenario)
            st.toast(f"Simulated '{scenario}' in {ROOMS[room_choice]['name']}")
        if st.button("Replay Incident"):
            st.session_state.demo_index = 0
            st.session_state.incident_log = []
            for k in st.session_state.timers:
                st.session_state.timers[k] = {"state": SAFE, "danger_start": None, "warn_start": None, "danger_longest": 0}
            st.session_state.view = "facility"
            st.rerun()
        st.markdown("---")
        st.markdown("**Mode**: Free Play (click ⚙️ to hide)")

def apply_simulation(room_key: str, scenario: str):
    df = st.session_state.data.copy()
    idx = st.session_state.demo_index
    if scenario.startswith("Spike"):
        mask = (df["room"] == room_key) & (df.index >= idx) & (df.index < idx + 5)
        df.loc[mask, "ppm"] = df.loc[mask, "ppm"] + 50.0
    elif scenario.startswith("Ramp"):
        for i in range(15):
            mask = (df["room"] == room_key) & (df.index == idx + i)
            df.loc[mask, "ppm"] = df.loc[mask, "ppm"] + 5.0 * (i + 1)
    elif scenario.startswith("O₂ drop"):
        # oxygen_mode aware: only make sense for O2 detector (corr_north)
        for i in range(20):
            mask = (df["room"] == room_key) & (df.index == idx + i)
            df.loc[mask, "ppm"] = df.loc[mask, "ppm"] - 0.5 * (i + 1)
    elif scenario.startswith("CO spike"):
        for i in range(10):
            mask = (df["room"] == room_key) & (df.index == idx + i)
            df.loc[mask, "ppm"] = df.loc[mask, "ppm"] + 20.0
    st.session_state.data = df

# ──────────────────────────────────────────────────────────────────────────────
# Debug panel
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("Debug", expanded=False):
    st.write({
        "view": st.session_state.get("view"),
        "room_key": st.session_state.get("room_key"),
        "detector_key": st.session_state.get("detector_key"),
        "demo_index": st.session_state.get("demo_index"),
    })
    st.json(st.session_state.get("last_event") or {})

# ──────────────────────────────────────────────────────────────────────────────
# Views
# ──────────────────────────────────────────────────────────────────────────────
def render_breadcrumbs():
    crumbs = ["Facility"]
    if st.session_state.room_key:
        crumbs.append(ROOMS[st.session_state.room_key]["name"])
    if st.session_state.detector_key:
        _, d = DETECTOR_INDEX[st.session_state.detector_key]
        crumbs.append(d["model"])
    st.caption(" › ".join(crumbs))

def render_facility():
    render_breadcrumbs()
    st.markdown("#### Facility Overview")
    col_map, col_side = st.columns([3, 1])

    with col_map:
        fig = build_facility_diagram(show_route=st.session_state.show_route, start_room=st.session_state.room_key or "boiler")
        clicked = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            key="facility_click_v4",
            override_height=560,
        )
        if clicked:
            payload = clicked[0]
            st.session_state.last_event = payload
            # Derive by position since customdata is not always present
            rk = get_clicked_room(payload)
            if rk:
                st.session_state.room_key = rk
                st.session_state.detector_key = None
                st.session_state.view = "room"
                st.rerun()

    with col_side:
        st.markdown("### Danger Leaderboard")
        rows = []
        for det_key, t in st.session_state.timers.items():
            room_key, d = DETECTOR_INDEX[det_key]
            rows.append({
                "Detector": d["model"],
                "Room": ROOMS[room_key]["name"],
                "State": t["state"].upper(),
                "Longest Danger (s)": int(t["danger_longest"]),
            })
        df_lead = pd.DataFrame(rows).sort_values("Longest Danger (s)", ascending=False)
        df_table(df_lead)
        if st.button("Replay Incident", use_container_width=True):
            st.session_state.demo_index = 0
            st.session_state.incident_log = []
            for k in st.session_state.timers:
                st.session_state.timers[k] = {"state": SAFE, "danger_start": None, "warn_start": None, "danger_longest": 0}
            st.session_state.view = "facility"
            st.rerun()

def render_room():
    render_breadcrumbs()
    rk = st.session_state.room_key
    if rk is None or rk not in ROOMS:
        st.session_state.view = "facility"
        st.rerun()
    room = ROOMS[rk]
    st.markdown(f"### {room['name']}")

    col_map, col_ai = st.columns([2, 1])
    with col_map:
        # background plan (optional SVG)
        try:
            st.image(room["room_img"], use_container_width=True)
        except Exception:
            st.info("Room image not found — showing detector map below.")

        # Interactive detector pins
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(visible=False, range=[0, 1000]),
            yaxis=dict(visible=False, range=[600, 0]),
            margin=dict(l=0, r=0, t=0, b=0),
            height=380,
        )
        xs, ys, labels, colors = [], [], [], []
        ppm, ts = last_ppm(rk)
        for det in room["detectors"]:
            stt = gas_state(det, ppm if ppm is not None else 0)
            x, y = det["room_pos"]
            xs.append(x); ys.append(y)
            labels.append(f"{det['model']} ({det['gas']})")
            colors.append(STATE_COLORS[stt])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=22, color=colors, line=dict(color="#111", width=1)),
            hoverlabel=dict(namelength=-1),
            text=[l.split(" (")[0] for l in labels],
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
        ))
        clicked = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            key=f"room_click_{rk}_v2",
            override_height=380,
        )

        c1, c2, _ = st.columns([1, 1, 2])
        if c1.button("⬅️ Back to Facility"):
            st.session_state.view = "facility"; st.rerun()
        if c2.button("Evacuation"):
            st.session_state.view = "evac"; st.rerun()

        if clicked:
            payload = clicked[0]
            st.session_state.last_event = payload
            # Map by nearest detector (since customdata may be absent)
            x = payload.get("x"); y = payload.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                det = min(room["detectors"], key=lambda d: (d["room_pos"][0]-x)**2 + (d["room_pos"][1]-y)**2)
                st.session_state.detector_key = det["key"]
                st.session_state.view = "detector"
                st.rerun()

    with col_ai:
        # Room-level quick status (worst of detectors)
        worst = SAFE
        for det in room["detectors"]:
            d_state = gas_state(det, (ppm or 0))
            if d_state == DANGER:
                worst = DANGER; break
            elif d_state == WARN:
                worst = WARN
        if worst == DANGER:
            st.error("Danger detected. Advise immediate evacuation along nearest safe route.")
        elif worst == WARN:
            st.warning("Warning. Levels trending upward. Increase ventilation and prepare for evacuation.")
        else:
            st.success("All clear. Monitoring normal.")
        # Shortcut buttons
        st.markdown("#### Detectors")
        for det in room["detectors"]:
            if st.button(f"Open: {det['model']} — {det['gas']}", key=f"open_{det['key']}", use_container_width=True):
                st.session_state.detector_key = det["key"]
                st.session_state.view = "detector"
                st.rerun()

def render_detector():
    render_breadcrumbs()
    dk = st.session_state.detector_key
    if dk is None or dk not in DETECTOR_INDEX:
        st.session_state.view = "room"
        st.rerun()
    rk, det = DETECTOR_INDEX[dk]

    st.markdown(f"### {ROOMS[rk]['name']} • {det['model']} ({det['gas']})")
    ppm, ts = last_ppm(rk)
    stt = gas_state(det, ppm if ppm is not None else 0)
    danger_time = time_in_state_str(dk, DANGER, ts if ts is not None else datetime.utcnow())
    warn_time = time_in_state_str(dk, WARN, ts if ts is not None else datetime.utcnow())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Reading", f"{ppm:.1f} ppm" if ppm is not None else "—")
    c2.metric("State", stt.upper())
    c3.metric("Time in Danger", danger_time)
    c4.metric("Time in Warning", warn_time)

    # AI Analyst Panel
    st.markdown("### AI Analyst")
    # Rolling stats
    df_room_full = st.session_state.data[st.session_state.data["room"] == rk].iloc[: st.session_state.demo_index + 1].copy()
    if not df_room_full.empty:
        df_room_full["roll_mean"] = df_room_full["ppm"].rolling(12, min_periods=3).mean()
        df_room_full["roll_std"] = df_room_full["ppm"].rolling(12, min_periods=3).std().replace(0, np.nan)
        last_val = df_room_full["ppm"].iloc[-1]
        mean = df_room_full["roll_mean"].iloc[-1] if not np.isnan(df_room_full["roll_mean"].iloc[-1]) else df_room_full["ppm"].mean()
        std = df_room_full["roll_std"].iloc[-1] if not np.isnan(df_room_full["roll_std"].iloc[-1]) else df_room_full["ppm"].std() or 1.0
        z = (last_val - mean) / std if std else 0.0
        slope = float(np.polyfit(np.arange(min(10, len(df_room_full))), df_room_full["ppm"].tail(10), 1)[0]) if len(df_room_full) >= 2 else 0.0

        # ETA to danger threshold
        warn_thr, danger_thr = det["warn"], det["danger"]
        eta = None
        if det.get("oxygen_mode"):
            # danger when dropping to threshold
            if slope < 0 and last_val > danger_thr:
                eta = (last_val - danger_thr) / (-slope)
        else:
            if slope > 0 and last_val < danger_thr:
                eta = (danger_thr - last_val) / slope

        c1, c2, c3 = st.columns(3)
        c1.metric("Anomaly (z-score)", f"{z:.2f}")
        c2.metric("Slope (Δppm/tick)", f"{slope:.2f}")
        c3.metric("ETA to Danger (ticks)", "≤1" if eta is not None and eta <= 1 else (f"{eta:.1f}" if eta is not None else "—"))

        # What-if control
        with st.expander("What‑if: Ventilation & Mitigation"):
            reduction = st.slider("Reduce slope (%)", 0, 90, 35, step=5, help="Simulate ventilation or process cutback")
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
    df_room = df_room_full
    pred = prediction_curve(rk, 15)

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

    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("⬅️ Room"):
        st.session_state.view = "room"; st.rerun()
    if c2.button("Evacuation"):
        st.session_state.view = "evac"; st.rerun()

    if stt == DANGER:
        st.session_state.view = "evac"; st.rerun()

def render_evac():
    render_breadcrumbs()
    st.markdown("### Evacuation Mode")
    st.info("AI is guiding evacuation. Follow green exit markers.")
    cols = st.columns([3, 1])
    with cols[0]:
        fig = build_facility_diagram(show_route=True, start_room=st.session_state.room_key or "boiler")
        st.plotly_chart(fig, use_container_width=True)
    with cols[1]:
        if st.button("Return to Room"):
            st.session_state.view = "room"; st.rerun()
        if st.button("Back to Facility"):
            st.session_state.view = "facility"; st.rerun()
    st.markdown("#### AI Evacuation Guidance")
    rk = st.session_state.room_key or "boiler"
    st.write(f"Starting from **{ROOMS[rk]['name']}**. Nearest exit determined dynamically; avoiding danger corridors.")

# ──────────────────────────────────────────────────────────────────────────────
# Demo tick (simulates time)
# ──────────────────────────────────────────────────────────────────────────────
def demo_tick():
    if st.session_state.demo_index < len(st.session_state.data) - 1:
        st.session_state.demo_index += 1

demo_tick()

# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────
def render_breadcrumbs():
    crumbs = ["Facility"]
    if st.session_state.room_key:
        crumbs.append(ROOMS[st.session_state.room_key]["name"])
    if st.session_state.detector_key:
        _, d = DETECTOR_INDEX[st.session_state.detector_key]
        crumbs.append(d["model"])
    st.caption(" › ".join(crumbs))

view = st.session_state.view
# Update timers each loop using latest per-room ppm
now_ts = datetime.utcnow()
for room_key, room in ROOMS.items():
    ppm, ts = last_ppm(room_key)
    ts_use = ts if ts is not None else now_ts
    for det in room["detectors"]:
        stt = gas_state(det, ppm if ppm is not None else 0)
        update_timers(det["key"], stt, ts_use)

if view == "facility":
    render_facility()
elif view == "room":
    render_room()
elif view == "detector":
    render_detector()
elif view == "evac":
    render_evac()
else:
    render_facility()
