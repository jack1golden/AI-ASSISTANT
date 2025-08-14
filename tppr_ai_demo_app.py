import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events

########################
# App config & constants
########################
st.set_page_config(
    page_title="OBW AI Safety Assistant",
    layout="wide",
    page_icon="🛡️",
)

SAFE, WARN, DANGER = "safe", "warn", "danger"
STATE_COLORS = {SAFE: "#10b981", WARN: "#f59e0b", DANGER: "#ef4444"}

########################
# Domain model (rooms & detectors)
########################
# Facility layout is a 1200x700 canvas (same coordinate system as v1)
ROOMS = {
    "boiler": {
        "name": "Boiler Room",
        "facility_pos": (190, 170),  # where the room sits on the facility canvas
        "room_img": "assets/room_boiler.svg",
        # detectors laid out within the room canvas (0..1000 x 0..600)
        "detectors": [
            {
                "key": "boiler_xnx",
                "model": "Honeywell XNX",
                "gas": "CH₄",
                "warn": 45.0,
                "danger": 55.0,
                "oxygen_mode": False,
                "facility_pos": (190, 170),  # pin for facility view
                "room_pos": (350, 280),      # pin for room view
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
                "oxygen_mode": True,   # reverse thresholds (lower is worse)
                "facility_pos": (640, 170),
                "room_pos": (480, 160),
            },
        ],
    },
    "corr_south": {
        "name": "Corridor South",
        "facility_pos": (640, 500),
        "room_img": "assets/room_boiler.svg",  # placeholder
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
        "room_img": "assets/room_boiler.svg",  # placeholder
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
        "room_img": "assets/room_boiler.svg",  # placeholder
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

# Fast index for detectors by key
DETECTOR_INDEX = {d["key"]: (room_key, d) for room_key, room in ROOMS.items() for d in room["detectors"]}

########################
# Session state
########################
if "data" not in st.session_state:
    st.session_state.data = pd.read_csv("demo_data.csv", parse_dates=["timestamp"])  # expects columns: timestamp, room, ppm
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

########################
# Helpers
########################

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
    return pd.DataFrame(
        {
            "timestamp": [start_ts + timedelta(minutes=i) for i in range(1, horizon + 1)],
            "ppm": [max(0.0, start_ppm + slope * i) for i in range(1, horizon + 1)],
        }
    )


def df_table(df: pd.DataFrame):
    return st.dataframe(df, use_container_width=True, hide_index=True)

########################
# Top bar & dev controls
########################
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
        st.markdown("### Developer Controls (Free Play)")
        st.toggle("Enable Audio Alerts", key="audio")
        if st.button("Replay Incident"):
            st.session_state.demo_index = 0
            st.session_state.incident_log = []
            for k in st.session_state.timers:
                st.session_state.timers[k] = {"state": SAFE, "danger_start": None, "warn_start": None, "danger_longest": 0}
            st.session_state.view = "facility"
            st.rerun()
        st.markdown("---")
        st.markdown("**Manual Triggers**")
        # Nudge next 5 records for a given room upward
        for room_key in ROOMS.keys():
            if st.button(f"Trigger Danger spike in: {ROOMS[room_key]['name']}"):
                idx = st.session_state.demo_index
                df = st.session_state.data.copy()
                mask = (df["room"] == room_key) & (df.index >= idx) & (df.index < idx + 5)
                df.loc[mask, "ppm"] = df.loc[mask, "ppm"] + 50.0
                st.session_state.data = df
                st.toast(f"Forced danger for {ROOMS[room_key]['name']}")
        st.markdown("---")
        st.markdown("**Mode**: Free Play (click ⚙️ to hide)")

########################
# Views
########################

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
        try:
            st.image("assets/facility.svg", use_container_width=True)
        except Exception:
            st.info("Facility map image not found — using coordinate canvas below.")
        # Build a Plotly scatter of room pins (click to enter room)
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(visible=False, range=[0, 1200]),
            yaxis=dict(visible=False, range=[700, 0]),
            margin=dict(l=0, r=0, t=0, b=0),
            height=520,
        )
        xs, ys, labels, colors, custom = [], [], [], [], []
        now_ts = None
        # Determine state per room by aggregating most severe detector state
        for room_key, room in ROOMS.items():
            ppm, ts = last_ppm(room_key)
            now_ts = ts or datetime.utcnow()
            # default to SAFE if no data yet
            worst_state = SAFE
            for det in room["detectors"]:
                stt = gas_state(det, ppm if ppm is not None else 0)
                if stt == DANGER:
                    worst_state = DANGER
                    break
                elif stt == WARN:
                    worst_state = WARN
            # update timers for each detector too
            for det in room["detectors"]:
                stt = gas_state(det, ppm if ppm is not None else 0)
                update_timers(det["key"], stt, now_ts)
            x, y = room["facility_pos"]
            xs.append(x)
            ys.append(y)
            labels.append(room["name"])
            colors.append(STATE_COLORS[worst_state])
            custom.append(room_key)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=18, color=colors, line=dict(color="#111", width=1)),
                text=labels,
                textposition="top center",
                customdata=custom,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="facility_click_v2")
        st.plotly_chart(fig, use_container_width=True)
        if clicked:
            room_clicked = clicked[0]["customdata"]
            st.session_state.room_key = room_clicked
            st.session_state.detector_key = None
            st.session_state.view = "room"
            st.rerun()

    with col_side:
        st.markdown("### Danger Leaderboard")
        rows = []
        for det_key, t in st.session_state.timers.items():
            room_key, d = DETECTOR_INDEX[det_key]
            rows.append(
                {
                    "Detector": d["model"],
                    "Room": ROOMS[room_key]["name"],
                    "State": t["state"].upper(),
                    "Longest Danger (s)": int(t["danger_longest"]),
                }
            )
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
        try:
            st.image(room["room_img"], use_container_width=True)
        except Exception:
            st.info("Room image not found — showing detector map below.")
        # Clickable detectors within the room
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(visible=False, range=[0, 1000]),
            yaxis=dict(visible=False, range=[600, 0]),
            margin=dict(l=0, r=0, t=0, b=0),
            height=360,
        )
        xs, ys, labels, colors, custom = [], [], [], [], []
        ppm, ts = last_ppm(rk)
        for det in room["detectors"]:
            stt = gas_state(det, ppm if ppm is not None else 0)
            x, y = det["room_pos"]
            xs.append(x)
            ys.append(y)
            labels.append(f"{det['model']} ({det['gas']})")
            colors.append(STATE_COLORS[stt])
            custom.append(det["key"])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=16, color=colors, line=dict(color="#111", width=1)),
                text=[l.split(" (")[0] for l in labels],
                textposition="top center",
                customdata=custom,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=f"room_click_{rk}")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.button("⬅️ Back to Facility"):
            st.session_state.view = "facility"
            st.rerun()
        if c2.button("Evacuation"):
            st.session_state.view = "evac"
            st.rerun()
        if clicked:
            det_key = clicked[0]["customdata"]
            st.session_state.detector_key = det_key
            st.session_state.view = "detector"
            st.rerun()

    with col_ai:
        # Room level quick status (worst-of detectors)
        worst = SAFE
        for det in room["detectors"]:
            d_state = gas_state(det, last_ppm(rk)[0] or 0)
            if d_state == DANGER:
                worst = DANGER
                break
            elif d_state == WARN:
                worst = WARN
        if worst == DANGER:
            st.error("Danger detected. Advise immediate evacuation along nearest safe route.")
        elif worst == WARN:
            st.warning("Warning. Levels trending upward. Increase ventilation and prepare for evacuation.")
        else:
            st.success("All clear. Monitoring normal.")
        # Detectors list
        st.markdown("#### Detectors")
        for det in room["detectors"]:
            btn = st.button(f"Open: {det['model']} — {det['gas']}", key=f"open_{det['key']}", use_container_width=True)
            if btn:
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

    st.markdown("### Live Readings & 15‑min Prediction")
    df_room = st.session_state.data[st.session_state.data["room"] == rk].iloc[: st.session_state.demo_index + 1]
    pred = prediction_curve(rk, 15)

    fig = go.Figure()
    if not df_room.empty:
        fig.add_trace(
            go.Scatter(x=df_room["timestamp"], y=df_room["ppm"], mode="lines+markers", name="Live", line=dict(width=3))
        )
    if not pred.empty:
        fig.add_trace(
            go.Scatter(x=pred["timestamp"], y=pred["ppm"], mode="lines", name="Prediction", line=dict(dash="dash"))
        )

    warn = det["warn"]
    danger = det["danger"]
    if det.get("oxygen_mode"):
        fig.add_hrect(y0=-1e6, y1=danger, fillcolor=STATE_COLORS[DANGER], opacity=0.10, line_width=0)
        fig.add_hrect(y0=danger, y1=warn, fillcolor=STATE_COLORS[WARN], opacity=0.08, line_width=0)
    else:
        fig.add_hrect(y0=danger, y1=1e6, fillcolor=STATE_COLORS[DANGER], opacity=0.10, line_width=0)
        fig.add_hrect(y0=warn, y1=danger, fillcolor=STATE_COLORS[WARN], opacity=0.08, line_width=0)

    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("⬅️ Room"):
        st.session_state.view = "room"
        st.rerun()
    if c2.button("Evacuation"):
        st.session_state.view = "evac"
        st.rerun()

    if stt == DANGER:
        st.session_state.view = "evac"
        st.rerun()


def render_evac():
    render_breadcrumbs()
    st.markdown("### Evacuation Mode")
    st.info("AI is guiding evacuation. Follow green exit markers.")
    cols = st.columns([3, 1])
    with cols[0]:
        try:
            st.image("assets/facility.svg", use_container_width=True)
        except Exception:
            st.write("Facility map unavailable")
    with cols[1]:
        if st.button("Return to Room"):
            st.session_state.view = "room"
            st.rerun()
        if st.button("Back to Facility"):
            st.session_state.view = "facility"
            st.rerun()
    st.markdown("#### AI Evacuation Guidance")
    rk = st.session_state.room_key or "boiler"
    st.write(f"Starting from **{ROOMS[rk]['name']}**. Nearest exit is **East Exit**. Avoid Corridor South if alarms are active.")

########################
# Demo tick (simulates time)
########################

def demo_tick():
    if st.session_state.demo_index < len(st.session_state.data) - 1:
        st.session_state.demo_index += 1

demo_tick()

########################
# Router
########################
view = st.session_state.view
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


