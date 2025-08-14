import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# --- Config ---
st.set_page_config(page_title="OBW AI Safety Assistant", layout="wide", page_icon="🛡️")
SAFE, WARN, DANGER = "safe", "warn", "danger"
STATE_COLORS = {SAFE: "#10b981", WARN: "#f59e0b", DANGER: "#ef4444"}

# --- Data ---
ROOMS = {
    "boiler": {"name": "Boiler Room", "pos": (200, 200)},
    "lab": {"name": "Process Lab", "pos": (200, 500)},
    "corr_north": {"name": "Corridor North", "pos": (600, 200)},
    "corr_south": {"name": "Corridor South", "pos": (600, 500)},
    "warehouse": {"name": "Cylinder Store", "pos": (1000, 200)},
    "control": {"name": "Control Room", "pos": (1000, 500)},
}

if "data" not in st.session_state:
    st.session_state.data = pd.read_csv("demo_data.csv", parse_dates=["timestamp"])
if "view" not in st.session_state:
    st.session_state.view = "facility"
if "room_key" not in st.session_state:
    st.session_state.room_key = None

# --- Helpers ---
def build_facility_diagram():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False, range=[0, 1200]),
        yaxis=dict(visible=False, range=[700, 0]),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        paper_bgcolor="#e5e7eb",
        plot_bgcolor="#f3f4f6",
    )
    fig.add_shape(type="rect", x0=50, y0=50, x1=1150, y1=650, line=dict(color="#374151", width=3))
    xs, ys, labels, custom, colors = [], [], [], [], []
    for key, room in ROOMS.items():
        x, y = room["pos"]
        fig.add_shape(type="rect", x0=x-80, y0=y-50, x1=x+80, y1=y+50,
                      fillcolor="#d1d5db", opacity=0.8, line=dict(color="#4b5563", width=2))
        fig.add_annotation(x=x, y=y-70, text=room["name"], showarrow=False, font=dict(size=12, color="#111827"))
        xs.append(x); ys.append(y); labels.append(room["name"])
        custom.append(key)
        colors.append(STATE_COLORS[SAFE])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=20, color=colors, line=dict(color="#111", width=1)),
        text=labels, textposition="top center",
        customdata=custom, hovertemplate="%{text}<extra></extra>",
    ))
    return fig

# --- Debug ---
with st.expander("Debug", expanded=False):
    st.write({
        "view": st.session_state.get("view"),
        "room_key": st.session_state.get("room_key"),
    })
    st.json(st.session_state.get("last_event") or {})

# --- Views ---
def render_facility():
    st.subheader("Facility Overview")
    fig = build_facility_diagram()
    clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="facility_click", override_height=520)
    if clicked and clicked[0].get("customdata"):
        payload = clicked[0]
        st.session_state.last_event = payload
        room_clicked = None
        if isinstance(payload, dict) and "customdata" in payload:
            room_clicked = payload["customdata"]
        else:
            idx = payload.get("pointIndex", payload.get("pointNumber")) if isinstance(payload, dict) else None
            if idx is not None:
                room_keys = list(ROOMS.keys())
                if 0 <= idx < len(room_keys):
                    room_clicked = room_keys[idx]
            if room_clicked is None and isinstance(payload, dict):
                x = payload.get("x"); y = payload.get("y")
                if x is not None and y is not None:
                    room_clicked = min(ROOMS.items(), key=lambda kv: (kv[1]["pos"][0]-x)**2 + (kv[1]["pos"][1]-y)**2)[0]
        st.session_state.room_key = room_clicked or st.session_state.room_key
        st.session_state.view = "room"
        st.experimental_rerun()

def render_room():
    rk = st.session_state.room_key
    if not rk or rk not in ROOMS:
        st.session_state.view = "facility"
        st.experimental_rerun()
    st.subheader(ROOMS[rk]["name"])
    if st.button("⬅ Back to Facility"):
        st.session_state.view = "facility"
        st.experimental_rerun()
    st.write("Room details and detectors will go here.")

# --- Router ---
if st.session_state.view == "facility":
    render_facility()
elif st.session_state.view == "room":
    render_room()


