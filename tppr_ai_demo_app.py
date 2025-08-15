
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="OBW AI — Pharma 2.5D", layout="wide", page_icon="🏭")

ROOM_DISPLAY = {
    "entry":"Entry","prod1":"Production 1","room12":"Room 12",
    "room1":"Room 1","prod2":"Production 2","room3":"Room 3","room2":"Room 2"
}
COLOR = {"healthy":"#10b981","inhibit":"#f59e0b","alarm":"#ef4444","fault":"#fb923c","over":"#e5e7eb"}

# Overlay button positions (center of each room) in background coords
ROOM_CENTERS = {"entry":(210,665),"prod1":(470,180),"room12":(720,180),"room1":(1010,180),
                "prod2":(470,380),"room3":(720,380),"room2":(1010,380)}
# Polygons for hit-testing (rough rectangles)
ROOMS_POLY = {
  "entry":[(100,600),(320,600),(320,730),(100,730)],
  "prod1":[(380,100),(560,100),(560,260),(380,260)],
  "room12":[(580,100),(860,100),(860,260),(580,260)],
  "room1":[(900,100),(1120,100),(1120,260),(900,260)],
  "prod2":[(380,300),(560,300),(560,460),(380,460)],
  "room3":[(580,300),(860,300),(860,460),(580,460)],
  "room2":[(900,300),(1120,300),(1120,460),(900,460)],
}

THRESH = {
    "prod1":{"warn":35.0,"danger":45.0,"oxygen_mode":False},
    "prod2":{"warn":35.0,"danger":45.0,"oxygen_mode":False},
    "room1":{"warn":30.0,"danger":40.0,"oxygen_mode":False},
    "room2":{"warn":30.0,"danger":40.0,"oxygen_mode":False},
    "room3":{"warn":30.0,"danger":40.0,"oxygen_mode":False},
    "room12":{"warn":30.0,"danger":40.0,"oxygen_mode":False},
    "entry":{"warn":30.0,"danger":40.0,"oxygen_mode":False},
}

def init_state():
    if "data" not in st.session_state:
        st.session_state.data = pd.read_csv("demo_data.csv", parse_dates=["timestamp"])
    st.session_state.setdefault("view","facility")
    st.session_state.setdefault("room_key",None)
    st.session_state.setdefault("detector_key",None)
    st.session_state.setdefault("demo_index",0)
    st.session_state.setdefault("dev",False)
    st.session_state.setdefault("evac_route_enabled",False)  # renamed to avoid conflicts
    st.session_state.setdefault("wireframe",False)
    st.session_state.setdefault("last_event",None)
init_state()

def _safe_float(v):
    try: return float(v)
    except: return None

def resolve_room_click(payload: dict):
    if not isinstance(payload, dict): return None
    cd = payload.get("customdata")
    if isinstance(cd,str) and cd in ROOMS_POLY: return cd
    x=_safe_float(payload.get("x")); y=_safe_float(payload.get("y"))
    if x is None or y is None: return None
    # nearest center fallback
    best=None; bestd=1e18
    for rk,(cx,cy) in ROOM_CENTERS.items():
        d=(cx-x)**2+(cy-y)**2
        if d<bestd: bestd=d; best=rk
    return best

def room_status(rk: str, idx: int) -> str:
    df = st.session_state.data; sub = df[df["room"]==rk]
    if sub.empty: return "fault"
    local_idx = min(len(sub)-1, idx)
    val=float(sub.iloc[local_idx]["ppm"])
    thr=THRESH.get(rk, {"warn":30.0,"danger":40.0,"oxygen_mode":False})
    warn,danger,is_o2 = thr["warn"],thr["danger"],thr["oxygen_mode"]
    if is_o2:
        if val<=danger: return "alarm"
        if val<=warn: return "inhibit"
        return "healthy"
    else:
        if val>=danger*1.15: return "over"
        if val>=danger: return "alarm"
        if val>=warn: return "inhibit"
        return "healthy"

def apply_simulation(room_key: str, mode: str, intensity: int, duration: int):
    df = st.session_state.data.copy()
    idx = st.session_state.demo_index
    sub_idx = df.index[df["room"]==room_key]
    # take next N indexes within this room's rows
    future = sub_idx[sub_idx >= sub_idx.min()+idx][:duration]
    for j,ix in enumerate(future):
        if mode=="Spike":
            df.loc[ix,"ppm"] = float(df.loc[ix,"ppm"]) + float(intensity)
        elif mode=="Ramp":
            df.loc[ix,"ppm"] = float(df.loc[ix,"ppm"]) + float(intensity)*(j+1)/max(1,duration)
        elif mode=="O₂ drop":
            df.loc[ix,"ppm"] = float(df.loc[ix,"ppm"]) - float(intensity)*(j+1)/max(1,duration)
        elif mode=="CO spike":
            df.loc[ix,"ppm"] = float(df.loc[ix,"ppm"]) + float(intensity)*0.4
    st.session_state.data = df

def build_facility():
    # Plotly figure with background image and overlay "button" markers + polygons
    W,H = 1200,700
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False, range=[0,W]),
        yaxis=dict(visible=False, range=[H,0]),
        margin=dict(l=0,r=0,t=0,b=0),
        height=560, paper_bgcolor="#0f172a", plot_bgcolor="#111827",
        images=[dict(source="assets/facility_2p5d.png", xref="x", yref="y",
                     x=0,y=0,sizex=W,sizey=H,sizing="stretch", layer="below")]
    )
    # Draw semi-transparent fills to indicate status
    for rk, poly in ROOMS_POLY.items():
        status = room_status(rk, st.session_state.demo_index)
        xs=[p[0] for p in poly]+[poly[0][0]]
        ys=[p[1] for p in poly]+[poly[0][1]]
        fig.add_trace(go.Scatter(
            x=xs,y=ys,mode="lines",fill="toself",
            fillcolor=COLOR[status]+"80",  # translucent
            line=dict(color="#4b5563", width=2),
            hovertemplate=f"{ROOM_DISPLAY[rk]}<extra></extra>",
            customdata=[rk]*len(xs),
            showlegend=False
        ))
    # Overlay round "buttons" at centers
    cx=[ROOM_CENTERS[r][0] for r in ROOM_CENTERS]
    cy=[ROOM_CENTERS[r][1] for r in ROOM_CENTERS]
    labels=[ROOM_DISPLAY[r] for r in ROOM_CENTERS]
    cds=list(ROOM_CENTERS.keys())
    fig.add_trace(go.Scatter(
        x=cx,y=cy,mode="markers+text",
        marker=dict(size=24, line=dict(color="#0b0b0b", width=1)),
        text=labels, textposition="bottom center",
        hovertemplate="%{text}<extra></extra>",
        customdata=cds,
        showlegend=False
    ))
    return fig

def render_facility():
    st.markdown("#### Facility — 2.5D")
    fig = build_facility()
    clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="facility_click", override_height=560)
    if clicked:
        payload = clicked[0] or {}
        st.session_state.last_event = payload
        rk = resolve_room_click(payload)
        if rk:
            st.session_state.room_key = rk
            st.session_state.view = "room"

def render_room():
    rk = st.session_state.room_key
    if not rk or rk not in ROOM_DISPLAY:
        st.session_state.view="facility"; return
    st.markdown(f"### {ROOM_DISPLAY[rk]}")
    col_left, col_right = st.columns([2,1])

    with col_left:
        # 2.5D room image with a clickable detector "button"
        st.image(f"assets/{rk}_room.png", use_container_width=True)
        # detector button & evac
        b1,b2 = st.columns(2)
        if b1.button("Open Detector", key=f"detbtn_{rk}"):
            st.session_state.view="detector"
            st.session_state.detector_key=f"{rk}_det1"
        if b2.button("Back to Facility", key=f"backfac_{rk}"):
            st.session_state.view="facility"

    with col_right:
        # AI mini analyst + latest status
        df = st.session_state.data
        sub = df[df["room"]==rk].iloc[: st.session_state.demo_index + 1].copy()
        status = room_status(rk, st.session_state.demo_index)
        st.metric("Status", status.upper())
        if not sub.empty:
            last = float(sub["ppm"].iloc[-1])
            st.metric("Last reading", f"{last:.1f} ppm")
        st.markdown("#### What‑if Ventilation")
        reduction = st.slider("Reduce slope (%)", 0, 90, 35, step=5, key=f"vent_{rk}")
        if not sub.empty:
            tail = sub["ppm"].tail(10).values
            slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail)>=2 else 0.0
            adj = slope*(1-reduction/100)
            st.caption(f"Slope now: {adj:+.2f} Δppm/tick")

    # live chart
    st.markdown("### Live Readings & Thresholds")
    df = st.session_state.data
    sub = df[df["room"]==rk].iloc[: st.session_state.demo_index + 1].copy()
    fig = go.Figure()
    if not sub.empty:
        sub["roll"]=sub["ppm"].rolling(12, min_periods=3).mean()
        fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["ppm"], mode="lines+markers", name="Live", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["roll"], mode="lines", name="Rolling", line=dict(dash="dot")))
    thr=THRESH.get(rk, {"warn":30.0,"danger":40.0,"oxygen_mode":False})
    warn,danger,is_o2 = thr["warn"],thr["danger"],thr["oxygen_mode"]
    if is_o2:
        fig.add_hrect(y0=-1e6, y1=danger, fillcolor=COLOR["alarm"], opacity=0.10, line_width=0)
        fig.add_hrect(y0=danger, y1=warn, fillcolor=COLOR["inhibit"], opacity=0.08, line_width=0)
    else:
        fig.add_hrect(y0=danger, y1=1e6, fillcolor=COLOR["alarm"], opacity=0.10, line_width=0)
        fig.add_hrect(y0=warn, y1=danger, fillcolor=COLOR["inhibit"], opacity=0.08, line_width=0)
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

def render_detector():
    rk = st.session_state.room_key
    if not rk: st.session_state.view="facility"; return
    st.markdown(f"### Detector — {ROOM_DISPLAY[rk]}")
    df = st.session_state.data
    sub = df[df["room"]==rk].iloc[: st.session_state.demo_index + 1].copy()
    if sub.empty:
        st.warning("No data for this room yet."); 
    else:
        last = sub["ppm"].iloc[-1]; mean=sub["ppm"].rolling(12, min_periods=3).mean().iloc[-1]
        std = sub["ppm"].rolling(12, min_periods=3).std().replace(0,np.nan).iloc[-1]
        if np.isnan(mean): mean = float(sub["ppm"].mean())
        if np.isnan(std) or std==0: std = float(sub["ppm"].std() or 1.0)
        z = (last - mean)/std
        tail = sub["ppm"].tail(10).values
        slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail)>=2 else 0.0
        c1,c2,c3 = st.columns(3)
        c1.metric("Live", f"{float(last):.1f} ppm")
        c2.metric("Anomaly (z)", f"{z:.2f}")
        c3.metric("Slope", f"{slope:+.2f} Δppm/tick")
    c1,c2 = st.columns(2)
    if c1.button("⬅️ Back to Room", key=f"backroom_{rk}"):
        st.session_state.view="room"
    if c2.button("⬅️ Facility", key=f"backfac2_{rk}"):
        st.session_state.view="facility"

# Header + Dev
col1,col2,col3 = st.columns([1,3,1])
with col1: st.write("🏭")
with col2: st.markdown("<h2 style='text-align:center;margin-top:10px;'>Pharma Facility — 2.5D Interactive</h2>", unsafe_allow_html=True)
with col3:
    if st.button("⚙️", key="gear"):
        st.session_state.dev = not st.session_state.get("dev", False)
    st.checkbox("Dev", key="dev", value=st.session_state.get("dev", False))

if st.session_state.get("dev", False):
    with st.sidebar:
        st.markdown("### Simulation Center")
        room_choice = st.selectbox("Room", list(ROOM_DISPLAY.keys()), key="sim_room")
        mode = st.selectbox("Mode", ["Spike","Ramp","O₂ drop","CO spike"], key="sim_mode")
        intensity = st.slider("Intensity", 5, 100, 40, key="sim_int")
        duration = st.slider("Duration (ticks)", 3, 30, 10, key="sim_dur")
        if st.button("Run Simulation", key="sim_run"):
            apply_simulation(room_choice, mode, intensity, duration)
            st.toast(f"Simulated {mode} in {ROOM_DISPLAY[room_choice]}")

# demo tick
if st.session_state.demo_index < len(st.session_state.data)-1:
    st.session_state.demo_index += 1

# Router
if st.session_state.view=="facility":
    render_facility()
elif st.session_state.view=="room":
    render_room()
elif st.session_state.view=="detector":
    render_detector()
else:
    render_facility()
