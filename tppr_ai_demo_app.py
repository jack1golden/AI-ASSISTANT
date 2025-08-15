
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="OBW AI Safety Assistant — Floorplan", layout="wide", page_icon="🏭")

SAFE, WARN, DANGER = "healthy", "inhibit", "alarm"
COLORS = {"healthy":"#10b981","inhibit":"#f59e0b","alarm":"#ef4444","fault":"#fb923c","over":"#e5e7eb"}

W, H = 1400, 840  # must match the background image
ROOMS_POLY = {
    "entry":        [(140,620),(360,620),(360,780),(140,780)],
    "gowning":      [(380,620),(620,620),(620,780),(380,780)],
    "reactor":      [(140,140),(420,140),(420,360),(140,360)],
    "cleanroom":    [(440,140),(760,140),(760,360),(440,360)],
    "granulation":  [(780,140),(1100,140),(1100,360),(780,360)],
    "packaging":    [(440,380),(760,380),(760,600),(440,600)],
    "warehouse":    [(780,380),(1100,380),(1100,600),(780,600)],
    "utilities":    [(1120,380),(1260,380),(1260,600),(1120,600)],
    "qc":           [(1120,140),(1260,140),(1260,360),(1120,360)],
}
ROOM_DISPLAY = {
    "entry":"Entry / Security",
    "gowning":"Gowning",
    "reactor":"Reactor Suite",
    "cleanroom":"Cleanroom (Grade C)",
    "granulation":"Granulation",
    "packaging":"Packaging Hall",
    "warehouse":"Warehouse (API)",
    "utilities":"Utilities / Boilers",
    "qc":"QC Lab",
}

# ── State
if "data" not in st.session_state:
    st.session_state.data = pd.read_csv("demo_data.csv", parse_dates=["timestamp"])
if "view" not in st.session_state:
    st.session_state.view = "facility"
if "room_key" not in st.session_state:
    st.session_state.room_key = None
if "demo_index" not in st.session_state:
    st.session_state.demo_index = 0
if "free_play" not in st.session_state:
    st.session_state.free_play = False
if "last_event" not in st.session_state:
    st.session_state.last_event = None

# ── Helpers
def _center(poly):
    xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
    return sum(xs)/len(xs), sum(ys)/len(ys)

def worst_status_for_room(rk: str) -> str:
    df = st.session_state.data
    sub = df[df["room"] == rk]
    if sub.empty:
        return "fault"
    idx = min(len(sub)-1, st.session_state.demo_index)
    val = float(sub.iloc[idx]["ppm"])
    # thresholds (demo)
    thr = {
        "reactor": (35, 45, False),
        "granulation": (30, 40, False),
        "cleanroom": (18, 17, True),  # oxygen mode (downwards)
        "packaging": (35, 45, False),
        "warehouse": (35, 50, False),
        "utilities": (45, 55, False),
        "entry": (40, 50, False),
        "gowning": (40, 50, False),
        "qc": (30, 40, False),
    }[rk]
    warn, danger, o2 = thr
    if o2:
        if val <= danger: return "alarm"
        if val <= warn: return "inhibit"
        return "healthy"
    else:
        if val >= danger*1.15: return "over"
        if val >= danger: return "alarm"
        if val >= warn: return "inhibit"
        return "healthy"

def resolve_room_click(payload: dict):
    if not isinstance(payload, dict): return None
    cd = payload.get("customdata")
    if isinstance(cd, str) and cd in ROOMS_POLY: return cd
    x, y = payload.get("x"), payload.get("y")
    try:
        x = float(x); y = float(y)
    except (TypeError, ValueError):
        return None
    best, bestd = None, float("inf")
    for rk, poly in ROOMS_POLY.items():
        cx, cy = _center(poly)
        d = (cx-x)**2 + (cy-y)**2
        if d < bestd: bestd, best = d, rk
    return best

def build_facility_floorplan(status_by_room: dict, wireframe: bool=False):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False, range=[0, W]),
        yaxis=dict(visible=False, range=[H, 0]),
        margin=dict(l=0, r=260, t=0, b=0),
        height=640,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        dragmode=False,
        showlegend=False,
    )
    # Background image of the pharma layout
    fig.add_layout_image(
        dict(
            source="assets/floorplan.png",
            xref="x", yref="y",
            x=0, y=0,
            sizex=W, sizey=H,
            sizing="stretch",
            opacity=1.0,
            layer="below",
        )
    )
    # Room overlays (semi-transparent so model shows through)
    for rk, poly in ROOMS_POLY.items():
        xs=[p[0] for p in poly]+[poly[0][0]]
        ys=[p[1] for p in poly]+[poly[0][1]]
        status = status_by_room.get(rk,"healthy")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor=COLORS[status]+"CC",  # add alpha
            line=dict(color="#111827", width=2),
            hovertemplate=f"{ROOM_DISPLAY[rk]}<extra></extra>",
            customdata=[rk]*len(xs),
            name=rk,
            showlegend=False,
        ))
        cx, cy = _center(poly)
        fig.add_annotation(x=cx, y=cy, text=f"<b>{ROOM_DISPLAY[rk]}</b>",
                           showarrow=False, font=dict(color="#e5e7eb", size=13))
    # Legend to the right
    legend_items=[("Healthy","#10b981"),("Inhibit","#f59e0b"),("Alarm","#ef4444"),
                  ("Fault","#fb923c"),("Activated / Over Range","#e5e7eb")]
    y_cursor=80
    for label,color in legend_items:
        fig.add_shape(type="rect", x0=W+20, y0=y_cursor, x1=W+50, y1=y_cursor+20,
                      fillcolor=color, line=dict(color="#334155"))
        fig.add_annotation(x=W+60, y=y_cursor+10, text=label, showarrow=False,
                           font=dict(color="#e5e7eb", size=12), xanchor="left", yanchor="middle")
        y_cursor += 30
    if wireframe:
        fig.add_shape(type="rect", x0=120, y0=120, x1=W-120, y1=H-120, line=dict(color="#64748b", width=1, dash="dot"))
    return fig

# ── Top bar
col_l, col_c, col_r = st.columns([1,3,1])
with col_l: st.write("🏭")
with col_c: st.markdown("<h2 style='text-align:center;margin-top:10px;'>Pharma Facility — Floorplan</h2>", unsafe_allow_html=True)
with col_r:
    if st.button("Dev", key="dev_btn", use_container_width=True):
        st.session_state.free_play = not st.session_state.get("free_play", False)

# ── Dev sidebar
if st.session_state.get("free_play", False):
    with st.sidebar:
        st.markdown("### Developer Controls")
        wire = st.checkbox("Wireframe", key="dbg_wire", value=False)
        st.markdown("#### Simulation Center")
        rk = st.selectbox("Room", list(ROOMS_POLY.keys()), key="sim_room")
        mode = st.selectbox("Mode", ["Spike","Ramp","O₂ drop","CO spike"], key="sim_mode")
        intensity = st.slider("Intensity", 5, 100, 50, step=5, key="sim_int")
        duration = st.slider("Duration (ticks)", 3, 30, 10, key="sim_dur")
        if st.button("Run Simulation", key="sim_run", use_container_width=True):
            df = st.session_state.data.copy()
            idx = st.session_state.demo_index
            if mode == "Spike":
                mask = (df["room"] == rk) & (df.index >= idx) & (df.index < idx + duration)
                df.loc[mask, "ppm"] = df.loc[mask, "ppm"] + float(intensity)
            elif mode == "Ramp":
                for i in range(duration):
                    m = (df["room"] == rk) & (df.index == idx + i)
                    df.loc[m, "ppm"] = df.loc[m, "ppm"] + float(intensity)*(i+1)/duration
            elif mode == "O₂ drop":
                for i in range(duration):
                    m = (df["room"] == rk) & (df.index == idx + i)
                    df.loc[m, "ppm"] = df.loc[m, "ppm"] - float(intensity)*(i+1)/duration
            elif mode == "CO spike":
                for i in range(duration):
                    m = (df["room"] == rk) & (df.index == idx + i)
                    df.loc[m, "ppm"] = df.loc[m, "ppm"] + float(intensity)*0.4
            st.session_state.data = df
            st.toast(f"Simulated {mode} in {ROOM_DISPLAY[rk]}")

# ── Facility view
wireframe = st.session_state.get("dbg_wire", False)
status_by_room = {rk: worst_status_for_room(rk) for rk in ROOMS_POLY.keys()}
fig = build_facility_floorplan(status_by_room, wireframe=wireframe)
clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="facility_click_floorplan", override_height=640)
if clicked:
    payload = clicked[0] or {}
    st.session_state.last_event = payload
    rk = payload.get("customdata") if isinstance(payload.get("customdata"), str) else None
    if rk is None:
        x, y = payload.get("x"), payload.get("y")
        try:
            x=float(x); y=float(y)
            best=None; bestd=1e18
            for k, poly in ROOMS_POLY.items():
                cx, cy = _center(poly)
                d=(cx-x)**2 + (cy-y)**2
                if d<bestd: bestd=d; best=k
            rk=best
        except:
            rk=None
    if rk:
        st.session_state.room_key = rk
        st.session_state.view = "room"

# ── Room detail + chart
if st.session_state.view == "room" and st.session_state.room_key:
    rk = st.session_state.room_key
    st.markdown(f"### {ROOM_DISPLAY[rk]}")
    df = st.session_state.data[st.session_state.data['room']==rk].iloc[: st.session_state.demo_index + 1].copy()
    c1,c2 = st.columns([3,2])
    with c1:
        if df.empty:
            st.warning("No data for this room.")
        else:
            df['roll_mean'] = df['ppm'].rolling(12, min_periods=3).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['ppm'], mode='lines+markers', name='Live', line=dict(width=3)))
            if 'roll_mean' in df.columns:
                fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['roll_mean'], mode='lines', name='Rolling mean', line=dict(dash='dot')))
            # thresholds
            thrmap = {
                "reactor": (35,45,False),
                "granulation": (30,40,False),
                "cleanroom": (18,17,True),
                "packaging": (35,45,False),
                "warehouse": (35,50,False),
                "utilities": (45,55,False),
                "entry": (40,50,False),
                "gowning": (40,50,False),
                "qc": (30,40,False),
            }
            warn, danger, o2 = thrmap[rk]
            if o2:
                fig2.add_hrect(y0=-1e6, y1=danger, fillcolor=COLORS['alarm'], opacity=0.10, line_width=0)
                fig2.add_hrect(y0=danger, y1=warn, fillcolor=COLORS['inhibit'], opacity=0.08, line_width=0)
            else:
                fig2.add_hrect(y0=danger, y1=1e6, fillcolor=COLORS['alarm'], opacity=0.10, line_width=0)
                fig2.add_hrect(y0=warn, y1=danger, fillcolor=COLORS['inhibit'], opacity=0.08, line_width=0)
            fig2.update_layout(template='plotly_dark', height=420, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation='h'))
            st.plotly_chart(fig2, use_container_width=True, key=f"room_timeseries_{rk}")
    with c2:
        # simple AI metrics
        if not df.empty:
            last_val = float(df['ppm'].iloc[-1])
            slope = float(np.polyfit(np.arange(min(10,len(df))), df['ppm'].tail(10), 1)[0]) if len(df)>=2 else 0.0
            st.metric("Live Reading", f"{last_val:.1f}")
            st.metric("Slope", f"{slope:.2f} Δ/tick")
            st.caption(f"Status: {status_by_room[rk].upper()}")
    if st.button("⬅️ Back to Facility", key="back_fac"):
        st.session_state.view = "facility"

# ── Demo tick
def demo_tick():
    if st.session_state.demo_index < len(st.session_state.data)-1:
        st.session_state.demo_index += 1
demo_tick()

with st.expander("Debug"):
    st.write({"view": st.session_state.view, "room_key": st.session_state.room_key, "demo_index": st.session_state.demo_index})
    st.json(st.session_state.last_event or {})
