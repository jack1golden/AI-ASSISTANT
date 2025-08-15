
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="OBW Safety — Floorplan", layout="wide", page_icon="🛡️")

HEALTHY="healthy"; INHIBIT="inhibit"; ALARM="alarm"; FAULT="fault"; OVER="over"
STATE_COLORS={HEALTHY:"#10b981", INHIBIT:"#f59e0b", ALARM:"#ef4444", FAULT:"#fb923c", OVER:"#e5e7eb"}

ROOMS={
    "entry":{"title":"Entry","poly":[(70,580),(300,580),(300,650),(70,650)],"detectors":[]},
    "room1":{"title":"Room 1","poly":[(820,60),(1120,60),(1120,260),(820,260)],
             "detectors":[{"key":"r1_xnx","gas":"CH₄","warn":45.0,"danger":55.0,"oxygen_mode":False}]},
    "room2":{"title":"Room 2","poly":[(820,270),(1120,270),(1120,470),(820,470)],
             "detectors":[{"key":"r2_o2","gas":"O₂","warn":18.0,"danger":17.0,"oxygen_mode":True}]},
    "room3":{"title":"Room 3","poly":[(520,270),(810,270),(810,470),(520,470)],
             "detectors":[{"key":"r3_co","gas":"CO","warn":30.0,"danger":40.0,"oxygen_mode":False}]},
    "room12":{"title":"Room 12","poly":[(520,60),(810,60),(810,260),(520,260)],
              "detectors":[{"key":"r12_nh3","gas":"NH₃","warn":35.0,"danger":45.0,"oxygen_mode":False}]},
    "prod1":{"title":"Production 1","poly":[(310,60),(510,60),(510,260),(310,260)],
             "detectors":[{"key":"p1_h2s","gas":"H₂S","warn":5.0,"danger":10.0,"oxygen_mode":False}]},
    "prod2":{"title":"Production 2","poly":[(310,270),(510,270),(510,470),(310,470)],
             "detectors":[{"key":"p2_h2","gas":"H₂","warn":35.0,"danger":50.0,"oxygen_mode":False}]},
}
DETECTOR_INDEX={d["key"]:(rk,d) for rk,r in ROOMS.items() for d in r["detectors"]}

def _init_state():
    if "data" not in st.session_state:
        try:
            st.session_state.data=pd.read_csv("demo_data.csv", parse_dates=["timestamp"])
        except Exception:
            rng=pd.date_range(end=pd.Timestamp.utcnow(), periods=600, freq="min")
            rows=[]; keys=[k for k in ROOMS.keys() if k!="entry"]
            for i,ts in enumerate(rng):
                rk=keys[i%len(keys)]; base_val=10+(i%30)*0.12
                rows.append({"timestamp":ts,"room":rk,"ppm":round(float(base_val+np.random.normal(0,0.5)),2)})
            st.session_state.data=pd.DataFrame(rows)
    st.session_state.setdefault("view","facility")
    st.session_state.setdefault("room_key",None)
    st.session_state.setdefault("demo_index", min(100, len(st.session_state.data)-1))
    st.session_state.setdefault("free_play", False)
    st.session_state.setdefault("show_route", False)
    st.session_state.setdefault("last_event", None)
    st.session_state.setdefault("wireframe", False)
_init_state()

def last_ppm(room_key:str):
    df=st.session_state.data; sub=df[df["room"]==room_key]
    if sub.empty: return None, None
    idx=min(len(sub)-1, st.session_state.demo_index)
    row=sub.iloc[idx]; return float(row["ppm"]), pd.to_datetime(row["timestamp"])

def state_from_threshold(det_cfg, value:float)->str:
    if value is None: return FAULT
    if det_cfg.get("oxygen_mode"):
        if value <= det_cfg["danger"]: return ALARM
        if value <= det_cfg["warn"]: return INHIBIT
        return HEALTHY
    else:
        if value >= det_cfg["danger"]*1.15: return OVER
        if value >= det_cfg["danger"]: return ALARM
        if value >= det_cfg["warn"]: return INHIBIT
        return HEALTHY

def worst_state_for_room(rk:str, ppm:float)->str:
    worst=HEALTHY
    for det in ROOMS[rk]["detectors"]:
        s=state_from_threshold(det, ppm)
        rank=[HEALTHY, INHIBIT, ALARM, OVER, FAULT]
        if rank.index(s)>rank.index(worst): worst=s
    if not ROOMS[rk]["detectors"]: return HEALTHY
    return worst

def build_facility_floorplan(status_by_room: dict, wireframe: bool = False):
    W,H=1200,700
    fig=go.Figure()
    fig.update_layout(template="plotly_white",
                      xaxis=dict(visible=False, range=[0,W]),
                      yaxis=dict(visible=False, range=[H,0]),
                      margin=dict(l=0,r=260,t=0,b=0),
                      height=560, paper_bgcolor="#0f172a", plot_bgcolor="#111827",
                      dragmode=False, showlegend=False)
    fig.add_shape(type="rect", x0=30,y0=30,x1=W-30,y1=H-30,
                  fillcolor="#1f2937", line=dict(color="#374151",width=2), layer="below")
    for rk,room in ROOMS.items():
        poly=room["poly"]
        xs=[p[0] for p in poly]+[poly[0][0]]
        ys=[p[1] for p in poly]+[poly[0][1]]
        fill=STATE_COLORS.get(status_by_room.get(rk, "healthy"), "#10b981")
        fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
                                 fillcolor=fill if not wireframe else "rgba(0,0,0,0)",
                                 line=dict(color="#4b5563",width=2),
                                 hovertemplate=f"{room['title']}<extra></extra>",
                                 name=rk, customdata=[rk]*len(xs), showlegend=False))
        cx=sum(p[0] for p in poly)/len(poly); cy=sum(p[1] for p in poly)/len(poly)
        fig.add_annotation(x=cx, y=cy, text=room["title"],
                           showarrow=False, font=dict(color="#e5e7eb", size=12))
    legend=[("Healthy","#10b981"),("Inhibit","#f59e0b"),("Alarm","#ef4444"),
            ("Fault","#fb923c"),("Activated / Over Range","#e5e7eb")]
    y=80
    fig.add_annotation(x=1220, y=50, text="<b>Legend</b>", showarrow=False, font=dict(color="#e5e7eb"))
    for label,color in legend:
        fig.add_shape(type="rect", x0=1220, y0=y, x1=1250, y1=y+20, fillcolor=color, line=dict(color="#334155"))
        fig.add_annotation(x=1260, y=y+10, text=label, showarrow=False, font=dict(color="#e5e7eb",size=12), xanchor="left", yanchor="middle")
        y+=30
    return fig

# top bar
col_logo, col_title, col_dev = st.columns([1,3,2])
with col_logo: st.write("🛡️")
with col_title: st.markdown("<h2 style='text-align:center;margin:6px 0'>OBW Floorplan — Next‑Gen</h2>", unsafe_allow_html=True)
with col_dev:
    st.session_state.free_play = st.toggle("Dev", key="dev_toggle", value=st.session_state.get("free_play", False))
    st.session_state.wireframe = st.checkbox("Wireframe", key="wireframe_cb", value=st.session_state.get("wireframe", False))

# quick buttons
qcols = st.columns(len(ROOMS))
for i,(rk,room) in enumerate(ROOMS.items()):
    with qcols[i]:
        if st.button(room["title"], key=f"roombtn_{rk}"):
            st.session_state.room_key=rk; st.session_state.view="room"

# sidebar sim
if st.session_state.free_play:
    with st.sidebar:
        st.markdown("### Simulation Center")
        sim_room = st.selectbox("Room", [k for k in ROOMS.keys() if k!="entry"], key="sim_room")
        sim_mode = st.selectbox("Mode", ["Spike","Ramp","O₂ drop","CO spike"], key="sim_mode")
        sim_intensity = st.slider("Intensity", 5, 120, 45, step=5, key="sim_intensity")
        sim_duration = st.slider("Duration (ticks)", 3, 40, 10, key="sim_duration")
        if st.button("Run Simulation", key="sim_run", use_container_width=True):
            df = st.session_state.data.copy()
            idx = st.session_state.demo_index
            if sim_mode=="Spike":
                mask=(df["room"]==sim_room)&(df.index>=idx)&(df.index<idx+sim_duration)
                df.loc[mask,"ppm"]=df.loc[mask,"ppm"]+float(sim_intensity)
            elif sim_mode=="Ramp":
                for i in range(sim_duration):
                    mask=(df["room"]==sim_room)&(df.index==idx+i)
                    df.loc[mask,"ppm"]=df.loc[mask,"ppm"]+float(sim_intensity)*(i+1)/sim_duration
            elif sim_mode=="O₂ drop":
                for i in range(sim_duration):
                    mask=(df["room"]==sim_room)&(df.index==idx+i)
                    df.loc[mask,"ppm"]=df.loc[mask,"ppm"]-float(sim_intensity)*(i+1)/sim_duration
            elif sim_mode=="CO spike":
                for i in range(sim_duration):
                    mask=(df["room"]==sim_room)&(df.index==idx+i)
                    df.loc[mask,"ppm"]=df.loc[mask,"ppm"]+float(sim_intensity)*0.4
            st.session_state.data=df; st.toast(f"Simulated {sim_mode} in {ROOMS[sim_room]['title']}")

def last_ppm(room_key:str):
    df=st.session_state.data; sub=df[df["room"]==room_key]
    if sub.empty: return None, None
    idx=min(len(sub)-1, st.session_state.demo_index)
    row=sub.iloc[idx]; return float(row["ppm"]), pd.to_datetime(row["timestamp"])

def compute_status_map():
    m={}
    for rk in ROOMS.keys():
        ppm,_=last_ppm(rk)
        m[rk]=worst_state_for_room(rk, ppm if ppm is not None else None)
    return m

def render_facility():
    status_by_room = compute_status_map()
    fig=build_facility_floorplan(status_by_room, wireframe=st.session_state.wireframe)
    clicked=plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                          key="facility_click_floorplan", override_height=560)
    if clicked:
        payload=clicked[0]
        rk = payload.get("customdata")
        if isinstance(rk, list): rk=rk[0]
        if rk not in ROOMS:
            x,y=payload.get("x"), payload.get("y")
            best=None; bestd=1e18
            for k,room in ROOMS.items():
                cx=sum(p[0] for p in room["poly"])/len(room["poly"])
                cy=sum(p[1] for p in room["poly"])/len(room["poly"])
                d=(cx-x)**2+(cy-y)**2
                if d<bestd: bestd=d; best=k
            rk=best
        st.session_state.room_key=rk; st.session_state.view="room"

def render_room():
    rk=st.session_state.room_key
    if rk is None or rk not in ROOMS:
        st.session_state.view="facility"; return
    room=ROOMS[rk]
    st.markdown(f"### {room['title']}")
    c1,c2=st.columns([2,1])
    with c1:
        df=st.session_state.data[st.session_state.data['room']==rk].iloc[:st.session_state.demo_index+1]
        if df.empty:
            st.info("No data for this room yet.")
        else:
            roll=df["ppm"].rolling(12,min_periods=3).mean()
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"],y=df["ppm"],mode="lines+markers",name="Live"))
            fig.add_trace(go.Scatter(x=df["timestamp"],y=roll,mode="lines",name="Rolling mean", line=dict(dash="dot")))
            if room["detectors"]:
                det=room["detectors"][0]; warn, danger = det["warn"], det["danger"]
                if det.get("oxygen_mode"):
                    fig.add_hrect(y0=-1e6,y1=danger, fillcolor=STATE_COLORS[ALARM], opacity=0.1, line_width=0)
                    fig.add_hrect(y0=danger,y1=warn, fillcolor=STATE_COLORS[INHIBIT], opacity=0.08, line_width=0)
                else:
                    fig.add_hrect(y0=danger,y1=1e6, fillcolor=STATE_COLORS[ALARM], opacity=0.1, line_width=0)
                    fig.add_hrect(y0=warn,y1=danger, fillcolor=STATE_COLORS[INHIBIT], opacity=0.08, line_width=0)
            fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True, key=f"room_timeseries_{rk}")
    with c2:
        ppm,_=last_ppm(rk)
        st.metric("Live Reading", f"{ppm:.1f} ppm" if ppm is not None else "—")
        status = compute_status_map().get(rk, HEALTHY)
        st.metric("Status", status.upper())
        if st.button("⬅️ Back to Facility", key=f"btn_back_{rk}", use_container_width=True):
            st.session_state.view="facility"

if st.button("▶ Next", key="tick_next"): st.session_state.demo_index = min(st.session_state.demo_index+1, len(st.session_state.data)-1)
if st.button("⏭ Reset", key="tick_reset"): st.session_state.demo_index = 0

if st.session_state.view=="facility": render_facility()
else: render_room()
