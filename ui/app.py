"""Operator dashboard -- Streamlit.

Talks to orchestrator.py only over its local HTTP API (see orchestrator.py
docstring for the endpoint list). No direct import of orchestrator code in
either direction -- this process and the orchestrator can be started,
stopped, and restarted independently.
"""
import json

import requests
import streamlit as st

ORCH_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="VERITAS-AI | Operator Console", layout="wide")

# Palette and look follow VERITAS-AI_Dashboard_SlideFlow.html.
st.markdown("""<style>
:root{--border:#1e3855;--muted:#91a4ba;--accent:#54d7c7;--ok:#69e09a;--warn:#f2c66d;--bad:#ff7081}
.stApp{background:radial-gradient(circle at 15% 10%,rgba(84,215,199,.08),transparent 28%),radial-gradient(circle at 85% 5%,rgba(104,169,255,.1),transparent 26%),#07111f;color:#eaf2fb}
[data-testid=stSidebar]{background:rgba(6,16,29,.9);border-right:1px solid var(--border)}
[data-testid=stHeader]{background:transparent}
.brand{display:flex;gap:12px;align-items:center;padding-bottom:16px;border-bottom:1px solid var(--border);margin-bottom:14px}
.brand b{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;background:linear-gradient(135deg,#54d7c7,#68a9ff);color:#06121f;font-size:20px}
.brand span{font-size:18px;font-weight:700}.brand small{display:block;font-size:11px;color:var(--muted);font-weight:400}
.mini{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin:14px 0 8px}
.step{display:flex;gap:10px;align-items:center;padding:9px 10px;border-radius:12px;border:1px solid transparent;color:var(--muted);font-size:13px;margin-bottom:6px}
.step i{width:28px;height:28px;border-radius:9px;display:grid;place-items:center;background:#122941;border:1px solid var(--border);font-style:normal;font-size:12px;font-weight:800}
.step.on{background:#102b42;border-color:#25516d;color:#eaf2fb}.step.on i{color:var(--accent);border-color:rgba(84,215,199,.4)}
.step.done{color:#eaf2fb}.step.done i{color:var(--ok);border-color:rgba(105,224,154,.3)}
.comp{display:flex;justify-content:space-between;gap:8px;padding:8px 10px;margin-bottom:6px;border:1px solid var(--border);border-radius:11px;background:#0b1a2c;font-size:12px}
.comp small{display:block;color:var(--muted);font-size:10px;margin-top:2px}
.comp.ok{border-left:3px solid var(--ok)}.comp.idle{border-left:3px solid var(--muted)}.comp.down{border-left:3px solid var(--bad);background:rgba(255,112,129,.08)}
.comp em{font-style:normal;font-weight:800;font-size:10px;letter-spacing:.06em}
.comp.ok em{color:var(--ok)}.comp.idle em{color:var(--muted)}.comp.down em{color:var(--bad)}
.alert{padding:11px 14px;border-radius:12px;border:1px solid rgba(255,112,129,.4);background:rgba(255,112,129,.08);font-size:13px;margin-bottom:14px}
.alert b{color:var(--bad)}
.kicker{font-size:11px;color:var(--accent);letter-spacing:.11em;text-transform:uppercase;font-weight:800}
h1,h2,h3{letter-spacing:0}
.stButton>button{border-radius:12px;font-weight:700;border:1px solid var(--border);background:#17314d;color:#eaf2fb}
.stButton>button[kind=primary]{background:linear-gradient(135deg,#54d7c7,#67efd9);color:#06121f;border:none}
[data-testid=stMetric]{background:#0b1a2c;border:1px solid var(--border);border-radius:12px;padding:10px 12px}
</style>""", unsafe_allow_html=True)

STEPS = ["Document", "Claims", "Interview", "Results"]
SLOW_TIMEOUTS = {"/generate-question": 180}  # up to two 60s Interviewer tries, with margin


def post(path, data=None, timeout=120):  # slow actions (LLM, STT, camera restart) share this; refused connections still fail instantly
    try:
        resp = requests.post(f"{ORCH_URL}{path}", data=data, timeout=timeout)
        if resp.status_code >= 400:
            st.error(resp.json().get("error", "request failed"))
            return None
        return resp.json()
    except requests.exceptions.Timeout:
        st.error(f"The orchestrator is still working after {timeout}s -- check its log, then try again.")
        return None
    except requests.exceptions.RequestException:
        st.error("Can't reach the orchestrator -- is `python orchestrator.py` running?")
        return None


def get(path):
    try:
        resp = requests.get(f"{ORCH_URL}{path}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def text_area(label, name):
    """A text area the app can fill or clear. Assigning st.session_state[key] loses to the
    stale value the browser sends with the next auto-refresh (the 1-3s fragments below), so
    a new value is shown through a fresh widget key instead."""
    rev = st.session_state.get(f"{name}_rev", 0)
    return st.text_area(label, value=st.session_state.get(f"{name}_value", ""), key=f"{name}_{rev}")


def set_text_area(name, value):
    st.session_state[f"{name}_value"] = value
    st.session_state[f"{name}_rev"] = st.session_state.get(f"{name}_rev", 0) + 1


def landing_screen(state):
    st.markdown('<div class="kicker">Stage 01 / Input</div>', unsafe_allow_html=True)
    st.title("Applicant Document")
    tab_new, tab_open = st.tabs(["New Session", "Open Old Session"])

    with tab_new:
        pdf = st.file_uploader("Report (PDF)", type="pdf")
        if pdf is not None and st.button("Create Session"):
            if post("/sessions", data=pdf.getvalue()) is not None:
                st.rerun()

    if (state or {}).get("phase", "idle") in ("idle", "stopped"):  # camera can't be switched mid-interview
        with st.expander("Test webcam and microphone"):
            camera_panel()
            device_panel()

    with tab_open:
        sessions = get("/sessions") or []
        if not sessions:
            st.write("No unfinished sessions.")
        else:
            options = {f"{s['created_at']} -- {s['label'] or s['session_id']}": s["session_id"] for s in sessions}
            choice = st.selectbox("Session", list(options))
            open_col, del_col = st.columns(2)
            if open_col.button("Open"):
                if post(f"/sessions/{options[choice]}/open") is not None:
                    st.rerun()
            if del_col.button("Delete"):
                st.session_state["confirm_delete"] = options[choice]
            if st.session_state.get("confirm_delete") == options[choice]:
                st.warning("Permanently delete this session and its logs?")
                if st.button("Yes, delete", type="primary"):
                    st.session_state.pop("confirm_delete")
                    try:
                        requests.delete(f"{ORCH_URL}/sessions/{options[choice]}", timeout=5).raise_for_status()
                    except requests.exceptions.RequestException as e:
                        st.error(f"delete failed: {e}")
                    else:
                        st.rerun()


@st.fragment(run_every="1s")
def camera_preview():
    try:
        resp = requests.get(f"{ORCH_URL}/camera/frame", timeout=2)
    except requests.exceptions.RequestException:
        resp = None
    if resp is not None and resp.status_code == 200:
        st.image(resp.content, use_container_width=True)
    else:
        st.caption("No frame yet -- camera still opening, or it failed (see Component Status).")


def camera_panel():
    """Pick a camera and test it. Choosing one opens it immediately, so it's
    already running by the time calibration starts."""
    cams = get("/cameras") or {"cameras": [], "selected": None}
    if not cams["cameras"]:
        st.warning("No camera detected.")
        return
    sel = cams["selected"] if cams["selected"] in cams["cameras"] else cams["cameras"][0]
    notes = cams.get("notes") or {}
    pick = st.selectbox("Camera", cams["cameras"], index=cams["cameras"].index(sel),
                        format_func=lambda i: f"Camera {i}" + (f" ({notes[str(i)]})" if notes.get(str(i)) else ""))
    if st.session_state.get("cam_started") != pick:
        st.session_state["cam_started"] = pick
        post("/camera", data=str(pick))
    if st.toggle("Show live preview"):
        camera_preview()


def device_panel():
    """Microphone and compute-device pickers (applied immediately, for this run only)."""
    dev = get("/devices")
    if dev is None:
        return
    mics = {m["index"]: m["name"] for m in dev["mics"]}
    if mics:
        ids = list(mics)
        cur = dev["mic_selected"] if dev["mic_selected"] in mics else ids[0]
        pick = st.selectbox("Microphone", ids, index=ids.index(cur), format_func=mics.get,
                            key="mic_pick", help="First entry is used if the system default isn't listed")
        if pick != dev["mic_selected"] and st.session_state.get("mic_applied") != pick:
            st.session_state["mic_applied"] = pick
            post("/devices", data=json.dumps({"mic": pick}))
    else:
        st.warning("No microphone detected.")
    c1, c2 = st.columns(2)
    for col, key, label in [(c1, "stt_device", "Speech-to-text runs on"), (c2, "vision_device", "Face model runs on")]:
        opts = ["cuda", "cpu"]
        choice = col.selectbox(label, opts, index=opts.index(dev[key]) if dev[key] in opts else 0, key=f"dev_{key}")
        if choice != dev[key]:
            post("/devices", data=json.dumps({key: choice}))


def claims_step(state):
    st.markdown('<div class="kicker">Stage 02 / Context</div>', unsafe_allow_html=True)
    st.title("Claims")
    st.write("Read the report, then list the claims to track (one per line).")
    with st.expander("Report text"):
        st.text(state["report_text"])

    if state["ledger"]:
        st.write("**Claims so far:**")
        for cid, c in state["ledger"].items():
            st.write(f"- `{cid}`: {c['text']}")

    if state["phase"] != "idle":
        st.info(f"Interview is {state['phase']} -- claims can no longer be edited.")
        return

    if st.button("Auto-extract claims from report"):
        with st.spinner("Asking the LLM..."):
            found = post("/claims/extract", timeout=180)
        if found and found["claims"]:
            set_text_area("claims_text", "\n".join(found["claims"]))
            st.rerun()
        elif found is not None:
            st.info("No new claims found.")
    text = text_area("Add claims (one per line) -- review and edit before saving", "claims_text")
    if st.button("Save Claims"):
        lines = [line for line in text.splitlines() if line.strip()]
        if lines and post("/claims", data=json.dumps(lines)) is not None:
            set_text_area("claims_text", "")
            st.rerun()

    camera_panel()
    device_panel()

    if state["ledger"] and st.button("Start Interview", type="primary"):
        post("/calibrate/start")
        st.rerun()

    if st.button("Switch Session"):
        st.session_state["view"] = (0, view_key(state))
        st.rerun()


def control_panel(state):
    if state["phase"] == "idle":
        claims_step(state)
        return

    st.markdown('<div class="kicker">Stage 03 / Live Session</div>', unsafe_allow_html=True)
    col_controls, col_state = st.columns([1, 2])

    with col_controls:
        st.subheader("Controls")
        if st.button("Switch Session"):
            st.session_state["view"] = (0, view_key(state))
            st.rerun()
        for label, path, kind in [
            ("Finish Calibration", "/calibrate/stop", "secondary"),
            ("Record", "/record/start", "secondary"),
            ("Generate Question", "/generate-question", "secondary"),
            ("Stop Interview", "/stop-interview", "primary"),
        ]:
            if label == "Generate Question":
                typed = text_area("Typed answer (only if mic/STT unavailable)", "typed_answer")
                # STT (CPU fallback on a long answer) + up to two 60s Analyst tries
                if st.button("Stop", key="stop_rec") and post("/record/stop", data=typed.encode("utf-8"), timeout=300) is not None:
                    set_text_area("typed_answer", "")
                    st.rerun()
            # Rerun only on success -- a rerun would wipe the error post() just showed.
            if st.button(label, type=kind) and post(path, timeout=SLOW_TIMEOUTS.get(path, 120)) is not None:
                st.rerun()

    @st.fragment(run_every="1s")
    def live_state():
        state = get("/state")
        if not state or not state.get("session_id"):
            st.warning("Orchestrator unreachable -- dashboard will keep retrying.")
            return

        st.metric("Phase", state["phase"])
        st.metric("Recording", "yes" if state["recording"] else "no")
        st.metric("Confidence", f"{state['confidence']:+.2f}")
        if state["stop_recommended"]:
            st.warning("Stop recommended")

        st.write("**Current question:**", state["current_question"] or "—")
        if state["recording"]:
            caption = ((get("/live") or {}).get("stt") or {}).get("partial")
            st.info(f"🎙 {caption}" if caption else "🎙 Listening…")
        st.write("**Flagged claims:**", state["flagged_claims"] or "none")
        st.write("**Cleared claims:**", state["cleared_claims"] or "none")
        st.write("**Ledger:**")
        st.json(state["ledger"])

        with st.expander("Report text"):
            st.text(state["report_text"])

    with col_state:
        live_state()


def view_key(state):
    return (state or {}).get("session_id"), (state or {}).get("phase")


def step_of(state):
    has_session = bool(state and state.get("session_id"))
    view = st.session_state.get("view")
    # A manual view only holds while the session and its phase are unchanged; a phase change
    # (e.g. Start Interview) or a new/opened session hands control back to the automatic step.
    if view and view[1] == view_key(state) and (has_session or view[0] == 0):
        return view[0]
    if not has_session:
        return 0
    return {"idle": 1, "calibration": 2, "interview": 2}.get(state["phase"], 3)


@st.fragment(run_every="3s")
def health_panel():
    health = get("/health")
    if health is None:
        st.markdown('<div class="comp down"><div>Orchestrator<small>not reachable on :8000</small></div><em>DOWN</em></div>',
                    unsafe_allow_html=True)
        return
    for h in health:
        st.markdown(f'<div class="comp {h["status"]}"><div>{h["name"]}<small>{h["detail"]}</small></div>'
                    f'<em>{h["status"].upper()}</em></div>', unsafe_allow_html=True)


def sidebar(step, state):
    has_session = bool(state and state.get("session_id"))
    with st.sidebar:
        st.markdown('<div class="brand"><b>V</b><span>VERITAS-AI<small>Operator Console</small></span></div>',
                    unsafe_allow_html=True)
        for i, name in enumerate(STEPS):
            label = f"{'✓' if i < step else i + 1}  {name}"
            if st.button(label, key=f"nav{i}", disabled=i > 0 and not has_session,
                         type="primary" if i == step else "secondary", use_container_width=True):
                st.session_state["view"] = (i, view_key(state))
                st.rerun()
        st.markdown('<div class="mini">Component Status</div>', unsafe_allow_html=True)
        health_panel()


@st.fragment(run_every="3s")
def alert_banner():
    bad = [h["name"] for h in get("/health") or [] if h["status"] == "down"]
    if bad:
        st.markdown(f'<div class="alert"><b>Not detected:</b> {", ".join(bad)} — see Component Status in the sidebar.</div>',
                    unsafe_allow_html=True)


@st.fragment(run_every="2s")
def log_body(which):
    data = get(f"/logs/{which}")
    st.code(data["text"] if data else "Orchestrator unreachable.", language="log", height=300)


@st.fragment(run_every="1s")
def live_body():
    live = get("/live")
    if live is None:
        st.warning("Orchestrator unreachable.")
        return
    cam, mic, sttc, sens = st.columns(4)
    with cam:
        st.caption("Webcam")
        camera_preview()
        v = live["vision"]
        st.caption(f"face model: {v['device'] or 'loading / off'}" + (f" -- {v['error']}" if v["error"] else ""))
        for ch, d in v["channels"].items():
            st.metric(ch, f"{d['value']:.2f}", help=f"{d['age']}s ago")
    with mic:
        st.caption("Microphone" + (" -- RECORDING" if live["mic"]["recording"] else ""))
        st.progress(live["mic"]["level"])
        st.caption(live["mic"]["error"] or live["mic"]["device"] or "opening...")
    with sttc:
        t = live["stt"]
        st.caption("Speech-to-text")
        st.write(f"**{t['state']}**" + (f" ({t['device']})" if t["device"] else ""))
        if t["partial"]:
            st.write(f"🎙 {t['partial']}")
        elif t["last_text"]:
            st.write(f"“{t['last_text']}”")
        if t["error"]:
            st.caption(t["error"])
    with sens:
        st.caption("ESP32 sensor")
        if not live["sensors"]:
            st.write("no data")
        for ch, d in live["sensors"].items():
            st.metric(ch, f"{d['value']:.1f}", help=f"{d['age']}s ago")


def live_monitor():
    if st.toggle("Live monitor (webcam, mic, speech-to-text, sensors)", key="live_on"):
        live_body()


def log_viewer():
    with st.expander("Logs (live)"):
        for tab, which in zip(st.tabs(["Orchestrator", "Dashboard"]), ["orchestrator", "dashboard"]):
            with tab:
                log_body(which)


def results_step(state):
    st.markdown('<div class="kicker">Stage 04 / Output</div>', unsafe_allow_html=True)
    st.title("Interview Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{state['confidence']:+.2f}")
    c2.metric("Flagged claims", len(state["flagged_claims"]))
    c3.metric("Cleared claims", len(state["cleared_claims"]))
    st.write("**Ledger:**")
    st.json(state["ledger"])
    with st.expander("Report text"):
        st.text(state["report_text"])
    if st.button("New Session", type="primary"):
        st.session_state["view"] = (0, view_key(state))
        st.rerun()


state = get("/state")
step = step_of(state)
sidebar(step, state)
alert_banner()
if step == 0:
    landing_screen(state)
elif step == 1:
    claims_step(state)
elif step == 2:
    control_panel(state)
else:
    results_step(state)
live_monitor()
log_viewer()
