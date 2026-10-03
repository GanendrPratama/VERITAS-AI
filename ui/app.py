"""Operator dashboard -- Streamlit.

Talks to orchestrator.py only over its local HTTP API (see orchestrator.py
docstring for the endpoint list). No direct import of orchestrator code in
either direction -- this process and the orchestrator can be started,
stopped, and restarted independently.
"""
import json

import requests
import streamlit as st

ORCH_URL = "http://localhost:8000"

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


def post(path, data=None):
    try:
        resp = requests.post(f"{ORCH_URL}{path}", data=data, timeout=5)
        if resp.status_code >= 400:
            st.error(resp.json().get("error", "request failed"))
            return None
        return resp.json()
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


def landing_screen():
    st.markdown('<div class="kicker">Stage 01 / Input</div>', unsafe_allow_html=True)
    st.title("Applicant Document")
    tab_new, tab_open = st.tabs(["New Session", "Open Old Session"])

    with tab_new:
        pdf = st.file_uploader("Report (PDF)", type="pdf")
        if pdf is not None and st.button("Create Session"):
            if post("/sessions", data=pdf.getvalue()) is not None:
                st.rerun()

    with tab_open:
        sessions = get("/sessions") or []
        if not sessions:
            st.write("No unfinished sessions.")
        else:
            options = {f"{s['created_at']} -- {s['label'] or s['session_id']}": s["session_id"] for s in sessions}
            choice = st.selectbox("Session", list(options))
            if st.button("Open"):
                if post(f"/sessions/{options[choice]}/open") is not None:
                    st.rerun()


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

    text = st.text_area("Add claims (one per line)")
    if st.button("Save Claims"):
        lines = [line for line in text.splitlines() if line.strip()]
        if lines and post("/claims", data=json.dumps(lines)) is not None:
            st.rerun()

    if state["ledger"] and st.button("Start Interview", type="primary"):
        post("/calibrate/start")
        st.rerun()

    if st.button("Switch Session"):
        st.session_state["force_landing"] = True
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
            st.session_state["force_landing"] = True
            st.rerun()
        if st.button("Finish Calibration"):
            post("/calibrate/stop")
            st.rerun()
        if st.button("Record"):
            post("/record/start")
            st.rerun()
        if st.button("Stop"):
            post("/record/stop")
            st.rerun()
        if st.button("Generate Question"):
            post("/generate-question")
            st.rerun()
        if st.button("Stop Interview", type="primary"):
            post("/stop-interview")
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
        st.write("**Flagged claims:**", state["flagged_claims"] or "none")
        st.write("**Cleared claims:**", state["cleared_claims"] or "none")
        st.write("**Ledger:**")
        st.json(state["ledger"])

        with st.expander("Report text"):
            st.text(state["report_text"])

    with col_state:
        live_state()


def step_of(state):
    if not state or not state.get("session_id") or st.session_state.get("force_landing"):
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


def sidebar(step):
    with st.sidebar:
        st.markdown('<div class="brand"><b>V</b><span>VERITAS-AI<small>Operator Console</small></span></div>',
                    unsafe_allow_html=True)
        st.markdown("".join(
            f'<div class="step {"on" if i == step else "done" if i < step else ""}"><i>{"✓" if i < step else i + 1}</i>{n}</div>'
            for i, n in enumerate(STEPS)), unsafe_allow_html=True)
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
        st.session_state["force_landing"] = True
        st.rerun()


state = get("/state")
step = step_of(state)
sidebar(step)
alert_banner()
if step == 3:
    results_step(state)
elif state and state.get("session_id") and not st.session_state.get("force_landing"):
    control_panel(state)
else:
    st.session_state["force_landing"] = False
    landing_screen()
log_viewer()
