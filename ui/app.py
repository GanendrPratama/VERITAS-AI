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

st.set_page_config(page_title="Lie-Detector Operator Console", layout="wide")


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
    st.title("Lie-Detector -- Session")
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


state = get("/state")
if state and state.get("session_id") and not st.session_state.get("force_landing"):
    control_panel(state)
else:
    st.session_state["force_landing"] = False
    landing_screen()
