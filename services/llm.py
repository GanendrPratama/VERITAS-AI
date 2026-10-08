"""Ollama client for the Analyst and Interviewer agents.

Latency choices from the design doc (Section 7): keep_alive keeps the model
resident in VRAM between the two agents' calls instead of reloading it each
turn; num_predict caps generation length since agent JSON output is short;
history is trimmed to the last N turns so prompt length stays flat across a
long interview instead of growing every turn.
"""
import json

import requests


class InvalidAgentJSON(Exception):
    pass


def trim_history(turns, context_turns):
    """Keep only the most recent `context_turns` turns. 0/None = no trimming."""
    if not context_turns:
        return turns
    return turns[-context_turns:]


def _generate(host, model, prompt, keep_alive, max_tokens):
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"num_predict": max_tokens},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def call_agent(host, model, prompt, validate=None, keep_alive="30m", max_tokens=256):
    """Call an agent and parse+validate its JSON. One retry on failure (Section 4);
    `validate` (if given) raises ValueError on a badly-shaped response, treated
    the same as a JSON parse failure -- one retry, then InvalidAgentJSON."""

    def _try(p):
        raw = _generate(host, model, p, keep_alive, max_tokens)
        data = json.loads(raw)  # may raise json.JSONDecodeError
        if validate:
            validate(data)  # may raise ValueError
        return data

    try:
        return _try(prompt)
    except (json.JSONDecodeError, ValueError):
        try:
            return _try(prompt + "\n\nReturn only valid JSON matching the schema above.")
        except (json.JSONDecodeError, ValueError) as e:
            raise InvalidAgentJSON(str(e))


def render_transcript(turns):
    """turns: [{"question": str, "answer": str}, ...]."""
    if not turns:
        return "(no turns yet)"
    return "\n".join(f"{i + 1}. Q: {t['question']}\n   A: {t['answer']}" for i, t in enumerate(turns))


def render_ledger(ledger):
    """ledger: {claim_id: {"text": str, "state": str, ...}}."""
    if not ledger:
        return "(no claims yet)"
    return "\n".join(f"{cid}: {c['text']} [{c['state']}]" for cid, c in ledger.items())


def build_analyst_prompt(template, report, claim_text, transcript_turns, answer, arousal):
    claim = claim_text if claim_text is not None else "(none -- identify a new claim)"
    return (
        template.replace("<<REPORT>>", report)
        .replace("<<CLAIM>>", claim)
        .replace("<<TRANSCRIPT>>", render_transcript(transcript_turns))
        .replace("<<ANSWER>>", answer)
        .replace("<<AROUSAL>>", arousal)
    )


def build_interviewer_prompt(template, report, ledger, transcript_turns):
    return (
        template.replace("<<REPORT>>", report)
        .replace("<<LEDGER>>", render_ledger(ledger))
        .replace("<<TRANSCRIPT>>", render_transcript(transcript_turns))
    )


def validate_analyst_response(data):
    if data.get("state") not in ("consistent", "contradicted", "evasive"):
        raise ValueError(f"bad state: {data.get('state')!r}")
    if data.get("plausibility") not in ("adequate", "vague", "non_answer", "refusal"):
        raise ValueError(f"bad plausibility: {data.get('plausibility')!r}")
    if "reasoning" not in data:
        raise ValueError("missing reasoning")


def validate_claims_response(data):
    claims = data.get("claims")
    if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
        raise ValueError("claims must be a list of strings")


def validate_interviewer_response(data, valid_claim_ids):
    if data.get("type") not in ("anomaly-probe", "deepen", "new-claim"):
        raise ValueError(f"bad type: {data.get('type')!r}")
    if not data.get("question"):
        raise ValueError("missing question")
    claim_id = data.get("claim_id")
    if data["type"] == "new-claim":
        if claim_id is not None:
            raise ValueError("new-claim must have claim_id=null")
    elif claim_id not in valid_claim_ids:
        raise ValueError(f"claim_id {claim_id!r} not in ledger")


if __name__ == "__main__":
    assert trim_history(list(range(10)), 6) == [4, 5, 6, 7, 8, 9]
    assert trim_history([1, 2], 6) == [1, 2]
    assert trim_history([1, 2, 3], 0) == [1, 2, 3]
    assert trim_history([1, 2, 3], None) == [1, 2, 3]

    assert render_transcript([]) == "(no turns yet)"
    assert render_transcript([{"question": "Q1?", "answer": "A1"}]) == "1. Q: Q1?\n   A: A1"

    assert render_ledger({}) == "(no claims yet)"
    assert render_ledger({"c1": {"text": "was at X", "state": "unverified"}}) == "c1: was at X [unverified]"

    prompt = build_analyst_prompt("R=<<REPORT>> C=<<CLAIM>> T=<<TRANSCRIPT>> A=<<ANSWER>> AR=<<AROUSAL>>",
                                   "report text", "claim text", [], "answer text", "low")
    assert prompt == "R=report text C=claim text T=(no turns yet) A=answer text AR=low", prompt

    prompt = build_analyst_prompt("C=<<CLAIM>>", "r", None, [], "a", "low")
    assert prompt == "C=(none -- identify a new claim)", prompt

    validate_claims_response({"claims": ["a", "b"]})
    try:
        validate_claims_response({"claims": "a"})
        assert False, "should have raised"
    except ValueError:
        pass

    validate_analyst_response({"state": "consistent", "plausibility": "adequate", "reasoning": "ok"})
    try:
        validate_analyst_response({"state": "maybe", "plausibility": "adequate", "reasoning": "ok"})
        assert False, "should have raised"
    except ValueError:
        pass

    validate_interviewer_response({"type": "deepen", "claim_id": "c1", "question": "?"}, {"c1"})
    validate_interviewer_response({"type": "new-claim", "claim_id": None, "question": "?"}, {"c1"})
    try:
        validate_interviewer_response({"type": "deepen", "claim_id": "nope", "question": "?"}, {"c1"})
        assert False, "should have raised"
    except ValueError:
        pass

    print("llm.py self-check passed")
