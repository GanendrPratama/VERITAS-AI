"""Ollama client for the Analyst and Interviewer agents.

Latency choices from the design doc (Section 7): keep_alive keeps the model
resident in VRAM between the two agents' calls instead of reloading it each
turn; num_predict caps generation length since agent JSON output is short;
history is trimmed to the last N turns so prompt length stays flat across a
long interview instead of growing every turn.

Further latency/robustness choices:
- Every prompt starts with the same REPORT block (report_prefix). Ollama reuses
  the KV cache for the longest prefix shared with the previous request, so the
  report -- by far the longest part of every prompt -- is evaluated once per
  session instead of on every Analyst and Interviewer call.
- num_ctx is set explicitly (session_num_ctx) and kept constant: Ollama's
  default context (2048-4096 tokens) silently cut long reports, and a changed
  num_ctx makes Ollama reload the model.
- `format` is a JSON schema (Ollama structured outputs) where the server
  supports it, so a wrong enum value or a claim_id outside LEDGER/FOCUS can't
  be generated at all -- no retry round-trip for it.
- Responses are streamed so a speculative call (the prefetched next question)
  can be cancelled; closing the stream stops Ollama generating.
"""
import json
import threading
import time

import requests


class InvalidAgentJSON(Exception):
    pass


class Cancelled(Exception):
    pass


_schema_supported = True  # flipped off once if this Ollama rejects a JSON-schema `format`


def trim_history(turns, context_turns):
    """Keep only the most recent `context_turns` turns. 0/None = no trimming."""
    if not context_turns:
        return turns
    return turns[-context_turns:]


def report_prefix(report):
    """The block every agent prompt starts with -- identical across calls so Ollama's prompt cache hits."""
    return f"REPORT (the subject's written report):\n{report}\n\n---\n\n"


def session_num_ctx(report, max_tokens=768, floor=4096, ceiling=16384):
    """Context size for a session: room for the report (~3 chars/token, conservative for
    Indonesian), instructions, LEDGER, TRANSCRIPT and the answer, rounded up to 2048.
    Fixed per report so it never changes between calls (a change reloads the model)."""
    need = len(report) // 3 + 2048 + max_tokens
    return max(floor, min(ceiling, -(-need // 2048) * 2048))


def _stats(final, started):
    ns = 1e6
    return {
        "total_ms": round((time.monotonic() - started) * 1000),
        "load_ms": round(final.get("load_duration", 0) / ns),
        "prompt_tokens": final.get("prompt_eval_count", 0),  # tokens evaluated, i.e. NOT from the cache
        "prompt_ms": round(final.get("prompt_eval_duration", 0) / ns),
        "output_tokens": final.get("eval_count", 0),
        "output_ms": round(final.get("eval_duration", 0) / ns),
    }


def _generate(host, model, prompt, keep_alive, max_tokens, schema=None, num_ctx=None, cancel=None, stats=None):
    global _schema_supported
    options = {"num_predict": max_tokens}
    if num_ctx:
        options["num_ctx"] = num_ctx
    use_schema = schema is not None and _schema_supported
    started = time.monotonic()
    with requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": schema if use_schema else "json",
            "stream": True,
            "keep_alive": keep_alive,
            "options": options,
        },
        stream=True,
        timeout=60,  # per read: no token for 60s = stuck
    ) as resp:
        if use_schema and resp.status_code in (400, 500) and "format" in resp.text.lower():
            _schema_supported = False  # Ollama < 0.5: plain JSON mode, the validator still checks the shape
            return _generate(host, model, prompt, keep_alive, max_tokens, None, num_ctx, cancel, stats)
        resp.raise_for_status()
        parts, final = [], {}
        for line in resp.iter_lines(chunk_size=None):  # each chunk as it arrives, so cancel is prompt
            if cancel is not None and cancel.is_set():
                raise Cancelled()  # leaving the `with` closes the stream; Ollama stops generating
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise requests.RequestException(chunk["error"])
            parts.append(chunk.get("response", ""))
            if chunk.get("done"):
                final = chunk
                break
    if stats is not None:
        stats.update(_stats(final, started))
    return "".join(parts)


def warm(host, model, prompt, keep_alive="30m", num_ctx=None):
    """Load the model and evaluate `prompt` (the report prefix) into Ollama's prompt cache,
    so the first real call of a session doesn't pay for either. Best-effort."""
    try:
        options = {"num_predict": 1}
        if num_ctx:
            options["num_ctx"] = num_ctx
        requests.post(f"{host}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive, "options": options,
        }, timeout=180).raise_for_status()
        return True
    except Exception as e:
        print(f"[llm] warm-up failed: {type(e).__name__}: {e}")
        return False


def warm_async(*args, **kwargs):
    threading.Thread(target=warm, args=args, kwargs=kwargs, daemon=True).start()


def call_agent(host, model, prompt, validate=None, keep_alive="30m", max_tokens=256,
               schema=None, num_ctx=None, cancel=None, stats=None):
    """Call an agent and parse+validate its JSON. One retry on failure (Section 4);
    `validate` (if given) raises ValueError on a badly-shaped response, treated
    the same as a JSON parse failure -- one retry, then InvalidAgentJSON.
    `schema`: JSON schema for Ollama structured outputs; `cancel`: threading.Event that
    aborts the call (raises Cancelled); `stats`: dict filled with Ollama's timings."""

    def _try(p):
        raw = _generate(host, model, p, keep_alive, max_tokens, schema, num_ctx, cancel, stats)
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


def build_analyst_prompt(template, report, claim_text, transcript_turns, answer, arousal, question=None):
    claim = claim_text if claim_text is not None else "(none -- identify a new claim)"
    return report_prefix(report) + (
        template.replace("<<QUESTION>>", question or "(not recorded)")
        .replace("<<CLAIM>>", claim)
        .replace("<<TRANSCRIPT>>", render_transcript(transcript_turns))
        .replace("<<ANSWER>>", answer)
        .replace("<<AROUSAL>>", arousal)
    )


def render_focus(ledger, focus_claim_id):
    if focus_claim_id is None:
        return "(none -- pick the claim yourself)"
    claim = ledger[focus_claim_id]
    return f"{focus_claim_id}: {claim['text']} [{claim['state']}]"


def build_claims_prompt(template, report):
    return report_prefix(report) + template


def build_interviewer_prompt(template, report, ledger, transcript_turns, focus_claim_id=None):
    return report_prefix(report) + (
        template.replace("<<FOCUS>>", render_focus(ledger, focus_claim_id))
        .replace("<<LEDGER>>", render_ledger(ledger))
        .replace("<<TRANSCRIPT>>", render_transcript(transcript_turns))
    )


ANALYST_STATES = ("consistent", "contradicted", "evasive")
PLAUSIBILITY = ("adequate", "vague", "non_answer", "refusal")

ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"enum": list(ANALYST_STATES)},
        "plausibility": {"enum": list(PLAUSIBILITY)},
        "reasoning": {"type": "string"},
        "new_claim_text": {"type": ["string", "null"]},
    },
    "required": ["state", "plausibility", "reasoning", "new_claim_text"],
}

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}, "maxItems": 12}},
    "required": ["claims"],
}


def interviewer_schema(valid_claim_ids, focus_claim_id=None):
    """With a FOCUS claim the model can only answer with that claim's id -- the claim picker
    used to fail when qwen2.5 wrote a question about a different claim."""
    if focus_claim_id is not None:
        types, ids = ["anomaly-probe", "deepen"], [focus_claim_id]
    else:
        types, ids = ["anomaly-probe", "deepen", "new-claim"], [*sorted(valid_claim_ids), None]
    return {
        "type": "object",
        "properties": {
            "type": {"enum": types},
            "claim_id": {"enum": ids},
            "question": {"type": "string"},
        },
        "required": ["type", "claim_id", "question"],
    }


def validate_analyst_response(data):
    if data.get("state") not in ANALYST_STATES:
        raise ValueError(f"bad state: {data.get('state')!r}")
    if data.get("plausibility") not in PLAUSIBILITY:
        raise ValueError(f"bad plausibility: {data.get('plausibility')!r}")
    if "reasoning" not in data:
        raise ValueError("missing reasoning")


def validate_claims_response(data):
    claims = data.get("claims")
    if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
        raise ValueError("claims must be a list of strings")


def validate_interviewer_response(data, valid_claim_ids, focus_claim_id=None, focus_state=None):
    """`focus_claim_id`: the claim the operator chose to probe; the question must be about it."""
    if data.get("type") not in ("anomaly-probe", "deepen", "new-claim"):
        raise ValueError(f"bad type: {data.get('type')!r}")
    if not data.get("question"):
        raise ValueError("missing question")
    claim_id = data.get("claim_id")
    if focus_claim_id is not None:
        # A null id is the same model habit as below; the prompt named the claim, so tie it back.
        # A different id means the model ignored FOCUS -- retry.
        if claim_id is not None and claim_id != focus_claim_id:
            raise ValueError(f"claim_id {claim_id!r} is not the chosen claim {focus_claim_id!r}")
        data["claim_id"] = focus_claim_id
        if data["type"] == "new-claim":
            data["type"] = "anomaly-probe" if focus_state in ("contradicted", "evasive") else "deepen"
        return
    if claim_id is None:
        # qwen2.5 often answers {"type": "deepen", "claim_id": null} (~half the time on a live
        # run); rejecting that killed the interview. A question tied to no claim is a new-claim.
        data["type"] = "new-claim"
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

    P = report_prefix("report text")
    prompt = build_analyst_prompt("C=<<CLAIM>> T=<<TRANSCRIPT>> A=<<ANSWER>> AR=<<AROUSAL>>",
                                   "report text", "claim text", [], "answer text", "low")
    assert prompt == P + "C=claim text T=(no turns yet) A=answer text AR=low", prompt

    p = report_prefix("r")
    prompt = build_analyst_prompt("C=<<CLAIM>>", "r", None, [], "a", "low")
    assert prompt == p + "C=(none -- identify a new claim)", prompt

    assert build_analyst_prompt("Q=<<QUESTION>>", "r", None, [], "a", "low", question="Di mana?") == p + "Q=Di mana?"
    assert build_analyst_prompt("Q=<<QUESTION>>", "r", None, [], "a", "low") == p + "Q=(not recorded)"

    # every agent's prompt starts with the same bytes, so Ollama's prompt cache covers the report
    ledger1 = {"c1": {"text": "a", "state": "unverified"}}
    for built in (build_claims_prompt("X", "r"), build_interviewer_prompt("F=<<FOCUS>>", "r", ledger1, [])):
        assert built.startswith(p), built

    assert session_num_ctx("") == 4096
    assert session_num_ctx("x" * 30000) == 14336
    assert session_num_ctx("x" * 10 ** 6) == 16384

    s1 = interviewer_schema({"c2", "c1"}, focus_claim_id="c2")
    assert s1["properties"]["claim_id"]["enum"] == ["c2"] and "new-claim" not in s1["properties"]["type"]["enum"]
    assert interviewer_schema({"c2", "c1"})["properties"]["claim_id"]["enum"] == ["c1", "c2", None]

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
    loose = {"type": "deepen", "claim_id": None, "question": "?"}
    validate_interviewer_response(loose, {"c1"})
    assert loose["type"] == "new-claim", loose
    try:
        validate_interviewer_response({"type": "deepen", "claim_id": "nope", "question": "?"}, {"c1"})
        assert False, "should have raised"
    except ValueError:
        pass

    focused = {"type": "new-claim", "claim_id": None, "question": "?"}
    validate_interviewer_response(focused, {"c1", "c2"}, focus_claim_id="c2", focus_state="evasive")
    assert focused["claim_id"] == "c2" and focused["type"] == "anomaly-probe", focused
    try:
        validate_interviewer_response({"type": "deepen", "claim_id": "c1", "question": "?"}, {"c1", "c2"},
                                      focus_claim_id="c2")
        assert False, "should have raised"
    except ValueError:
        pass

    ledger = {"c1": {"text": "a", "state": "unverified"}, "c2": {"text": "b", "state": "evasive"}}
    assert build_interviewer_prompt("F=<<FOCUS>>", "r", ledger, []) == p + "F=(none -- pick the claim yourself)"
    assert build_interviewer_prompt("F=<<FOCUS>>", "r", ledger, [], "c2") == p + "F=c2: b [evasive]"

    print("llm.py self-check passed")
