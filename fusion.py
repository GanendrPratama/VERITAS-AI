"""The fusion rule (design doc Section 2). Kept separate from orchestrator.py
so it's a pure function over the ledger -- easy to test, no I/O.

Ledger shape (see orchestrator.py):
    {claim_id: {"text": str, "state": "unverified"|"consistent"|"contradicted"|"evasive",
                "turns": [{"arousal": "low"|"elevated"|"high"|"unavailable"}, ...]}}
"""


def bucket_arousal(deltas, config):
    """deltas: {channel: sd_above_baseline or None}. Max across available
    channels decides the bucket -- one channel spiking is enough."""
    values = [v for v in deltas.values() if v is not None]
    if not values:
        return "unavailable"
    peak = max(values)
    if peak >= config["arousal_high_sd"]:
        return "high"
    if peak >= config["arousal_elevated_sd"]:
        return "elevated"
    return "low"


def _is_flagged(claim):
    if claim["state"] == "contradicted":
        return True
    if claim["state"] == "evasive":
        return any(t["arousal"] == "high" for t in claim["turns"])
    return False


def _is_cleared(claim):
    return claim["state"] == "consistent" and not any(t["arousal"] == "high" for t in claim["turns"])


def update_confidence(ledger, config):
    """Returns (confidence, flagged_claim_ids, cleared_claim_ids)."""
    flagged = [cid for cid, c in ledger.items() if _is_flagged(c)]
    cleared = [cid for cid, c in ledger.items() if _is_cleared(c)]
    confidence = -config["w_flag"] * len(flagged) + config["w_clear"] * len(cleared)
    confidence = max(-1.0, min(1.0, confidence))
    return confidence, flagged, cleared


def _selfcheck():
    cfg = {"arousal_elevated_sd": 1.0, "arousal_high_sd": 2.0, "w_flag": 0.20, "w_clear": 0.10}

    assert bucket_arousal({}, cfg) == "unavailable"
    assert bucket_arousal({"gsr": None, "hr": None}, cfg) == "unavailable"
    assert bucket_arousal({"gsr": 0.5}, cfg) == "low"
    assert bucket_arousal({"gsr": 1.2, "hr": None}, cfg) == "elevated"
    assert bucket_arousal({"gsr": 0.1, "hr": 2.5}, cfg) == "high"  # max across channels wins

    ledger = {
        "c1": {"text": "was at X", "state": "contradicted", "turns": [{"arousal": "low"}]},
        "c2": {"text": "did Y", "state": "evasive", "turns": [{"arousal": "high"}]},
        "c3": {"text": "did Y2", "state": "evasive", "turns": [{"arousal": "low"}]},
        "c4": {"text": "said Z", "state": "consistent", "turns": [{"arousal": "low"}]},
        "c5": {"text": "said Z2", "state": "consistent", "turns": [{"arousal": "high"}]},
        "c6": {"text": "unchecked", "state": "unverified", "turns": []},
    }
    confidence, flagged, cleared = update_confidence(ledger, cfg)
    assert set(flagged) == {"c1", "c2"}, flagged  # contradicted, evasive+high
    assert set(cleared) == {"c4"}, cleared  # consistent+high (c5) doesn't clear
    assert abs(confidence - (-0.20 * 2 + 0.10 * 1)) < 1e-9, confidence

    # clamp
    many_flags = {f"f{i}": {"state": "contradicted", "turns": []} for i in range(10)}
    confidence, _, _ = update_confidence(many_flags, cfg)
    assert confidence == -1.0

    print("fusion.py self-check passed")


if __name__ == "__main__":
    _selfcheck()
