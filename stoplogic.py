"""The four stop conditions (design doc Section 3), OR'd together with one
guard on condition 2. Pure function, kept separate from orchestrator.py for
testing -- never evaluated on agent prose, only on the ledger/confidence
fields orchestrator.py already trusts.
"""


def _all_claims_probed(ledger):
    if not ledger:
        return False
    return all(c["state"] in ("consistent", "contradicted") for c in ledger.values())


def _every_claim_probed_once(ledger):
    return bool(ledger) and all(len(c["turns"]) >= 1 for c in ledger.values())


def should_stop(ledger, confidence, confidence_history, question_count, elapsed_minutes, operator_override, config):
    """Returns (should_stop: bool, reason: str | None)."""
    if operator_override:
        return True, "operator_override"

    if question_count >= config["max_questions"] or elapsed_minutes >= config["max_minutes"]:
        return True, "hard_cap"

    if _all_claims_probed(ledger):
        return True, "all_claims_probed"

    stable_turns = config["stable_turns"]
    if _every_claim_probed_once(ledger) and len(confidence_history) >= stable_turns:
        recent = confidence_history[-stable_turns:]
        if all(abs(c) >= config["high_bound"] for c in recent):
            return True, "confidence_threshold"

    return False, None


def _selfcheck():
    cfg = {"max_questions": 5, "max_minutes": 30, "stable_turns": 2, "high_bound": 0.35}

    empty = {}
    assert should_stop(empty, 0.0, [], 0, 0, False, cfg) == (False, None)

    assert should_stop(empty, 0.0, [], 0, 0, True, cfg) == (True, "operator_override")
    assert should_stop(empty, 0.0, [], 5, 0, False, cfg) == (True, "hard_cap")
    assert should_stop(empty, 0.0, [], 0, 30, False, cfg) == (True, "hard_cap")

    all_resolved = {
        "c1": {"state": "consistent", "turns": [{"arousal": "low"}]},
        "c2": {"state": "contradicted", "turns": [{"arousal": "high"}]},
    }
    assert should_stop(all_resolved, 0.0, [], 0, 0, False, cfg) == (True, "all_claims_probed")

    # confidence threshold: guard blocks it until every claim has been probed at least once
    unprobed = {"c1": {"state": "unverified", "turns": []}}
    assert should_stop(unprobed, 0.9, [0.9, 0.9], 1, 1, False, cfg) == (False, None)

    probed_once = {"c1": {"state": "unverified", "turns": [{"arousal": "low"}]}}
    assert should_stop(probed_once, 0.9, [0.9, 0.9], 1, 1, False, cfg) == (True, "confidence_threshold")
    # not stable yet (only one prior turn at threshold)
    assert should_stop(probed_once, 0.9, [0.1, 0.9], 1, 1, False, cfg) == (False, None)

    print("stoplogic.py self-check passed")


if __name__ == "__main__":
    _selfcheck()
