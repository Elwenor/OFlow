from oflow.benchmark import autotune, equivalent


def test_equivalence_checks_schema_booleans_labels_and_nonfinite():
    assert equivalent({"x": [1.0, "a"]}, {"x": [1.00001, "a"]})
    assert not equivalent({"x": 1}, {"y": 1})
    assert not equivalent(True, 1)
    assert not equivalent(float("nan"), float("nan"))
    assert not equivalent(float("inf"), float("inf"))
    assert not equivalent(["keep"], ["drop"])
    assert not equivalent(20000000, 20000001)


def test_invalid_candidate_never_wins_including_one_bad_repetition():
    calls = 0

    def flaky():
        nonlocal calls
        calls += 1
        return {"answer": calls != 2}

    report = autotune(
        lambda: {"answer": True},
        {
            "valid": lambda: {"answer": True},
            "invalid": lambda: {"answer": False},
            "flaky": flaky,
        },
        environment={"hardware": "test", "backend": "synthetic", "model_version": "none"},
    )
    assert report["winner"] == "valid"
    assert not report["candidates"]["flaky"]["correct"]
    assert len({tuple(x) for x in report["orders"]}) > 1
    assert len(report["candidates"]["valid"]["seconds"]) == 3


def test_no_equivalent_winner_is_reported_honestly():
    report = autotune(lambda: 1, {"invalid": lambda: 9}, environment={"hardware": "test"})
    assert report["winner"] is None
