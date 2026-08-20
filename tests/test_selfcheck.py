from mixtriad.selfcheck import run_selfcheck


def test_selfcheck_quick_passes():
    rep = run_selfcheck(full=False, verbose=False)
    assert rep["passed"] is True
    assert len(rep["checks"]) == 4
    assert rep["checks"][0]["cases"] == 255 and rep["checks"][0]["failures"] == 0


def test_selfcheck_cli_exit_code():
    from mixtriad.cli import main
    assert main(["selfcheck"]) == 0


def test_selfcheck_fails_on_broken_qm(monkeypatch):
    import mixtriad.selfcheck as SC
    monkeypatch.setattr(SC, "quine_mccluskey", lambda pos, dc=(): [])
    rep = SC.run_selfcheck(verbose=False)
    assert rep["passed"] is False
    assert rep["checks"][0]["failures"] > 0


def test_selfcheck_fails_on_broken_calibration(monkeypatch):
    import pandas as pd
    import mixtriad.selfcheck as SC
    monkeypatch.setattr(SC.fsqca, "direct_calibrate",
                        lambda x, a, b, c: pd.Series([0.7] * len(x), index=x.index))
    rep = SC.run_selfcheck(verbose=False)
    assert rep["passed"] is False
    assert rep["checks"][1]["failures"] > 0


def test_selfcheck_fails_on_broken_regression(monkeypatch):
    import mixtriad.selfcheck as SC
    monkeypatch.setattr(SC, "run_regressions", lambda df, schema: {})
    rep = SC.run_selfcheck(verbose=False)
    assert rep["passed"] is False
    assert rep["checks"][3]["failures"] > 0


def test_selfcheck_full_fails_on_broken_boosting(monkeypatch):
    import pandas as pd
    import mixtriad.boosting as B
    import mixtriad.selfcheck as SC

    class _Fake:
        metrics = pd.DataFrame({"model": ["x"]})
    monkeypatch.setattr(B, "run_boosting", lambda *a, **k: _Fake())
    rep = SC.run_selfcheck(full=True, verbose=False)
    assert rep["passed"] is False
