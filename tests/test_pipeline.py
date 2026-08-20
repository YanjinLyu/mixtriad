from examples.make_dataset import make
from mixtriad import FsqcaSpec, Pipeline, Schema


def test_end_to_end_smoke(tmp_path):
    df = make(n=160, seed=1)
    schema = Schema(outcome="engagement",
                    antecedents=["creator_reach", "account_topic_focus",
                                 "emotional_intensity", "visual_realism",
                                 "video_duration"],
                    controls=["publication_age"],
                    log_transform=["creator_reach"])
    p = Pipeline(df, schema, outdir=str(tmp_path))
    p.stage1_regression()
    assert "ols" in p.reg and p.reg["ols"].shape[0] > 3
    res = p.stage2_boosting(seeds=(42, 123), n_trials=5)
    assert set(res.metrics.columns) == {"model", "RMSE", "MAE", "R2"}
    q = p.stage3_fsqca(FsqcaSpec(directional={"creator_reach": 1}))
    assert "intermediate" in q["solutions"]
    assert (tmp_path / "fig_importance.png").exists()
