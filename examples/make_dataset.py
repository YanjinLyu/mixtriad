"""Generate the synthetic demonstration dataset shipped with MixTriad.

The schema mirrors a cross-platform corpus of AI-generated misinformation
short videos (11 antecedents on four information-ecology dimensions plus a
publication-age control and an engagement outcome).  Values are simulated -
no real platform data are redistributed - but the generative process encodes
the qualitative structure reported in the accompanying study: reach and
topic focus dominate, technology quality matters non-linearly, and virality
is heavy-tailed.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)
N = 388


def make(n: int = N, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    platform = rng.choice(["tiktok", "shorts"], size=n, p=[0.55, 0.45])

    creator_reach = np.round(np.exp(rng.normal(9.2, 1.8, n))).astype(int)      # followers
    creator_activity = np.round(np.exp(rng.normal(4.5, 1.0, n))).astype(int)   # videos posted
    audience_youthfulness = np.clip(
        rng.normal(np.where(platform == "tiktok", 0.72, 0.55), 0.08, n), 0.2, 0.95)

    topic_category = rng.choice([1, 2, 3, 4], size=n, p=[0.28, 0.27, 0.25, 0.20])
    video_duration = np.clip(rng.gamma(3.0, 18.0, n), 8, 180)                  # seconds
    headline_length = np.clip(rng.normal(52, 22, n), 6, 140).astype(int)       # characters
    hashtag_count = rng.poisson(4.2, n)
    emotional_intensity = np.clip(rng.normal(4.0, 1.4, n), 1, 7)               # 1-7 coder scale
    visual_realism = np.clip(rng.normal(4.3, 1.3, n), 1, 7)
    av_coherence = np.clip(
        0.55 * visual_realism + rng.normal(1.9, 0.9, n), 1, 7)
    account_topic_focus = np.clip(rng.beta(2.2, 2.8, n), 0.02, 0.98)           # share of niche videos
    publication_age = rng.integers(3, 120, n)                                  # days online

    log_reach = np.log1p(creator_reach)
    z = (
        0.45 * log_reach
        + 0.90 * np.maximum(log_reach - 10.0, 0)            # saturating-then-steep reach effect
        + 0.90 * account_topic_focus
        + 1.60 * (account_topic_focus > 0.55)               # algorithmic-niche threshold
        + 0.10 * emotional_intensity
        + 0.055 * emotional_intensity * visual_realism      # emotion x realism interaction
        + 0.16 * av_coherence
        - 0.020 * np.maximum(video_duration - 45.0, 0)      # attention decay beyond ~45 s
        - 0.004 * headline_length
        + 0.04 * hashtag_count
        + 0.20 * (topic_category == 3)
        + 0.45 * audience_youthfulness
        + 0.04 * np.log1p(creator_activity)
        + 0.010 * publication_age
        # configurational kicker: reach x focus x emotion (the fsQCA target)
        + 0.90 * ((log_reach > 9.5)
                  & (account_topic_focus > 0.55)
                  & (emotional_intensity > 4.5)).astype(float)
        + rng.normal(0, 0.75, n)
    )
    engagement = np.round(np.exp(z - 4.0)).astype(int)                          # heavy-tailed count

    return pd.DataFrame({
        "video_id": [f"V{idx:04d}" for idx in range(1, n + 1)],
        "platform": platform,
        "creator_reach": creator_reach,
        "creator_activity": creator_activity,
        "audience_youthfulness": np.round(audience_youthfulness, 3),
        "topic_category": topic_category,
        "video_duration": np.round(video_duration, 1),
        "headline_length": headline_length,
        "hashtag_count": hashtag_count,
        "emotional_intensity": np.round(emotional_intensity, 2),
        "visual_realism": np.round(visual_realism, 2),
        "av_coherence": np.round(av_coherence, 2),
        "account_topic_focus": np.round(account_topic_focus, 3),
        "publication_age": publication_age,
        "engagement": engagement,
    })


if __name__ == "__main__":
    df = make()
    df.to_csv("examples/misinfo_videos_synthetic.csv", index=False)
    print(f"wrote examples/misinfo_videos_synthetic.csv  ({len(df)} rows)")
