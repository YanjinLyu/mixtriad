# MixTriad report

## Stage 1 - regression
OLS R² = 0.757
| model            | term                  |   estimate |   std_error |   p_value | sig   |
|:-----------------|:----------------------|-----------:|------------:|----------:|:------|
| OLS(log outcome) | const                 |    -7.6809 |      0.6883 |    0      | ***   |
| OLS(log outcome) | log_creator_reach     |     0.77   |      0.0421 |    0      | ***   |
| OLS(log outcome) | log_creator_activity  |     0.0615 |      0.0589 |    0.2964 |       |
| OLS(log outcome) | audience_youthfulness |     0.1503 |      0.4466 |    0.7364 |       |
| OLS(log outcome) | video_duration        |    -0.0146 |      0.002  |    0      | ***   |
| OLS(log outcome) | headline_length       |    -0.0016 |      0.0024 |    0.5003 |       |
| OLS(log outcome) | hashtag_count         |    -0.0027 |      0.0239 |    0.9111 |       |
| OLS(log outcome) | emotional_intensity   |     0.3667 |      0.0422 |    0      | ***   |
| OLS(log outcome) | visual_realism        |     0.2284 |      0.0491 |    0      | ***   |
| OLS(log outcome) | av_coherence          |     0.171  |      0.0577 |    0.003  | **    |
| OLS(log outcome) | account_topic_focus   |     4.194  |      0.2475 |    0      | ***   |
| OLS(log outcome) | publication_age       |     0.008  |      0.0017 |    0      | ***   |
| OLS(log outcome) | topic_category_2      |    -0.3083 |      0.1433 |    0.0315 | *     |
| OLS(log outcome) | topic_category_3      |     0.1189 |      0.151  |    0.4311 |       |
| OLS(log outcome) | topic_category_4      |    -0.0103 |      0.1787 |    0.9542 |       |

## Stage 2 - tuned gradient boosting
| model                       |   RMSE |   MAE |    R2 |
|:----------------------------|-------:|------:|------:|
| OLS regression              |  1.13  | 0.899 | 0.701 |
| Random forest (default)     |  1.061 | 0.828 | 0.738 |
| Gradient boosting (default) |  1.012 | 0.784 | 0.761 |
| XGBoost (Bayesian-tuned)    |  0.928 | 0.741 | 0.799 |

Retained antecedents (> 0.141): creator_reach, account_topic_focus, emotional_intensity, video_duration

## Stage 3 - fsQCA (outcome)
Conditions: creator_reach, account_topic_focus, emotional_intensity, video_duration
### Intermediate solution
| recipe                                                    |   consistency |   raw_coverage |   unique_coverage |
|:----------------------------------------------------------|--------------:|---------------:|------------------:|
| creator_reach * account_topic_focus * ~video_duration     |         0.951 |          0.388 |             0.1   |
| creator_reach * account_topic_focus * emotional_intensity |         0.945 |          0.363 |             0.072 |
| creator_reach * emotional_intensity * ~video_duration     |         0.918 |          0.365 |             0.082 |
Solution consistency = 0.908, coverage = 0.530

## Stage 3 - fsQCA (negated)
Conditions: creator_reach, account_topic_focus, emotional_intensity, video_duration
### Intermediate solution
| recipe                                |   consistency |   raw_coverage |   unique_coverage |
|:--------------------------------------|--------------:|---------------:|------------------:|
| ~creator_reach * ~account_topic_focus |         0.914 |          0.601 |             0.601 |
Solution consistency = 0.914, coverage = 0.601
