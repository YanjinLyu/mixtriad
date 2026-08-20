# MixTriad report

## Stage 1 - regression
OLS R² = 0.067
| model            | term            |   estimate |   std_error |   p_value | sig   |
|:-----------------|:----------------|-----------:|------------:|----------:|:------|
| OLS(log outcome) | const           |     3.5866 |      0.1979 |    0      | ***   |
| OLS(log outcome) | headline_length |    -0.0256 |      0.0093 |    0.0057 | **    |
| OLS(log outcome) | word_count      |     0.0762 |      0.0547 |    0.1636 |       |
| OLS(log outcome) | exclamations    |    -0.1534 |      0.2464 |    0.5335 |       |
| OLS(log outcome) | caps_ratio      |     0.2447 |      0.512  |    0.6327 |       |
| OLS(log outcome) | clickbait_cues  |     0.2211 |      0.2497 |    0.376  |       |
| OLS(log outcome) | is_fake         |     1.3737 |      0.1754 |    0      | ***   |

## Stage 2 - tuned gradient boosting
| model                       |   RMSE |   MAE |     R2 |
|:----------------------------|-------:|------:|-------:|
| Random forest (default)     |  2.979 | 2.402 | -0.176 |
| Gradient boosting (default) |  2.848 | 2.369 | -0.075 |
| XGBoost (Bayesian-tuned)    |  2.726 | 2.305 |  0.015 |
| OLS regression              |  2.705 | 2.295 |  0.03  |

Retained antecedents (> -0.009): is_fake, headline_length, word_count, exclamations, clickbait_cues

## Stage 3 - fsQCA (outcome)
Conditions: is_fake, headline_length, word_count, exclamations, clickbait_cues
### Intermediate solution

Solution consistency = 0.000, coverage = 0.000
