# Human evaluation results

Submitted V4 values. Scores: 1–5; higher is better. SD is across image means; CI is the stratified prompt-bootstrap 95% interval.

| Language | Dimension | Method | Prompts | Mean ± SD | 95% CI |
|---|---|---|---:|---:|---|
| en | attribute | sdxl | 36 | 3.954 ± 1.096 | [3.602, 4.278] |
| en | attribute | rpg | 36 | 4.278 ± 0.882 | [4.028, 4.528] |
| en | attribute | kolors | 36 | 3.861 ± 1.040 | [3.583, 4.120] |
| en | attribute | rpg_kolors | 36 | 4.120 ± 1.096 | [3.815, 4.407] |
| en | attribute | ragd | 36 | 4.593 ± 0.722 | [4.370, 4.796] |
| en | attribute | bireg | 36 | 4.269 ± 1.026 | [3.981, 4.519] |
| en | spatial | sdxl | 34 | 3.402 ± 1.219 | [3.049, 3.765] |
| en | spatial | rpg | 34 | 3.520 ± 1.274 | [3.137, 3.902] |
| en | spatial | kolors | 34 | 3.402 ± 1.315 | [3.020, 3.804] |
| en | spatial | rpg_kolors | 34 | 3.431 ± 1.332 | [3.029, 3.853] |
| en | spatial | ragd | 34 | 4.324 ± 1.000 | [4.000, 4.608] |
| en | spatial | bireg | 34 | 3.676 ± 1.389 | [3.265, 4.088] |
| en | quality | sdxl | 60 | 3.944 ± 0.326 | [3.861, 4.022] |
| en | quality | rpg | 60 | 4.017 ± 0.256 | [3.950, 4.072] |
| en | quality | kolors | 60 | 4.011 ± 0.086 | [4.000, 4.033] |
| en | quality | rpg_kolors | 60 | 4.033 ± 0.243 | [3.978, 4.094] |
| en | quality | ragd | 60 | 4.033 ± 0.227 | [3.978, 4.089] |
| en | quality | bireg | 60 | 4.044 ± 0.264 | [3.983, 4.106] |
| zh | attribute | sdxl | 27 | 3.469 ± 1.005 | [3.111, 3.827] |
| zh | attribute | rpg | 27 | 3.531 ± 1.099 | [3.136, 3.938] |
| zh | attribute | kolors | 27 | 4.049 ± 0.714 | [3.790, 4.309] |
| zh | attribute | rpg_kolors | 27 | 3.753 ± 1.065 | [3.370, 4.136] |
| zh | attribute | ragd | 27 | 3.951 ± 0.866 | [3.630, 4.272] |
| zh | attribute | bireg | 27 | 4.136 ± 0.622 | [3.901, 4.370] |
| zh | spatial | sdxl | 27 | 3.222 ± 1.202 | [2.790, 3.667] |
| zh | spatial | rpg | 27 | 3.358 ± 1.261 | [2.889, 3.827] |
| zh | spatial | kolors | 27 | 4.235 ± 0.973 | [3.852, 4.580] |
| zh | spatial | rpg_kolors | 27 | 4.037 ± 1.130 | [3.605, 4.432] |
| zh | spatial | ragd | 27 | 4.222 ± 0.925 | [3.877, 4.556] |
| zh | spatial | bireg | 27 | 4.259 ± 1.059 | [3.852, 4.630] |
| zh | quality | sdxl | 29 | 4.000 ± 0.267 | [3.897, 4.103] |
| zh | quality | rpg | 29 | 3.977 ± 0.333 | [3.862, 4.092] |
| zh | quality | kolors | 29 | 3.977 ± 0.198 | [3.897, 4.034] |
| zh | quality | rpg_kolors | 29 | 4.057 ± 0.346 | [3.931, 4.172] |
| zh | quality | ragd | 29 | 3.989 ± 0.227 | [3.897, 4.069] |
| zh | quality | bireg | 29 | 4.115 ± 0.312 | [4.000, 4.218] |

## Inter-rater agreement

| Language | Dimension | Ordinal Krippendorff alpha | Images | Applicable ratings |
|---|---|---:|---:|---:|
| en | attribute | 0.842011 | 216 | 648 |
| en | spatial | 0.869049 | 204 | 612 |
| en | quality | 0.797289 | 360 | 1080 |
| zh | attribute | 0.719559 | 162 | 486 |
| zh | spatial | 0.878509 | 162 | 486 |
| zh | quality | 0.725812 | 174 | 522 |

## Paired differences: BiReG minus baseline

| Language | Dimension | Baseline | Mean difference | 95% CI |
|---|---|---|---:|---|
| en | attribute | sdxl | 0.315 | [-0.019, 0.676] |
| en | attribute | rpg | -0.009 | [-0.278, 0.259] |
| en | attribute | kolors | 0.407 | [0.102, 0.713] |
| en | attribute | rpg_kolors | 0.148 | [-0.074, 0.370] |
| en | attribute | ragd | -0.324 | [-0.639, -0.037] |
| en | spatial | sdxl | 0.275 | [-0.167, 0.716] |
| en | spatial | rpg | 0.157 | [-0.186, 0.549] |
| en | spatial | kolors | 0.275 | [-0.010, 0.569] |
| en | spatial | rpg_kolors | 0.245 | [-0.069, 0.588] |
| en | spatial | ragd | -0.647 | [-1.108, -0.186] |
| en | quality | sdxl | 0.100 | [-0.006, 0.217] |
| en | quality | rpg | 0.028 | [-0.061, 0.122] |
| en | quality | kolors | 0.033 | [-0.022, 0.089] |
| en | quality | rpg_kolors | 0.011 | [-0.072, 0.089] |
| en | quality | ragd | 0.011 | [-0.072, 0.100] |
| zh | attribute | sdxl | 0.667 | [0.284, 1.037] |
| zh | attribute | rpg | 0.605 | [0.185, 1.012] |
| zh | attribute | kolors | 0.086 | [-0.148, 0.296] |
| zh | attribute | rpg_kolors | 0.383 | [0.099, 0.704] |
| zh | attribute | ragd | 0.185 | [-0.099, 0.469] |
| zh | spatial | sdxl | 1.037 | [0.593, 1.494] |
| zh | spatial | rpg | 0.901 | [0.321, 1.444] |
| zh | spatial | kolors | 0.025 | [-0.296, 0.284] |
| zh | spatial | rpg_kolors | 0.222 | [-0.111, 0.593] |
| zh | spatial | ragd | 0.037 | [-0.407, 0.457] |
| zh | quality | sdxl | 0.115 | [0.011, 0.218] |
| zh | quality | rpg | 0.138 | [0.011, 0.264] |
| zh | quality | kolors | 0.138 | [0.057, 0.230] |
| zh | quality | rpg_kolors | 0.057 | [-0.057, 0.172] |
| zh | quality | ragd | 0.126 | [0.046, 0.218] |
