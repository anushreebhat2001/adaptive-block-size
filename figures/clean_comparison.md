# Clean comparison report

_baseline = `adablock`_


### dream / gsm8k

_n = 200 prompts (intersection across schedulers)_

| scheduler | acc | Δacc vs adablock | t/s | Δt/s vs adablock |
|---|---|---|---|---|
| **fixed-32** | 0.000 | — | 19.1 | — |
| **ours-oracle** | 0.000 | — | 17.5 | — |

### llada / gsm8k

_n = 200 prompts (intersection across schedulers)_

| scheduler | acc | Δacc vs adablock | t/s | Δt/s vs adablock |
|---|---|---|---|---|
| **adablock** | 0.760 | +0.000 | 14.5 | +0.0 |
| **fixed-16** | 0.770 | +0.010 | 13.3 | -1.1 |
| **fixed-32** | 0.755 | -0.005 | 14.6 | +0.1 |
| **fixed-4** | 0.735 | -0.025 | 9.5 | -5.0 |
| **fixed-8** | 0.745 | -0.015 | 12.0 | -2.5 |
| **ours-oracle** | 0.740 | -0.020 | 13.5 | -1.0 |
| **ours-teacher** | 0.740 | -0.020 | 14.6 | +0.2 |

### llada / humaneval

_n = 164 prompts (intersection across schedulers)_

| scheduler | acc | Δacc vs adablock | t/s | Δt/s vs adablock |
|---|---|---|---|---|
| **adablock** | 0.866 | +0.000 | 18.2 | +0.0 |
| **fixed-16** | 0.890 | +0.024 | 18.1 | -0.1 |
| **fixed-32** | 0.896 | +0.030 | 20.1 | +1.8 |
| **fixed-4** | 0.841 | -0.024 | 12.7 | -5.5 |
| **fixed-8** | 0.866 | +0.000 | 15.9 | -2.4 |
| **ours-oracle** | 0.890 | +0.024 | 18.1 | -0.2 |
| **ours-teacher** | 0.921 | +0.055 | 19.6 | +1.4 |

### llada / math

_n = 200 prompts (intersection across schedulers)_

| scheduler | acc | Δacc vs adablock | t/s | Δt/s vs adablock |
|---|---|---|---|---|
| **adablock** | 0.260 | +0.000 | 19.3 | +0.0 |
| **fixed-16** | 0.305 | +0.045 | 19.7 | +0.4 |
| **fixed-32** | 0.265 | +0.005 | 21.4 | +2.1 |
| **fixed-4** | 0.260 | +0.000 | 13.6 | -5.7 |
| **fixed-8** | 0.295 | +0.035 | 17.1 | -2.2 |
| **ours-oracle** | 0.275 | +0.015 | 19.4 | +0.1 |
| **ours-teacher** | 0.275 | +0.015 | 20.7 | +1.4 |