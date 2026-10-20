split test: 128 common problems (114 contests); numbers in %

| run | pass@1 | pass@8 | cov@8 | select@8 | gap@8 | Chk rate | turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | 42.48 | 87.20 | 64.84 | 58.20 | 6.65 | 38.8 | 7.04 |
| sspo | 44.24 | 85.49 | 66.83 | 60.34 | 6.49 | 41.8 | 6.83 |
| drgrpo | 45.65 | 85.30 | 65.61 | 57.99 | 7.62 | 43.5 | 5.93 |
| pkpo | 44.92 | 86.63 | 65.64 | 59.76 | 5.88 | 41.9 | 6.92 |

paired select@8: sspo minus other (points, 95% contest-cluster bootstrap, Holm)

| comparison | diff | 95% CI | p | p_holm |
|---|---:|---|---:|---:|
| sspo - frozen | +2.15 | [-2.14, +6.34] | 0.328 | 0.934 |
| sspo - drgrpo | +2.35 | [-2.26, +6.77] | 0.311 | 0.934 |
| sspo - pkpo | +0.58 | [-3.38, +4.41] | 0.777 | 0.934 |

paired pass@1: sspo minus other (points, 95% contest-cluster bootstrap, Holm)

| comparison | diff | 95% CI | p | p_holm |
|---|---:|---|---:|---:|
| sspo - frozen | +1.76 | [-1.85, +5.45] | 0.361 | 1.000 |
| sspo - drgrpo | -1.42 | [-5.34, +2.45] | 0.470 | 1.000 |
| sspo - pkpo | -0.68 | [-4.12, +2.77] | 0.700 | 1.000 |
