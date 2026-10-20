split test: 128 common problems (114 contests); numbers in %

| run | pass@1 | pass@8 | cov@8 | select@8 | gap@8 | Chk rate | turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | 42.48 | 87.20 | 64.84 | 58.20 | 6.65 | 38.8 | 7.04 |
| sspo | 49.95 | 88.22 | 69.19 | 64.28 | 4.91 | 46.6 | 6.88 |
| drgrpo | 50.10 | 87.81 | 66.02 | 59.34 | 6.69 | 44.6 | 5.83 |
| pkpo | 45.61 | 86.79 | 65.51 | 59.02 | 6.49 | 41.3 | 7.15 |

paired select@8: sspo minus other (points, 95% contest-cluster bootstrap, Holm)

| comparison | diff | 95% CI | p | p_holm |
|---|---:|---|---:|---:|
| sspo - frozen | +6.08 | [+0.65, +11.63] | 0.029 | 0.058 |
| sspo - drgrpo | +4.94 | [+0.26, +9.78] | 0.040 | 0.058 |
| sspo - pkpo | +5.26 | [+0.92, +9.85] | 0.016 | 0.048 |

paired pass@1: sspo minus other (points, 95% contest-cluster bootstrap, Holm)

| comparison | diff | 95% CI | p | p_holm |
|---|---:|---|---:|---:|
| sspo - frozen | +7.47 | [+2.91, +12.22] | 0.001 | 0.003 |
| sspo - drgrpo | -0.15 | [-4.45, +4.12] | 0.955 | 0.955 |
| sspo - pkpo | +4.35 | [+0.19, +8.56] | 0.042 | 0.083 |
