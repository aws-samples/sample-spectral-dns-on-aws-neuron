# References

Retained runs the checks compare against (columns `step,t,E,Omega,eps_enstrophy,eps_dEdt,step_wall_ms`; `E` kinetic
energy, `Omega` enstrophy, `eps_enstrophy = 2 nu Omega`, `eps_dEdt = -dE/dt`). `step_wall_ms` is the wall time of the
machine that produced the file and is not compared.

| file | what | used by |
|---|---|---|
| `oracle_64.csv` | fp64 NumPy oracle (`tools/oracle.py`), 64^3, 204 steps to t = 10.01, fp32 wavenumber constants | 64^3 rows, tolerance 1e-6 |
| `oracle_128.csv` | same oracle at 128^3, 408 steps | 128^3 single-core row, 1e-6 |
| `oracle128_f64consts.csv` | 128^3 oracle with float64 wavenumber constants (the constants the multi-core driver builds) | documentation of the 6.8e-6 gap between the two paths |
| `tgv_dist2_128_32.csv` | 32-rank Trainium1 (trn1.32xlarge) run, 128^3 | 128^3 multi-core rows, 4e-9 |
| `tgv_dist2_256_32.csv` | 32-rank Trainium1 run, 256^3 | 256^3 rows, 4e-9 |
| `tgv_dist2_512_32.csv` | 32-rank Trainium1 run, 512^3 | 512^3 rows, 2e-7 |

The two single-core references and the multi-core references differ from each other by up to 6.8e-6 in `E` and
1.6e-5 in `eps` at 128^3: the single-core path keeps its wavenumber constants in fp32 from the start, the multi-core
driver builds them in fp64 per rank and rounds once. Each path is checked against the reference built with its own
constants; the 4e-9 agreement figure belongs to the multi-core path.
