This module contains the instances used for tuning (in `instances/`) and the script to solve instances (`tune.py`).

We ran two different tuning experiments: one for the maximum number of customers to destroy $D$, and the record-to-record travel initial threshold $H$.
$D$ can be adjusted by setting the `--max_num_destroy` argument, and $H$ can be adjusted by setting the `--rrt_start_threshold_pct` argument in `tune.py`.

The following command solves all instances with the benchmark runtimes, $D=10$ and $H=0.05$.
``` python 
poetry run python tune.py instances/*.json --benchmark_runtime --max_num_destroy 10 --rrt_start_threshold_pct 0.05
```
