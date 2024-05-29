# Routing and appointment scheduling

This repository contains the code implementation of our paper [*A queueing-based approach for integrated routing and appointment scheduling*](https://doi.org/10.1016/j.ejor.2024.05.038).

## Installation

To use this repository, make sure to have [Poetry](https://python-poetry.org/) installed with version 1.2 or higher. The following command will then create a virtual environment with all necessary dependencies:

```shell
poetry install
```

If you don't want to use Poetry, you can use install all packages listed in the `pyproject.toml` file (requires Python 3.9 or higher).


## Usage

You can use the `benchmark.py` script to solve instances. Here's an example:

``` shell
poetry run python benchmark.py instances/n6-idx0-distribution0-travel0-serv1.json \
--num_procs 4 \
--weight_travel 1 \
--weight_idle 2.5 \
--weight_wait 10 \
--algorithm lns \
--seed 1 \
--max_runtime 1
```

This command solves an instance with six clients with high service time variance. 
The objective weights are 1, 2.5 and 10 for the travel, idle and waiting time, respectively. The selected algorithm is LNS with seed 1 and runs for one second.

All results presented in our paper can be found in this repository. In particular, the raw results data is stored in `data/` and the notebooks that generate tables and figures can be found in `notebooks/`.


## Paper and citation

Please consider citing [our paper](https://doi.org/10.1016/j.ejor.2024.05.038) (open access) if this repository has been useful to you:

``` bibtex
@article{Bekker_et_al2024,
title = {A queueing-based approach for integrated routing and appointment scheduling},
journal = {European Journal of Operational Research},
year = {2024},
issn = {0377-2217},
doi = {https://doi.org/10.1016/j.ejor.2024.05.038},
url = {https://www.sciencedirect.com/science/article/pii/S0377221724003977},
author = {René Bekker and Bharti Bharti and Leon Lan and Michel Mandjes},
}
```
