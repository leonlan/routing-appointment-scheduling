# Routing and appointment scheduling

This repository contains the code implementation of our paper [*A queueing-based approach for integrated routing and appointment scheduling*](preprint.pdf).

## Installation

To use this repository, make sure to have [Poetry](https://python-poetry.org/) installed with version 1.2 or higher. The following command will then create a virtual environment with all necessary dependencies:

```shell
poetry install
```

If you don't want to use Poetry, you can use install all packages listed in the `pyproject.toml` file (requires Python 3.9 or higher).


## Usage

You can use the `benchmark.py` script to solve instances. Here's an example:

``` shell
poetry run python benchmark.py instances/*.json \
--num_procs 4 \
--weight_travel 1 \
--weight_idle 2.5 \
--weight_wait 10 \
--algorithm lns \
--seed 1 \
--max_runtime 1
```

This command solves all instances, four in parallel at a time. The objective weights are 1, 2.5 and 10 for the travel, idle and waiting time contributions, respectively. The selected algorithm is LNS with seed 1 and runs for one second.

All results presented in our paper can be found in this repository. In particular, the raw results data is stored in `data/` and the notebooks that generate tables and figures can be found in `notebooks/`.
