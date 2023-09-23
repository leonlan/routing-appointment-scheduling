from ast import literal_eval as make_tuple
from pathlib import Path
from typing import Union

import pandas as pd


def read_single_instance_benchmark_output(path: Union[str, Path]) -> pd.DataFrame:
    with open(path, "r") as fh:
        line = fh.read()

    print(line)
    row = list(make_tuple(line.strip()))

    # Parse the instance name to identify the instance characteristics
    values = [s.strip("nidxdistributiontravelserv") for s in row[0].split("-")]
    data = values + row[1:]

    headers = [
        "n",
        "idx",
        "distribution",
        "travel",
        "serv",
        "obj",
        "iters",
        "time",
        "alg",
        "cost_profile",
    ]
    dtypes = [int, int, int, int, int, float, float, float, str, str]
    return pd.DataFrame([data], columns=headers).astype(dict(zip(headers, dtypes)))
