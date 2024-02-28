from pathlib import Path

import pytest

from ras.classes.ProblemData import ProblemData


@pytest.fixture(scope="session")
def six_clients():
    """
    Instance with six clients.
    """
    loc = Path(__file__).resolve().parent / "data/six_clients.json"
    return ProblemData.from_file(loc)
