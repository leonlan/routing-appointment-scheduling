import numpy as np
import numpy.random as rnd
from numpy.testing import assert_allclose

from ras import full_enumeration, tsp
from ras.appointment.true_optimal import compute_idle_wait as true_objective_function
from ras.classes import CostEvaluator, ProblemData
from ras.saa.utils import (
    _sample_distance_matrices,
    _sample_service_times,
)
from ras.saa.zhan import zhan_mip


def test_zhan_equals_tsp():
    """
    Tests that Zhan's SAA reduces to TSP with only one sample.
    """
    seed = 1
    loc = "tests/data/six_clients.json"
    data = ProblemData.from_file(loc)
    cost_evaluator = CostEvaluator(
        true_objective_function, 1, 2, np.array([3, 3, 3, 3, 3, 3, 3])
    )
    max_runtime = 10
    num_scenarios = 1

    # Because there is only one scenario, the appointment scheduling aspect of RAS
    # is redundant and the only goal is to find the best TSP tour. The scenario
    # is not used for the travel time objective.
    m, *_ = zhan_mip(seed, data, cost_evaluator, max_runtime, num_scenarios)
    objective = m.getObjective().getValue()

    # Compare against the total distance from a TSP-obtained tour.
    result = tsp(seed, data, cost_evaluator, 100)
    visits = result.solution.visits
    frm, to = [0] + visits, visits + [0]
    expected = data.distances[frm, to].sum()

    assert_allclose(objective, expected)


def test_zhan_equals_full_enumeration():
    """
    Tests that Zhan's SAA reduces to full enumeration with enough samples.
    """
    seed = 1
    loc = "tests/data/two_clients.json"
    data = ProblemData.from_file(loc)
    cost_evaluator = CostEvaluator(true_objective_function, 1, 2, np.array([3, 4, 2]))
    max_runtime = 10
    num_scenarios = 1000

    m, *_ = zhan_mip(seed, data, cost_evaluator, max_runtime, num_scenarios)
    objective = m.getObjective().getValue()

    res = full_enumeration(seed, data, cost_evaluator)
    expected = res.solution.objective()

    assert_allclose(objective, expected, rtol=0.5)


def test_sample_distances_matrices(six_clients):
    """
    Tests that the mean and scv of the sampled distances matrix is close to the
    original distances matrix with enough samples.
    """
    rng = rnd.default_rng(seed=1)
    samples = _sample_distance_matrices(six_clients, 100000, rng)

    mean = np.mean(samples, axis=2)

    assert_allclose(mean, six_clients.distances, atol=0.2)


def test_sample_service_times(six_clients):
    """
    Tests that the mean and scv of the sampled service times is close to the
    original service times with enough samples.
    """
    rng = rnd.default_rng(seed=1)
    samples = _sample_service_times(six_clients, 100000, rng)

    mean = np.mean(samples, axis=1)

    assert_allclose(mean, six_clients.service, atol=0.2)
