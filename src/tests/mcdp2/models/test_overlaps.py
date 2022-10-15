import logging

import numpy as np
import pytest

from mcdp2.models.overlaps import _count_transitions, MC, compute_pvalue, _compute_overlap_mcs, \
    _compute_pmf_from_mcs_dp, normal_approx


@pytest.mark.skip
@pytest.mark.parametrize("ref,query,chrlens,context,expected", [
    ([('c1', 5, 10), ('c1', 20, 30)],
     [('c1', 0, 2), ('c1', 4, 8)],
     [('c1', 1000)],
     {'c1': [('A', 0, 1000)]},
     {'total-pvalue': 1}),
])
def test_compute_pvalue(ref, query, chrlens, context, expected):
    logger = logging.getLogger("mcdp2")
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler())
    result = compute_pvalue(ref,query,chrlens,context)
    assert result == expected


@pytest.mark.parametrize("query,context,expected", [
    ([], {'a': [("A", 0, 10)]},
     {"A": [9, 0, 0, 0]}),
    ([], {'a': [("A", 0, 10)], 'b': [("A", 0, 100)]},
     {"A": [108, 0, 0, 0]}),
    ([], {'a': [("A", 0, 10)], 'b': [("B", 0, 100)]},
     {"A": [9, 0, 0, 0], "B": [99, 0, 0, 0]}),
    ([('a', 1, 2)], {'a': [("A", 0, 2)]}, {"A": [0, 1, 0, 0]}),
    ([('a', 1, 2), ('a', 5, 9)], {'a': [("A", 0, 2), ("B", 2, 7), ("C", 7, 11)]},
     {"A": [0, 1, 0, 0], "B": [2, 1, 1, 1], "C": [1, 0, 1, 2]}),
    ([('a', 0, 1)], {'a': [("A", 0, 4)]}, {"A": [2, 0, 1, 0]}),
    ([('a', 0, 1)], {'a': [("A", 0, 1), ("B", 1, 4)]},
     {"A": [0, 0, 0, 0], "B": [2, 0, 1, 0]}),
])
def test_count_transitions(query, context, expected):
    result = _count_transitions(query, context)
    assert result == expected


@pytest.mark.parametrize("tprobs,expected", [
    ([[0.8, 0.2], [0.6, 0.4]], [0.75, 0.25]),
    ([[0.7, 0.3], [0.6, 0.4]], [2/3, 1/3]),
    ([[0.7, 0.3], [0.5, 0.5]], [0.625, 0.375]),
])
def test_mc_count_equilibrium(tprobs, expected):
    result = MC.count_equilibrium(np.array(tprobs))
    assert result == pytest.approx(np.array(expected))


@pytest.mark.parametrize("ref,context,mcs,expected", [
    ([], [], {}, []),
    ([(0, 1)], [('A', 0, 1)], {"A": MC([[0,0],[0,0]], [0,0])},
     [(MC([[0,0],[0,0]], [0,0]), MC([[0,0],[0,0]], [0,0]))]),
    ([(1, 2)], [('A', 0, 2)], {"A": MC([[0.5,0.5],[0.5,0.5]], [4,2])},
     [(MC([[0.5,0.0],[0.5,0.0]], [4,2]), MC([[0.5,0.5],[0.5,0.5]], [4,2]))]),
    ([(2, 4)], [('A', 0, 4)], {"A": MC([[0.1, 0.9], [0.8, 0.2]], [4, 7])},
     [(MC([[0.0289, 0], [0.0632, 0]], [4, 7]), MC([[0.5977, 0.4023], [0.3576, 0.6424]], [4, 7]))]),
    ([(2, 4)],
     [('A', 0, 3), ('B', 3, 7)],
     {"A": MC([[0.1, 0.9], [0.8, 0.2]], [4, 7]), "B": MC([[0.3, 0.7], [0.4, 0.6]])},
     [(MC([[0.0867, 0], [0.1896, 0]], [4, 7]), MC([[0.3711, 0.6289], [0.3368, 0.6632]], [4, 7]))]),
    ([(2, 4), (5, 6)],
     [('A', 0, 3), ('B', 3, 7)],
     {"A": MC([[0.1, 0.9], [0.8, 0.2]], [4, 7]), "B": MC([[0.3, 0.7], [0.4, 0.6]], [7, 2])},
     [(MC([[0.0867, 0], [0.1896, 0]], [4, 7]), MC([[0.3711, 0.6289], [0.3368, 0.6632]], [4, 7])),
      (MC([[0.37, 0], [0.36, 0]], [7, 2]), MC([[0.37,0.63], [0.36, 0.64]], [7,2]))]),
    ([(0, 1)],
     [('A', 0, 1)],
     {"A": MC([[0.4, 0.6], [0.2, 0.8]], [47, 42])},
     [(MC([[0.4, 0], [0.2, 0]], [47, 42]), MC([[0.4, 0.6], [0.2, 0.8]], [47, 42]))]),
    ([(0, 1), (4, 8)],
     [('A', 0, 1), ('B', 1, 3), ('C', 3, 4), ('B', 4, 8), ('A', 8, 9)],
     {"A": MC([[0.4, 0.6], [0.2, 0.8]], [47, 42]),
      "B": MC([[0.3, 0.7], [0.9, 0.1]], [-1, -2]),
      "C": MC([[0.5, 0.5], [0.7, 0.3]], [7, 9])},
     [(MC([[0.4, 0], [0.2, 0]], [47, 42]), MC([[0.4, 0.6], [0.2, 0.8]], [47, 42])),
      (MC([[0.0152928, 0.0], [0.0141264, 0.0]], [-1, -2]), MC([[0.561658, 0.438342], [0.570989, 0.429011]], [-1, -2])),]),
])
def test_compute_overlap_mcs(ref, context, mcs, expected):
    result = _compute_overlap_mcs(ref, context, mcs)
    assert len(result) == len(expected)
    for (r_nohit, r_any), (e_nohit, e_any) in zip(result, expected):
        assert r_nohit.tprobs == pytest.approx(e_nohit.tprobs)
        assert r_nohit.iprobs == pytest.approx(e_nohit.iprobs)
        assert r_any.tprobs == pytest.approx(e_any.tprobs)
        assert r_any.iprobs == pytest.approx(e_any.iprobs)


m1 = MC([[1, 0],[1, 0]])
m2 = MC([[0, 1], [0, 1]])
m3 = MC([[0.5, 0.5], [0.5, 0.5]])
m4 = MC([[0.5, 0], [0.5, 0]])
@pytest.mark.parametrize("mcs,expected", [
    ([], [1]),
    ([(m1, m1)], [1, 0]),
    ([(m1, m1), (m1, m1)], [1, 0, 0]),
    ([(m1, m1), (m1, m1), (m1, m1)], [1, 0, 0, 0]),
    ([(MC([[0, 0], [0, 0]]), m2)], [0, 1]),
    ([(MC([[0, 0], [0, 0]]), m2), (MC([[0, 0], [0, 0]]), m2), ], [0, 0, 1]),
    ([(m4, m3)], [1/2, 1/2]),
    ([(m4, m3), (m4, m3)], [1/4, 1/2, 1/4]),
    ([(m4, m3), (m4, m3), (m4, m3)], [1/8, 3/8, 3/8, 1/8]),
    ([(m4, m3), (m4, m3), (m4, m3), (m4, m3)], [1/16, 4/16, 6/16, 4/16, 1/16]),
])
def test_compute_pmf_from_mcs_dp(mcs, expected):
    result = _compute_pmf_from_mcs_dp(mcs)
    assert result == pytest.approx(expected)


class TestNormalApprox:
    def test_empty_ref(self):
        result = normal_approx([])
        expected = 0, 0
        assert result == expected

    @pytest.mark.parametrize("ref,query,context", [
        ([('c1', 1, 2)], [('c1', 1, 2)], {'c1': [('A', 0, 3)]}),
        ([('c1', 1, 2)], [('c1', 1, 2)], {'c1': [('A', 0, 10)]}),
        ([('c1', 1, 2)], [('c1', 1, 2)], {'c1': [('A', 0, 1000)]}),
        ([('c1', 1, 2)], [('c1', 1, 2)], {'c1': [('A', 0, 1000000)]}),
        ([('c1', 1000, 2000)], [('c1', 1, 2000)], {'c1': [('A', 0, 1000000)]}),
        ([('c1', 1, 2)], [('c1', 1, 2), ('c1', 4, 9)], {'c1': [('A', 0, 10)]}),
        ([('c1', 1, 2), ('c1', 3, 4)], [('c1', 1, 2), ('c1', 4, 9)], {'c1': [('A', 0, 10)]}),
        ([('c1', 1000, 1001), ('c1', 1002, 1003)], [('c1', 1, 2), ('c1', 4, 9)], {'c1': [('A', 0, 10000)]}),
        ([('c1', 1, 2), ('c1', 3, 8)], [('c1', 1, 2), ('c1', 4, 9)], {'c1': [('A', 0, 10)]}),
    ])
    def test_mean_sd_against_dp(self, ref, query, context):
        logger = logging.getLogger("mcdp2")
        logger.setLevel(logging.DEBUG)
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler())
        computation = compute_pvalue(ref, query, None, context, algorithm="both")
        dp_mean, dp_sd = computation['total']['full_dp_mean'], computation['total']['full_dp_sd']
        clt_mean, clt_sd = computation['total']['clt_mean'], computation['total']['clt_sd']
        assert clt_mean == pytest.approx(dp_mean)
        assert clt_sd == pytest.approx(dp_sd)
