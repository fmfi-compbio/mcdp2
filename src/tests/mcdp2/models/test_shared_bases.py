import logging

import pytest

from mcdp2.models.common import MC
from mcdp2.models.shared_bases import count_bases_chr, shared_bases_approx, \
    _compute_gap_interval_ctx_labels, sample_number_of_shared_bases, shared_bases_approx_sampled


@pytest.mark.parametrize("xs,ys,expected", [
    ([], [], 0),
    ([(0, 1)], [(0, 1)], 1),
    ([(0, 10)], [(0, 10)], 10),
    ([(0, 10)], [(0, 100)], 10),
    ([(0, 10), (20, 40)], [(0, 100)], 30),
    ([(0, 100)], [(0, 10), (20, 40)], 30),
    ([(0, 10), (20, 40)], [(50, 100)], 0),
    ([(0, 10), (20, 40)], [(5, 20), (50, 100), ], 5),
    ([(0, 10), (20, 40)], [(5, 21), (50, 100), ], 6),
])
def test_count_bases_chr(xs, ys, expected):
    result = count_bases_chr(xs, ys)
    assert result == expected


class TestComputeGapIntervalCtxLabels:
    @pytest.mark.parametrize("rs,ctx,expected", [
        ([], [], ([], [])),
        ([(0, 1)], [("a", 0, 1)], ([[]], [[("a", 1)]])),
        ([(6, 10)], [("a", 0, 20)], ([[("a", 6)]], [[("a", 4)]])),
        ([(6, 10), (12, 15)], [("a", 0, 20)],
         ([[("a", 6)], [("a", 2)]], [[("a", 4)], [("a", 3)]])),
        ([(6, 10), (12, 15)], [("a", 0, 8), ("b", 8, 11), ("a", 11, 20)],
         ([[("a", 6)], [("b", 1), ("a", 1),]], [[('a', 2), ('b', 2)], [("a", 3)]])),
    ])
    def test_by_hand(self, rs, ctx, expected):
        result = _compute_gap_interval_ctx_labels(rs, ctx)
        assert result == expected


class TestSharedBasesApprox:
    @pytest.mark.parametrize("rs,mc,ctx,expected", [
        ([], {"a": None}, [("a", 0, 1)], (0, 0)),
        ([(1, 2)], {"a": MC([[1e-12, 1-1e-12], [1e-12, 1-1e-12]])}, [("a", 0, 2)], (1, 0)),
        ([(1, 2)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 2)], (0.5, 0.25)),
        ([(1, 21)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 21)], (10, 5)),
        ([(0, 20)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 21)], (10, 5)),
        ([(0, 20), (40, 80)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 100)], (30, 15)),
        ([(0, 20), (40, 80)], {"a": MC([[0.5, 0.5], [0.5, 0.5]]), "b": MC([[0.5, 0.5], [0.5, 0.5]])},
         [("a", 0, 10), ("b", 10, 25), ("a", 25, 48), ("b", 48, 49), ("a", 49, 50), ("a", 50, 100)], (30, 15)),
    ])
    def test_by_hand(self, rs, mc, ctx, expected):
        logger = logging.getLogger("mcdp2.shared_bases")
        logger.setLevel(logging.DEBUG)
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler())
        result = shared_bases_approx(rs=rs, ctx=ctx, markov_chains=mc)
        assert result == pytest.approx(expected)

    @pytest.mark.skip(reason="too long")
    @pytest.mark.parametrize("rs,mc,ctx", [
        ([(1, 2)], {"a": MC([[1e-12, 1 - 1e-12], [1e-12, 1 - 1e-12]])}, [("a", 0, 2)]),
        ([(1, 2)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 2)]),
        ([(1, 21)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 21)]),
        ([(0, 20)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 21)]),
        ([(0, 20), (40, 80)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 100)]),
        (
                [(0, 20), (40, 80)],
                {"a": MC([[0.5, 0.5], [0.5, 0.5]]), "b": MC([[0.5, 0.5], [0.5, 0.5]])},
                [("a", 0, 10), ("b", 10, 25), ("a", 25, 48), ("b", 48, 49), ("a", 49, 50),
                 ("a", 50, 100)],
        ),
        ([(1, 4), (8, 16), (17, 18), (30, 50)],
         {"a": MC([[0.4, 0.6], [0.8, 0.2]]), "b":MC([[0.47, 0.53], [0.97, 0.03]])},
         [("a", 0, 3), ("b", 3, 10), ("a", 10, 26), ("b", 26, 28), ("a", 28, 34), ("b", 34, 75)])

    ])
    def test_against_sampled(self, rs, mc, ctx):
        exact = shared_bases_approx(rs=rs, ctx=ctx, markov_chains=mc)
        sampled = shared_bases_approx_sampled(rs=rs, ctx=ctx, markov_chains=mc, tries=100000)
        assert exact == pytest.approx(sampled, rel=1e-2, abs=0.1)


class TestSampledNumberOfSharedBases:
    @pytest.mark.parametrize("rs,mc,ctx", [
        ([], {}, []),
        ([(0, 1)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 1)]),
    ])
    def test_sampler_produces_correct_range(self, rs, mc, ctx):
        number_of_shared_bases = sample_number_of_shared_bases(rs=rs, ctx=ctx, markov_chains=mc)
        max_res = sum(e - b for b, e in rs)
        assert 0 <= number_of_shared_bases <= max_res

    @pytest.mark.parametrize("rs,mc,ctx,expected", [
        ([(1, 2)], {"a": MC([[1e-12, 1 - 1e-12], [1e-12, 1 - 1e-12]])}, [("a", 0, 2)], (1, 0)),
        ([(1, 2)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 2)], (0.5, 0.25)),
        ([(1, 21)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 21)], (10, 5)),
        ([(0, 20)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 21)], (10, 5)),
        ([(0, 20), (40, 80)], {"a": MC([[0.5, 0.5], [0.5, 0.5]])}, [("a", 0, 100)], (30, 15)),
        (
        [(0, 20), (40, 80)], {"a": MC([[0.5, 0.5], [0.5, 0.5]]), "b": MC([[0.5, 0.5], [0.5, 0.5]])},
        [("a", 0, 10), ("b", 10, 25), ("a", 25, 48), ("b", 48, 49), ("a", 49, 50), ("a", 50, 100)],
        (30, 15)),
    ])
    def test_sampling_is_circa_correct(self, rs, mc, ctx, expected):
        result = shared_bases_approx_sampled(rs=rs, ctx=ctx, markov_chains=mc, tries=1000)
        assert result == pytest.approx(expected, rel=1e-1)

