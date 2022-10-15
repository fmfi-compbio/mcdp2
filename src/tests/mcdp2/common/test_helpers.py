import itertools
import time

import pytest
from mcdp2.common.helpers import *
from mcdp2.common.helpers import compute_joint_pmf_py, compute_joint_pmf_cpp


@pytest.mark.parametrize("intervals,chrnames,expected", [
    ([], [], []),
    ([('a', 1, 10)], ['a'], [('a', 1, 10)]),
    ([('a', 1, 10)], ['b'], []),
    ([('a', 1, 10), ('b', 2, 100)], ['b'], [('b', 2, 100)]),
])
def test_filter_intervals_by_chrnames(intervals, chrnames, expected):
    result = filter_intervals_by_chrnames(intervals, chrnames)
    assert result == expected


class TestMergeNondisjointIntervals:
    cases = [
        ([], []),
        ([('a', 2, 1),], []),
        ([('a', 2, 1), ('a', 10, 4)], []),
        ([('a', 1, 2)], [('a', 1, 2)]),
        ([('a', 1, 2), ('a', 3, 4)], [('a', 1, 2), ('a', 3, 4)]),
        ([('a', 1, 2), ('a', 2, 3), ('a', 3, 4), ('a', 5, 6)], [('a', 1, 4), ('a', 5, 6)]),
        ([('a', 1, 2), ('a', 1, 2)], [('a', 1, 2)]),
        ([('a', 1, 10), ('a', 10, 11)], [('a', 1, 11)]),
        ([('a', 1, 70), ('a', 10, 11)], [('a', 1, 70)]),
        ([('a', 1, 70), ('a', 2, 70)], [('a', 1, 70)]),
        ([('b', 1, 10), ('a', 10, 11)], [('a', 10, 11), ('b', 1, 10)]),
        ([('a', 1, 10), ('a', 10, 11), ('a', 5, 100)], [('a', 1, 100),]),
        ([('a', 1, 5), ('a', 10, 11), ('a', 5, 100)], [('a', 1, 100),]),
    ]

    @pytest.mark.parametrize("intervals,expected", cases)
    def test_merge_nondisjoint_intervals(self, intervals, expected):
        result = merge_nondisjoint_intervals(intervals)
        assert result == expected

    @staticmethod
    def old_implementation(intervals):
        intervals = sorted(filter(lambda x: x[1] < x[2], intervals))
        if len(intervals) == 0:
            return []

        output = []
        open_int = intervals[0]
        for c, b, e in itertools.chain(intervals[1:], [(None, None, None)]):
            if open_int[0] != c or open_int[2] < b:
                # we can close the opened interval
                output.append(open_int)
                open_int = c, b, e
            else:
                # current interval overlaps with the opened, merge
                open_int = c, open_int[1], e
        return output

    @pytest.mark.skip(reason="The old implementation is indeed incorrect")
    @pytest.mark.parametrize("intervals,expected", cases)
    def test_old_implementation(self, intervals, expected):
        result = self.old_implementation(intervals)
        assert result == expected


@pytest.mark.parametrize("reference,query,expected", [
    ([], [], 0),
    ([('a', 0, 1)], [], 0),
    ([('a', 0, 1)], [('a', 0, 1)], 1),
    ([('a', 0, 1)], [('a', 0, 1), ('a', 1, 2)], 1),
    ([('a', 0, 1), ('a', 0, 1)], [('a', 0, 1), ('a', 1, 2)], 2),
    ([('a', 0, 1), ('a', 0, 1)], [('a', 0, 2)], 2),
    ([('a', 0, 2)], [('a', 0, 1), ('a', 1, 2)], 1),
    ([('a', 0, 2)], [('a', 0, 1), ('b', 1, 2)], 1),
    ([('a', 0, 2)], [('b', 0, 1), ('b', 1, 2)], 0),
    ([("a", 1, 10)], [("a", 30, 40)], 0),
    ([("a", 1, 10)], [("a", 2, 15)], 1),
    ([("a", 1, 10)], [("a", 10, 15)], 0),
    ([("a", 1, 10)], [("b", 1, 10)], 0),
    ([("a", 1, 10)], [("a", 0, 1)], 0),
    ([("a", 1, 10)], [("a", 0, 1), ("a", 10, 12)], 0),
    ([("a", 0, 10)], [("a", 0, 1), ("a", 10, 12)], 1),
    ([("a", 0, 11)], [("a", 0, 1), ("a", 10, 12)], 1),
    ([("a", 0, 11), ("a", 12, 20)], [("a", 0, 1), ("a", 10, 12)], 1),
    ([("a", 0, 11), ("a", 12, 20)], [("a", 0, 1), ("a", 10, 12), ("a", 14, 20)], 2),
    ([("a", 0, 11), ("a", 12, 20), ("a", 70, 71)],
     [("a", 0, 1), ("a", 10, 12), ("a", 14, 20), ("a", 70, 71)], 3),
    ([("a", 0, 11), ("a", 12, 20), ("a", 70, 71)],
     [("a", 0, 1), ("a", 10, 12), ("a", 14, 20), ("a", 69, 71)], 3),
    ([("a", 0, 11), ("a", 12, 20), ("a", 70, 71)],
     [("a", 0, 1), ("a", 10, 12), ("a", 14, 20), ("a", 69, 72)], 3),
    ([("a", 0, 2), ("a", 10, 15), ("a", 16, 18), ("a", 20, 21)],
     [("a", 0, 2), ("a", 9, 12), ("a", 17, 20), ("a", 69, 72)], 3),
])
def test_count_overlaps_genome(reference, query, expected):
    result = count_overlaps_genome(reference, query)
    assert result == expected


class TestMaskIntervals:
    @pytest.mark.parametrize("intervals,masking,expected", [
        ([], [], []),
        ([('c1', 1, 10)], [('c1', 1, 10)], []),
        ([('c1', 1, 10)], [('c1', 0, 10)], []),
        ([('c1', 1, 10)], [('c1', 0, 16)], []),
        ([('c1', 0, 10)], [('c1', 0, 16)], []),
        ([('c1', 1, 10)], [('c1', 1, 8)], [('c1', 8, 10)]),
        ([('c1', 1, 10)], [('c1', 4, 8)], [('c1', 1, 4), ('c1', 8, 10)]),
        ([('c1', 1, 10), ('c1', 12, 20)],
         [('c1', 4, 18)],
         [('c1', 1, 4), ('c1', 18, 20)]),
        ([('c1', 1, 100)],
         [('c1', 10, 20), ('c1', 30, 40), ('c1', 60, 75)],
         [('c1', 1, 10), ('c1', 20, 30), ('c1', 40, 60), ('c1', 75, 100)]),
    ])
    def test_masking(self, intervals, masking, expected):
        result = mask_intervals(intervals, masking)
        assert result == expected


@pytest.mark.parametrize("pmfs,expected", [
    ([], []),
    ([[1,2,3,4,5]], [1,2,3,4,5]),
    ([[1], [1]], [1]),
    ([[0.5, 0.5], [1]], [0.5, 0.5]),
    ([[1,2,3,4,5], [1]], [1,2,3,4,5]),
    ([[1,2,3,4,5], [1], [1]], [1,2,3,4,5]),
    ([[1,2,3,4,5], [1], [1], [1]], [1,2,3,4,5]),
    ([[1/2, 1/2], [1/2, 1/2]], [1/4, 1/2, 1/4]),
    ([[1/2, 1/2], [1/2, 1/2], [1/2, 1/2]], [1/8, 3/8, 3/8, 1/8]),
    ([[1/2, 1/2], [1/2, 1/2], [1/2, 1/2], [0, 0, 0, 1]], [0, 0, 0, 1/8, 3/8, 3/8, 1/8]),
    ([[1/2, 1/2], [1/2, 1/2], [1/2, 1/2], [1/2, 1/2]], [1/16, 4/16, 6/16, 4/16, 1/16]),
    ([[1/3,1/3,1/3], [1/3,1/3,1/3]], [1/9, 2/9, 3/9, 2/9, 1/9]),
])
def test_compute_joint_pmf(pmfs, expected):
    result = compute_joint_pmf_py(pmfs)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("pmfs", [
    [],
    [[1, 2, 3, 4, 5]],
    [[1], [1]],
    [[0.5, 0.5], [1]],
    [[1,2,3,4,5], [1]],
    [[1,2,3,4,5], [1], [1]],
    [[1,2,3,4,5], [1], [1], [1]],
    [[1/2, 1/2], [1/2, 1/2]],
    [[1/2, 1/2], [1/2, 1/2], [1/2, 1/2]],
    [[1/2, 1/2], [1/2, 1/2], [1/2, 1/2], [0.0, 0.0, 0.0, 1.0]],
    [[1/2, 1/2], [1/2, 1/2], [1/2, 1/2], [1/2, 1/2]],
    [[1/3,1/3,1/3], [1/3,1/3,1/3]],
])
def test_compare_cpp_joint_pmf_with_python(pmfs):
    result = compute_joint_pmf_cpp(pmfs)
    expected = compute_joint_pmf_py(pmfs)
    assert result == pytest.approx(expected)


@pytest.mark.skip(reason="speed comparison, no verification")
@pytest.mark.parametrize("size, count", [
    (10, 2), (100, 2), (1000, 2), (2000, 2),
    (100, 5), (100, 10), (100, 50),
])
def test_compare_joint_pmf_speed(size, count):
    pmfs = [[1/size for _ in range(size)] for _ in range(count)]
    repeats = 5

    start_time = time.perf_counter()
    for _ in range(repeats):
        compute_joint_pmf_py(pmfs)
    end_time = time.perf_counter()
    avg_time_py = (end_time - start_time) / repeats

    start_time = time.perf_counter()
    for _ in range(repeats):
        compute_joint_pmf_cpp(pmfs)
    end_time = time.perf_counter()
    avg_time_cpp = (end_time - start_time) / repeats

    print(f"{size}x{count}:: Python time: {avg_time_py}, C++ time: {avg_time_cpp}")
