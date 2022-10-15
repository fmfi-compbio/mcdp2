from collections import defaultdict
from enum import IntEnum
import numpy as np

import cpp_extensions.cpp_module


def filter_intervals_by_chrnames(intervals, chr_names):
    chr_names = set(chr_names)
    result = [(t, b, e) for t, b, e in intervals if t in chr_names]
    return result


def merge_nondisjoint_intervals(intervals):
    intervals = filter(lambda interval: interval[1] < interval[2], intervals)
    intervals = sorted(intervals)
    if len(intervals) < 2:
        return intervals
    result = []
    c, b, e = intervals[0]
    for c2, b2, e2 in intervals[1:]:
        if c2 != c:
            result.append((c, b, e))
            c, b, e = c2, b2, e2
            continue
        if e < b2:
            result.append((c, b, e))
            c, b, e = c2, b2, e2
            continue
        e = max(e, e2)
    result.append((c, b, e))
    return result


def split_intervals_by_chrname(intervals):
    result = defaultdict(list)
    for chrom, b, e in sorted(intervals):
        result[chrom].append((b, e))
    return dict(result)


def count_overlaps_genome(reference, query):
    REFERENCE_CLOSE = 0
    QUERY_CLOSE = 1
    REFERENCE_OPEN = 2
    QUERY_OPEN = 3
    events = []
    for c, b, e in reference:
        events.append((c, b, REFERENCE_OPEN))
        events.append((c, e, REFERENCE_CLOSE))
    for c, b, e in query:
        events.append((c, b, QUERY_OPEN))
        events.append((c, e, QUERY_CLOSE))

    result = 0
    events = sorted(events)
    is_ref_opened = False
    is_touched_already = False
    is_query_opened = False
    for c, pos, t in events:
        if t == REFERENCE_CLOSE:
            if is_query_opened and not is_touched_already:
                result += 1
            is_ref_opened = is_touched_already = False
        elif t == QUERY_CLOSE:
            is_query_opened = False
            if is_ref_opened and not is_touched_already:
                result += 1
                is_touched_already = True
        elif t == REFERENCE_OPEN:
            is_ref_opened = True
            is_touched_already = False
        elif t == QUERY_OPEN:
            is_query_opened = True
            if is_ref_opened and not is_touched_already:
                result += 1
                is_touched_already = True
    return result


def count_overlaps_chr(reference, query):
    ref_mod = [("a", b, e) for b, e in reference]
    qu_mod = [("a", b, e) for b, e in query]
    return count_overlaps_genome(ref_mod, qu_mod)


def mask_intervals(intervals, masking):
    class Event(IntEnum):
        INT_CLOSE = 0
        MASK_CLOSE = 1
        MASK_OPEN = 2
        INT_OPEN = 3

    if len(intervals) == 0:
        return []
    if len(masking) == 0:
        return intervals

    events = []
    for c, b, e in intervals:
        events.append((c, b, Event.INT_OPEN))
        events.append((c, e, Event.INT_CLOSE))
    for c, b, e in masking:
        events.append((c, b, Event.MASK_OPEN))
        events.append((c, e, Event.MASK_CLOSE))

    result = []
    prev_location = None
    is_mask_open = False
    is_int_open = False
    for c, pos, e in sorted(events):
        match e:
            case Event.INT_OPEN:
                is_int_open = True
            case Event.INT_CLOSE:
                if not is_mask_open:
                    result.append((c, prev_location[1], pos))
                is_int_open = False
            case Event.MASK_OPEN:
                if is_int_open:
                    result.append((c, prev_location[1], pos))
                is_mask_open = True
            case Event.MASK_CLOSE:
                is_mask_open = False
        prev_location = (c, pos)
    return result


def matrix_pow(base, e):
    if e == 0:
        return np.identity(base.shape[0], dtype=np.longdouble)
    elif e == 1:
        return base
    elif e % 2 == 0:
        return matrix_pow(base @ base, e // 2)
    else:
        return base @ matrix_pow(base @ base, e // 2)


def compute_joint_pmf_py(pmfs):
    if len(pmfs) == 0:
        return []
    elif len(pmfs) == 1:
        return pmfs[0]

    current_line = pmfs[0]
    for num in range(1, len(pmfs)):
        pmf = pmfs[num]
        new_total_length = len(current_line) + len(pmf) - 1
        next_line = [0 for _ in range(new_total_length)]
        for i in range(new_total_length):
            # 0 <= j < len(current_line) && 0 <= i - j < len(pmf)
            # --> 0 >= j - i > - len(pmf)
            # --> i >= j > i - len(pmf)
            # -----> max(0, i - len(pmf)+1) <= j < min(i+1, len(current_line))
            for j in range(max(0, i - len(pmf) + 1), min(i + 1, len(current_line))):
                next_line[i] += current_line[j] * pmf[i - j]
        current_line = next_line
    return current_line


def compute_joint_pmf_cpp(pmfs):
    flat_data, sizes = _flatten_pmfs(pmfs)
    result = cpp_extensions.cpp_module.compute_joint_pmf_flattened(flat_data, sizes)
    return result


compute_joint_pmf = compute_joint_pmf_cpp


def _flatten_pmfs(pmfs):
    flat_data = [x for pmf in pmfs for x in pmf]
    sizes = [len(pmf) for pmf in pmfs]
    return flat_data, sizes


def compute_mean_from_pmf(pmf):
    result = sum(value * prob for value, prob in enumerate(pmf))
    return result


def compute_sd_from_pmf(pmf):
    mean = compute_mean_from_pmf(pmf)
    result = np.sum((value - mean) ** 2 * prob for value, prob in enumerate(pmf))
    return np.sqrt(result)


def compute_z_score(value, mean, sd):
    if sd > 0:
        z_score = (value - mean) / sd
    else:
        z_score = 0
    return z_score
