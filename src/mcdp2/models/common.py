from collections import defaultdict

import numpy as np
import pathos.multiprocessing

from mcdp2.common.helpers import split_intervals_by_chrname


def _count_transitions(query, context):
    t00 = 0
    t01 = 1
    t10 = 2
    t11 = 3
    CTX = 0
    Q = 1
    result = defaultdict(lambda: [0, 0, 0, 0])

    query_by_chrom = split_intervals_by_chrname(query)

    for name, context_intervals in context.items():
        query_intervals = sorted(query_by_chrom.get(name, []))
        chromosome_length = context_intervals[-1][2]
        events = []
        for ctx, b, e in context_intervals:
            if b == 0:
                b = 1
            events.append((b, CTX, ctx))
            if ctx not in result:
                result[ctx] = [0, 0, 0, 0]
        for b, e in query_intervals:
            if b > 0:
                # a special case
                events.append((b, Q, t01))
            if e - b > 1:
                events.append((b + 1, Q, t11))
            events.append((e, Q, t10))
            events.append((min(e + 1, chromosome_length), Q, t00))
        if (len(query_intervals) > 0 and query_intervals[0][0] >= 2) \
                or len(query_intervals) == 0:
            events.append((1, Q, t00))
        events.append((chromosome_length, Q, 0))
        events = sorted(events)

        current_context: int = None
        current_transition: int = None
        last_pos = 1
        for pos, t, x in events:
            if current_context is not None and current_transition is not None:
                length = pos - last_pos
                result[current_context][current_transition] += length
            last_pos = pos
            if t == CTX:
                current_context = x
            else:
                current_transition = x

    return dict(result)


def map_parallel(fn, inputs, threads):
    if threads > 1:
        with pathos.multiprocessing.Pool(threads-1) as pool:
            result = pool.map(fn, inputs)
    else:
        result = list(map(fn, inputs))
    return result


class MC:
    def __init__(self, tprobs, iprobs=None):
        self.tprobs = np.array(tprobs, dtype=np.longdouble)
        if iprobs is None:
            iprobs = MC.count_equilibrium(self.tprobs)
        self.iprobs = np.array(iprobs, dtype=np.longdouble)

    @staticmethod
    def from_transitions(transitions, pseudocount=1):
        tt = [x + pseudocount for x in transitions]
        t1 = tt[0] + tt[1]
        t2 = tt[2] + tt[3]
        tprobs = np.array([[tt[0] / t1, tt[1] / t1], [tt[2] / t2, tt[3] / t2]],
                          dtype=np.longdouble)
        return MC(tprobs)

    @staticmethod
    def count_equilibrium(tprobs):
        eigenvalues, eigenvectors = np.linalg.eig(tprobs.astype(float))
        inner = np.diag([1 if x > 0.99 else 0 for x in eigenvalues])
        p_inv = np.linalg.inv(eigenvectors)
        equilibrium = eigenvectors @ inner @ p_inv
        return np.array(equilibrium[0, :], dtype=np.longdouble)

    def __repr__(self):
        return f"MC(t: {self.tprobs}, init: {self.iprobs})"
