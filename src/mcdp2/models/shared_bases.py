import logging
import math
import random
import statistics
from enum import IntEnum

import numpy as np
import scipy.stats

from mcdp2.common.helpers import split_intervals_by_chrname, matrix_pow, compute_z_score
from mcdp2.models.common import _count_transitions, MC, map_parallel


def compute_pvalue(reference,
                   query,
                   context,
                   threads: int = 4):
    """Compute shared bases statistics distribution for context-dependent two-state Markov chain"""
    logger = logging.getLogger("mcdp2")

    # train a Markov chain for each context
    transitions = _count_transitions(query, context)
    markov_chains = {ctx: MC.from_transitions(transitions) for ctx, transitions in
                     transitions.items()}

    ref_by_chrname = split_intervals_by_chrname(reference)
    query_by_chrname = split_intervals_by_chrname(query)
    chromosome_names = list(sorted(context.keys()))
    inputs = [
        (ref_by_chrname.get(chrname, []),
         query_by_chrname.get(chrname, []),
         context[chrname])
        for chrname in chromosome_names
    ]

    # compute number of shared bases for each chromosome
    fn = lambda x: count_bases_chr(x[0], x[1])
    shared_bases_counts = map_parallel(fn, inputs, threads)
    total_shared_bases_count = sum(shared_bases_counts)

    chromosome_lengts = [
        context[chrname][-1][-1]
        for chrname in chromosome_names
    ]
    reference_lengths = [
        sum(e-b for b, e in x[0])
        for x in inputs
    ]
    query_lengths = [
        sum(e - b for b, e in x[1])
        for x in inputs
    ]
    total_length = sum(chromosome_lengts)
    result = {
        "total": {
            "total_genome_length": total_length,
            "reference_size": len(reference),
            "reference_bases_count": int(sum(x for x in reference_lengths)),
            "reference_coverage": float(sum(x for x in reference_lengths) / total_length),
            "query_size": len(query),
            "query_bases_count": int(sum(x for x in query_lengths)),
            "query_coverage": float(sum(x for x in query_lengths) / total_length),
            "shared_bases_count": int(total_shared_bases_count),
            "shared_coverage": float(total_shared_bases_count / total_length)
        },
        "by_chromosomes": {
            name: {
                "reference_size": len(inputs[i][0]),
                "reference_bases_count": int(reference_lengths[i]),
                "reference_coverage": float(reference_lengths[i] / chromosome_lengts[i]),
                "query_size": len(inputs[i][1]),
                "query_bases_count": int(query_lengths[i]),
                "query_coverage": float(query_lengths[i] / chromosome_lengts[i]),
                "shared_bases_count": int(shared_bases_counts[i]),
                "shared_coverage": float(shared_bases_counts[i] / chromosome_lengts[i])
            }
            for i, name in enumerate(chromosome_names)
        },
    }

    # compute mean and variance of the number of shared bases
    fn = lambda x: shared_bases_approx(rs=x[0], ctx=x[2], markov_chains=markov_chains)
    approxes = map_parallel(fn=fn, inputs=inputs, threads=threads)
    zscores = [compute_z_score(value=shared_bases_counts[i],
                               mean=approxes[i][0],
                               sd=math.sqrt(approxes[i][1]))
               for i in range(len(chromosome_names))]
    pvalues = [scipy.stats.norm.sf(shared_bases_counts[i],
                                   approxes[i][0],
                                   math.sqrt(approxes[i][1]))
                              if approxes[i][1] > 1e-300 else 1
                              for i in range(len(chromosome_names))]
    total_approx = sum(x[0] for x in approxes), sum(x[1] for x in approxes)
    total_zscore = compute_z_score(total_shared_bases_count, total_approx[0], math.sqrt(total_approx[1]))
    total_pvalue = scipy.stats.norm.sf(total_shared_bases_count, total_approx[0], math.sqrt(total_approx[1])) \
        if total_approx[1] > 1e-300 else 1

    result['total'] |= {"mean_sb": float(total_approx[0]),
                        "sd_sb": float(math.sqrt(total_approx[1])),
                        "pvalue_sb": float(total_pvalue),
                        "zscore_sb": float(total_zscore), }
    for i, name in enumerate(chromosome_names):
        result['by_chromosomes'][name] |= {"mean_sb": float(approxes[i][0]),
                                           "sd_sb": float(math.sqrt(approxes[i][1])),
                                           "pvalue_sb": float(pvalues[i]),
                                           "zscore_sb": float(zscores[i]), }

    return result


def count_bases_chr(rs, qs):
    class Event(IntEnum):
        REND = 1
        QEND = 3
        RBEGIN = 0
        QBEGIN = 2

    events = []
    for b, e in rs:
        events.append((b, Event.RBEGIN))
        events.append((e, Event.REND))
    for b, e in qs:
        events.append((b, Event.QBEGIN))
        events.append((e, Event.QEND))
    events = sorted(events)

    last_pos = 0
    is_ref_open = False
    is_query_open = False
    shared_bases_count = 0
    for pos, event in events:
        match event:
            case Event.RBEGIN:
                is_ref_open = True
            case Event.REND:
                if is_query_open:
                    shared_bases_count += pos - last_pos
                is_ref_open = False
            case Event.QBEGIN:
                is_query_open = True
            case Event.QEND:
                if is_ref_open:
                    shared_bases_count += pos - last_pos
                is_query_open = False
        last_pos = pos

    return shared_bases_count


def shared_bases_approx(rs, ctx, markov_chains):
    """Returns mean and variance (sic!) of the number of shared bases on a single chromosome."""
    logger = logging.getLogger("mcdp2.shared_bases")
    m = len(rs)
    if m == 0:
        return 0, 0

    # Y_j := number of shared bases within j-th reference interval
    # T := \sum_{j=1}^m Y_j
    # we want to compute E[T] and Var[T]
    # S_j (0 <= j < m) := state at the last position of j-th interval (S_0 := beginning, equilibrium)
    # First let's compute E[T|S_0], Var[T|S_0]
    # This is computed by computing E[Y_m|S_{m-1}], Var[Y_m|S_{m-1}]
    # Then E[Y_{m-1} + Y_{m}|S_{m-2}], Var[Y_{m-1}+Y_{m}|S_{m-2}], etc.
    # In order to compute them, we need:
    # E[Y_m|S_{m-1}], Var[Y_m|S_{m-1}]
    # E[Y_j|S_{j-1}, S_j], Var[Y_j|S_{j-1}, S_j] for 0 <= j <= m-2
    # Pr[S_j | S_{j-1}] for 1 <= j <= m-1
    # we also need equilibrium probs of the first Markov chain for Pr[S_0]

    # get context labels by amounts
    gap_labels, interval_labels = _compute_gap_interval_ctx_labels(rs, ctx)
    logger.debug(f"gap labels: {gap_labels}")
    logger.debug(f"interval labels: {interval_labels}")

    # precompute 'em values, and then run the old code
    inter_e, inter_v, trans = _compute_elementary_values(gap_labels,
                                                         interval_labels,
                                                         markov_chains)
    logger.debug(f"{inter_e=}")
    logger.debug(f"{inter_v=}")
    logger.debug(f"{trans=}")
    last_e = np.zeros(2, dtype=np.longdouble)
    last_v = np.zeros(2, dtype=np.longdouble)
    for s0 in range(2):
        for st in range(2):
            last_e[s0] += inter_e[-1][s0][st] * trans[-1][s0][st]
            last_v[s0] += inter_v[-1][s0][st] * trans[-1][s0][st]
            last_v[s0] += inter_e[-1][s0][st]**2 * trans[-1][s0][st] * (1 - trans[-1][s0][st])
        # indentation is intentional
        last_v[s0] -= 2 * inter_e[-1][s0][0] * inter_e[-1][s0][1] * trans[-1][s0][0] * trans[-1][s0][1]

    equilibrium = markov_chains[ctx[0][0]].iprobs

    # compute the whole stuff
    result_e = last_e.copy()
    result_v = last_v.copy()
    for j in range(m-2, -1, -1):
        result_e, result_v = _merge_evt_between_intervals((inter_e[j], inter_v[j], trans[j]), (result_e, result_v))

    true_result_e = 0
    true_result_v = 0
    for s0 in range(2):
        true_result_e += result_e[s0] * equilibrium[s0]
        true_result_v += result_v[s0] * equilibrium[s0]
        true_result_v += result_e[s0]**2 * equilibrium[s0] * (1 - equilibrium[s0])
    # indentation is intentional
    true_result_v -= 2 * result_e[0] * result_e[1] * equilibrium[0] * equilibrium[1]

    return true_result_e, true_result_v


def _compute_gap_interval_ctx_labels(rs, ctx):
    gap_labels = []
    interval_labels = []

    class Event(IntEnum):
        REND = 0
        CCHANGE = 1
        RBEGIN = 2

    events = []
    for b, e in rs:
        events.append((b, Event.RBEGIN, None))
        events.append((e, Event.REND, None))
    for ctx_label, b, e in ctx:
        events.append((b, Event.CCHANGE, ctx_label))
    events = sorted(events)

    last_pos = 0
    accum = []
    last_ctx_label = None
    for pos, ev, ctx_label in events:
        match ev:
            case Event.RBEGIN:
                if last_ctx_label is not None:
                    if pos != last_pos:
                        accum.append((last_ctx_label, pos - last_pos))
                    gap_labels.append(accum)
                accum = []
            case Event.REND:
                if last_ctx_label is not None:
                    if pos != last_pos:
                        accum.append((last_ctx_label, pos - last_pos))
                    interval_labels.append(accum)
                accum = []
            case Event.CCHANGE:
                if last_ctx_label is not None:
                    if pos != last_pos:
                        accum.append((last_ctx_label, pos - last_pos))
                last_ctx_label = ctx_label
        last_pos = pos
    return gap_labels, interval_labels


def _compute_elementary_values(gap_labels, interval_labels, markov_chains):
    """Compute conditional mean and variances for Y_j for
    the computation of mean and variance of
    the number of shared bases between annotations.

    :param gap_labels: sequence of pairs (ctx_label, length),
        denoting the context labels in gap before j-th interval
    :param interval_labels: sequence of pairs (ctx_label, length),
        denoting the context label inside j-th interval
    :param markov_chains: dictionary of markov chains for each context label
    :return: tuple of `(inter_e, inter_v, trans)`,
        where inter_e[j][a][b] = E[Y_j | S_{j-1}=a, S_{j}=b],
        inter_v[j][a][b] = Var[Y_j | S_{j-1}=a, S_{j}=b],
        trans[j][a][b] = Pr[S_j = b | S_{j-1} = a]
    """
    logger = logging.getLogger("mcdp2.shared_bases._compute_elementary_values")
    logger.debug(f"Arguments: ({gap_labels}, {interval_labels}, {markov_chains})")
    m = len(gap_labels)
    inter_e = np.zeros((m, 2, 2), dtype=np.longdouble)
    inter_v = np.zeros((m, 2, 2), dtype=np.longdouble)
    trans = np.zeros((m, 2, 2), dtype=np.longdouble)

    for j in range(m):
        logger.debug(f"Processing {j}-th interval...")
        gap_transition_matrix = np.identity(2, dtype=np.longdouble)
        for ctx_label, length in gap_labels[j]:
            gap_transition_matrix = gap_transition_matrix @ matrix_pow(markov_chains[ctx_label].tprobs, length)
        logger.debug(f"{gap_transition_matrix=}")
        l = len(interval_labels[j])

        subinter_e = np.zeros((l, 2, 2), dtype=np.longdouble)
        subinter_v = np.zeros((l, 2, 2), dtype=np.longdouble)
        subtrans = np.zeros((l, 2, 2), dtype=np.longdouble)
        for i, (ctx_label, length) in enumerate(interval_labels[j]):
            transition_matrix = markov_chains[ctx_label].tprobs
            e, v, t = _compute_elementary_evt(transition_matrix, length)
            subinter_e[i] = e
            subinter_v[i] = v
            subtrans[i] = t
        inter_e[j] = subinter_e[0]
        inter_v[j] = subinter_v[0]
        trans[j] = subtrans[0]
        for i in range(1, l):
            inter_e[j], inter_v[j], trans[j] = _merge_evt_inside_interval(
                (inter_e[j], inter_v[j], trans[j]), (subinter_e[i], subinter_v[i], subtrans[i])
            )

        # it is still not done yet. We need to incorporate the gap matrices
        # E[Y_j | S_{j-1}=a, S_j=b] = \sum_{h=0}^1 E[Y_j|S_{j-1}=a, S_j=b, H = h] * Pr[H = h|S_{j-1}=a, S_j=b] =
        # =  \sum_{h=0}^1 E[Y_j | H = h, S_j=b] * Pr[H = h, S_j=b|S_{j-1}=a] / Pr[S_j=b|S_{j-1}=a])
        true_inter_e = np.zeros((2, 2), dtype=np.longdouble)
        true_inter_v = np.zeros((2, 2), dtype=np.longdouble)
        th = np.zeros((2, 2, 2), dtype=np.longdouble)
        for s0 in range(2):
            for st in range(2):
                for sh in range(2):
                    th[sh][s0][st] = gap_transition_matrix[s0][sh] * trans[j][sh][st] / \
                                        (gap_transition_matrix[s0][0] * trans[j][0][st]
                                         + gap_transition_matrix[s0][1] * trans[j][1][st])
        for s0 in range(2):
            for st in range(2):
                for sh in range(2):
                    true_inter_e[s0][st] += inter_e[j][sh][st] * th[sh][s0][st]
                    true_inter_v[s0][st] += inter_v[j][sh][st] * th[sh][s0][st]
                    true_inter_v[s0][st] += inter_e[j][sh][st] ** 2 * th[sh][s0][st] * (1 - th[sh][s0][st])
                true_inter_v[s0][st] -= 2 * inter_e[j][0][st] * inter_e[j][1][st] * th[0][s0][st] * th[1][s0][st]
        true_trans = gap_transition_matrix @ trans[j]
        inter_e[j], inter_v[j], trans[j] = true_inter_e, true_inter_v, true_trans
    return inter_e, inter_v, trans


def _compute_elementary_evt(x, length):
    if length == 0:
        return None
    elif length == 1:
        # inter_e[j][a][b] = E[Y_j | S_{j-1}=a, S_{j}=b],
        # inter_v[j][a][b] = Var[Y_j | S_{j-1}=a, S_{j}=b],
        # trans[j][a][b] = Pr[S_j = b | S_{j-1} = a]
        t = x.copy()
        e = np.array([[0, 1], [0, 1]], dtype=np.longdouble)
        v = np.array([[0, 0], [0, 0]], dtype=np.longdouble)  # intended
        return e, v, t
    elif length > 1:
        evt_half = _compute_elementary_evt(x, length // 2)
        evt_new = _merge_evt_inside_interval(evt_half, evt_half)
        if length % 2 == 1:
            evt_new = _merge_evt_inside_interval(evt_new, _compute_elementary_evt(x, 1))
        return evt_new
    else:
        raise ValueError(f"m cannot be negative! Got {length} instead!")


def _merge_evt_inside_interval(evt1, evt2):
    e1, v1, t1 = evt1
    e2, v2, t2 = evt2
    e = np.array([[0, 0], [0, 0]], dtype=np.longdouble)
    v = np.array([[0, 0], [0, 0]], dtype=np.longdouble)
    t = np.array([[0, 0], [0, 0]], dtype=np.longdouble)

    # Pr[S_h | S_0, S_t]
    th = np.zeros((2, 2, 2), dtype=np.longdouble)

    for s0 in range(2):
        for st in range(2):
            for sh in range(2):
                th[sh][s0][st] = t1[s0][sh] * t2[sh][st] / \
                                 (t1[s0][0] * t2[0][st] + t1[s0][1] * t2[1][st])

    for s0 in range(2):
        for st in range(2):
            for sh in range(2):
                e[s0][st] += (e1[s0][sh] + e2[sh][st]) * th[sh][s0][st]
                v[s0][st] += (v1[s0][sh] + v2[sh][st]) * th[sh][s0][st]
                v[s0][st] += (e1[s0][sh] + e2[sh][st]) ** 2 \
                             * th[sh][s0][st] * (1 - th[sh][s0][st])
                t[s0][st] += t1[s0][sh] * t2[sh][st]
            # sic! the indent is intentional
            v[s0][st] -= 2 * (e1[s0][0] + e2[0][st]) * (e1[s0][1] + e2[1][st]) \
                         * th[0][s0][st] * th[1][s0][st]
    return e, v, t


def _merge_evt_between_intervals(evt_head, ev_tail):
    e1, v1, t1 = evt_head
    e2, v2 = ev_tail
    e = np.array([0, 0], dtype=np.longdouble)
    v = np.array([0, 0], dtype=np.longdouble)

    for s0 in range(2):
        for st in range(2):
            e[s0] += (e1[s0][st] + e2[st]) * t1[s0][st]
            v[s0] += (v1[s0][st] + v2[st]) * t1[s0][st]
            v[s0] += (e1[s0][st] + e2[st])**2 * t1[s0][st] * (1 - t1[s0][st])
        # indentation is intentional
        v[s0] -= 2 * (e1[s0][0] + e2[0]) * (e1[s0][1] + e2[1]) \
                         * t1[s0][0] * t1[s0][1]

    return e, v


def shared_bases_approx_sampled(rs, ctx, markov_chains, tries=100):
    sampled_values = [sample_number_of_shared_bases(rs, ctx, markov_chains) for _ in range(tries)]
    mean = statistics.mean(sampled_values)
    variance = statistics.variance(sampled_values)  # return sampled variance
    return mean, variance


def sample_number_of_shared_bases(rs, ctx, markov_chains):
    if len(rs) == 0:
        return 0
    sequence = []
    initial_weights = [markov_chains[ctx[0][0]].iprobs[i] for i in range(2)]
    prev_state = random.choices([0, 1], weights=initial_weights, k=1)[0]
    for ctx_label, b, e in ctx:
        for _ in range(e - b):
            weights = [markov_chains[ctx_label].tprobs[prev_state][i] for i in range(2)]
            new_state = random.choices([0, 1], weights=weights, k=1)[0]
            sequence.append(new_state)
            prev_state = new_state
    number_of_shared_bases = 0
    for b, e in rs:
        number_of_shared_bases += sum(sequence[b:e])
    return number_of_shared_bases
