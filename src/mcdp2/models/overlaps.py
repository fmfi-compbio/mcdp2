import logging
from typing import Literal

from copy import deepcopy

import numpy as np
import scipy.stats

from mcdp2.common.helpers import split_intervals_by_chrname, count_overlaps_chr, matrix_pow, compute_mean_from_pmf, \
    compute_sd_from_pmf, compute_z_score, compute_joint_pmf
from mcdp2.models.common import _count_transitions, map_parallel, MC

ALGORITHM = Literal['exact', 'fast', 'both']


def compute_pvalue(reference,
                   query,
                   chromosome_lengths,  # technically this is redundant for now,
                   # since we have the context
                   context,
                   threads: int = 4,
                   algorithm: ALGORITHM = 'exact'):
    """Compute overlap statistics distribution for context-dependent two-state Markov chain"""
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

    # compute number of overlaps for each chromosome
    overlaps_fn = lambda x: count_overlaps_chr(x[0], x[1])
    overlap_counts = map_parallel(overlaps_fn, inputs, threads)

    # compute overlap markov chains for each interval
    mc_fn = lambda x: _update_iprobs_cumulative(
            _compute_overlap_mcs(
                ref_intervals=x[0],
                context_intervals=x[2],
                markov_chains=markov_chains
            )
        )
    overlap_mcs = map_parallel(mc_fn, inputs, threads)
    total_overlap_count = sum(overlap_counts)

    result = {
        "total": {
            "reference_size": len(reference),
            "query_size": len(query),
            "overlap_count": int(total_overlap_count),
        },
        "by_chromosomes": {
            name: {
                "reference_size": len(inputs[i][0]),
                "query_size": len(inputs[i][1]),
                "overlap_count": int(overlap_counts[i]),
            }
            for i, name in enumerate(chromosome_names)
        },
    }

    if algorithm == 'fast' or algorithm == 'both':
        # compute CLT approximation for each chromosome
        clt_approxs = map_parallel(normal_approx, overlap_mcs, threads)
        clt_approx_zscores = [compute_z_score(value=overlap_counts[i],
                                              mean=clt_approxs[i][0],
                                              sd=clt_approxs[i][1])
                              for i in range(len(chromosome_names))]
        clt_approx_pvalues = [scipy.stats.norm.sf(overlap_counts[i],
                                                  clt_approxs[i][0],
                                                  clt_approxs[i][1])
                              if clt_approxs[i][1] > 1e-300 else 1
                              for i in range(len(chromosome_names))]

        # merge CLT approximations for the whole genome
        total_clt_approx = (
            sum(m for m, _ in clt_approxs), np.sqrt(np.sum(s ** 2 for _, s in clt_approxs)))
        total_clt_approx_zscore = compute_z_score(total_overlap_count, *total_clt_approx)
        total_clt_approx_pvalue = scipy.stats.norm.sf(total_overlap_count,
                                                      total_clt_approx[0],
                                                      total_clt_approx[1]) \
            if total_clt_approx[1] > 1e-300 else 1
        result['total'] |= {"clt_mean": float(total_clt_approx[0]),
                            "clt_sd": float(total_clt_approx[1]),
                            "clt_pvalue": float(total_clt_approx_pvalue),
                            "clt_zscore": float(total_clt_approx_zscore), }
        for i, name in enumerate(chromosome_names):
            result['by_chromosomes'][name] |= {"clt_mean": float(clt_approxs[i][0]),
                                               "clt_sd": float(clt_approxs[i][1]),
                                               "clt_pvalue": float(clt_approx_pvalues[i]),
                                               "clt_zscore": float(clt_approx_zscores[i]), }
    if algorithm == 'exact' or algorithm == 'both':
        # compute full DP pmfs
        full_dp_pmfs = map_parallel(_compute_pmf_from_mcs_dp, overlap_mcs, threads)
        full_dp_normal_approxs = [(compute_mean_from_pmf(pmf), compute_sd_from_pmf(pmf))
                                  for pmf in full_dp_pmfs]
        full_dp_pvalues = [sum(pmf[value:])
                           for value, pmf in zip(overlap_counts, full_dp_pmfs)]
        full_dp_zscores = [compute_z_score(value, *params)
                           for value, params in zip(overlap_counts, full_dp_normal_approxs)]

        # merge full DP pmfs for the whole genome
        total_full_dp_pmf = compute_joint_pmf(full_dp_pmfs)
        total_full_dp_normal_approx = (compute_mean_from_pmf(total_full_dp_pmf),
                                       compute_sd_from_pmf(total_full_dp_pmf))
        # if the statistics is more than mean, then compute p-value by summing the right tail
        # otherwise, compute the left tail and subtract it from 1
        if total_overlap_count >= total_full_dp_normal_approx[0]:
            total_full_dp_pvalue = sum(total_full_dp_pmf[total_overlap_count:])
        else:
            total_full_dp_pvalue = 1 - sum(total_full_dp_pmf[:total_overlap_count])
        total_full_dp_zscore = compute_z_score(total_overlap_count, *total_full_dp_normal_approx)
        result['total'] |= {"full_dp_mean": float(total_full_dp_normal_approx[0]),
                            "full_dp_sd": float(total_full_dp_normal_approx[1]),
                            "full_dp_pvalue": float(total_full_dp_pvalue),
                            "full_dp_zscore": float(total_full_dp_zscore), }
        for i, name in enumerate(chromosome_names):
            result['by_chromosomes'][name] |= {"full_dp_mean": float(full_dp_normal_approxs[i][0]),
                                               "full_dp_sd": float(full_dp_normal_approxs[i][1]),
                                               "full_dp_pvalue": float(full_dp_pvalues[i]),
                                               "full_dp_zscore": float(full_dp_zscores[i]), }
        result["_full_dp_pmfs"] = {
            "total": list(map(float, total_full_dp_pmf)),
            "by_chromosomes": {
                name: list(map(float, full_dp_pmfs[i]))
                for i, name in enumerate(chromosome_names)
            }
        }

    return result


def _compute_pmf_from_mcs_dp(mcs):
    """The original DP algorithm."""
    if len(mcs) == 0:
        return [1]

    m = len(mcs)
    prev_line = np.array([[0, 0] for _ in range(m + 1)], dtype=np.longdouble)
    next_line = np.array([[0, 0] for _ in range(m + 1)], dtype=np.longdouble)
    last_column = np.array([[0, 0] for _ in range(m + 1)], dtype=np.longdouble)

    # computing the 0th line, i.e. DP[j=*, k=0]
    prev_line[0] = mcs[0][1].iprobs
    for j in range(1, m + 1):
        prev_line[j] = prev_line[j - 1] @ mcs[j - 1][0].tprobs
    last_column[0] = prev_line[-1].copy()

    for kappa in range(1, m + 1):
        for j in range(kappa, m + 1):
            next_line[j] = next_line[j - 1] @ mcs[j - 1][0].tprobs + \
                           prev_line[j - 1] @ (mcs[j - 1][1].tprobs - mcs[j - 1][0].tprobs)
        last_column[kappa] = next_line[-1].copy()
        prev_line = next_line
        next_line = np.array([[0, 0] for _ in range(m + 1)], dtype=np.longdouble)

    result = np.array([np.sum(last_column[j]) for j in range(m + 1)], dtype=np.longdouble)
    return result


def _compute_overlap_mcs(ref_intervals, context_intervals, markov_chains):
    CTX, R = 0, 1
    BEGIN, END = 0, 1
    if len(ref_intervals) == 0:
        return []

    events = []
    for b, e in ref_intervals:
        events.append((b, R, BEGIN))
        events.append((e, R, END))

    for ctx, b, e in context_intervals:
        events.append((b, CTX, ctx))
    events = sorted(events)

    result = []
    last_pos = None
    is_inside_interval = False
    current_iprobs = None
    current_nohit_matrix = np.identity(2, dtype=np.longdouble)
    current_any_matrix = np.identity(2, dtype=np.longdouble)
    current_ctx = None
    for pos, t, x in events:
        if current_ctx is not None:
            local_any = markov_chains[current_ctx].tprobs
            local_zero = np.array([[local_any[0, 0], 0.0], [local_any[1, 0], 0.0]],
                                  dtype=np.longdouble)
            dist = pos - last_pos
            any_t = matrix_pow(local_any, dist)
            zero_t = matrix_pow(local_zero, dist)

            if is_inside_interval:
                current_nohit_matrix = current_nohit_matrix @ zero_t
                current_any_matrix = current_any_matrix @ any_t
            else:
                current_nohit_matrix = current_nohit_matrix @ any_t  # sic!
                current_any_matrix = current_any_matrix @ any_t

        if t == CTX:
            if current_ctx is None:
                current_iprobs = markov_chains[x].iprobs
            current_ctx = x
        else:
            if x == BEGIN:
                is_inside_interval = True
            else:
                result.append((MC(current_nohit_matrix, current_iprobs),
                               MC(current_any_matrix, current_iprobs)))
                is_inside_interval = False
                current_nohit_matrix = np.identity(2, dtype=np.longdouble)
                current_any_matrix = np.identity(2, dtype=np.longdouble)
                current_iprobs = markov_chains[current_ctx].iprobs
        last_pos = pos
    return result


def _update_iprobs_cumulative(mcs):
    if len(mcs) == 0:
        return []
    result = [mcs[0]]
    current_iprobs = mcs[0][0].iprobs @ mcs[0][1].tprobs

    for mc_zero, mc_any in mcs[1:]:
        new_pair = (deepcopy(mc_zero), deepcopy(mc_any))
        new_pair[0].iprobs = current_iprobs
        new_pair[1].iprobs = current_iprobs
        result.append(new_pair)
        current_iprobs = current_iprobs @ mc_any.tprobs
    return result


def normal_approx_upper_limit(overlap_mcs):
    """Sum of standard deviations is an upper bound for the standard deviation of a sum"""
    mu_total = 0
    sd_sum = 0
    for mc_zero, _ in overlap_mcs:
        p_0 = sum(mc_zero.iprobs @ mc_zero.tprobs)
        mu = 1 - p_0
        sd = np.sqrt(p_0 * (1 - p_0))
        mu_total += mu
        sd_sum += sd
    return mu_total, sd_sum


def normal_approx_upper_bound(overlap_mcs):
    if len(overlap_mcs) == 0:
        return 0, 0
    if len(overlap_mcs) == 1:
        p_0 = sum(overlap_mcs[0][0].iprobs @ overlap_mcs[0][0].tprobs)
        mu = 1 - p_0
        sd = np.sqrt(p_0 * (1 - p_0))
        return mu, sd
    mus = [1 - sum(mc_zero.iprobs @ mc_zero.tprobs) for mc_zero, _ in overlap_mcs]
    sds = [np.sqrt(p * (1 - p)) for p in mus]
    # let X = [i-th interval is overlapped], Y = [(i+1)-th interval is overlapped]
    # E(XY) = Pr[X = 1, Y = 1]
    e_xys = [sum(m1[1].iprobs @ (m1[1].tprobs - m1[0].tprobs) @ (m2[1].tprobs - m2[0].tprobs))
             for m1, m2 in zip(overlap_mcs[:-1], overlap_mcs[1:])]
    # Cov(X, Y) = E(XY) - E(X)E(Y)
    covs = [max(0, e_xy - mus[i] * mus[i + 1]) for i, e_xy in enumerate(e_xys)]
    # Corr(X, Y) = Cov(X, Y) / [ SD(X) * SD(Y) ]
    corrs = [cov / sds[i] / sds[i + 1] for i, cov in enumerate(covs)]

    total_variance = 0
    left_sds = 0
    right_sds = sum(sds)
    for i in range(len(overlap_mcs)):
        total_variance += sds[i] ** 2
        left_corr = corrs[i - 1] if i > 0 else 0
        right_corr = corrs[i] if i < len(overlap_mcs) - 1 else 0
        total_variance += left_corr * left_sds * sds[i]
        left_sds += sds[i]
        right_sds -= sds[i]
        total_variance += right_corr * right_sds * sds[i]
    total_mu = sum(mus)
    total_sd = np.sqrt(total_variance)
    return total_mu, total_sd


def normal_approx(overlap_mcs):
    logger = logging.getLogger('mcdp2.overlaps')
    if len(overlap_mcs) == 0:
        return 0, 0
    # if len(overlap_mcs) == 1:
    #     p_0 = sum(overlap_mcs[0][0].iprobs @ overlap_mcs[0][0].tprobs)
    #     mu = 1 - p_0
    #     sd = np.sqrt(p_0 * (1 - p_0))
    #     return mu, sd

    m = len(overlap_mcs)
    logger.debug(f'{m=}')
    # head = I_i, tail = I_{i+1} + ... I_m
    # E[tail|s_next]
    X0, XA = overlap_mcs[-1][0].tprobs, overlap_mcs[-1][1].tprobs
    logger.debug(f'X0_{m}={X0}')
    logger.debug(f'XA_{m}={XA}')

    et_0 = 1 - np.sum(np.array([1, 0], dtype=np.longdouble) @ X0)
    et_1 = 1 - np.sum(np.array([0, 1], dtype=np.longdouble) @ X0)
    logger.debug(f'E[I_{m}|s_prev=0]={et_0}, E[I_{m}|S_prev=1]={et_1}')
    # Var[tail|s_next]
    vt_0 = et_0 * (1 - et_0)
    vt_1 = et_1 * (1 - et_1)
    logger.debug(f'Var[I_{m}|s_prev=0]={vt_0}, Var[I_{m}|S_prev=1]={vt_1}')

    for i in range(m - 2, -1, -1):
        logger.debug(f"Adding I_{i + 1}...")
        X0, XA = overlap_mcs[i][0].tprobs, overlap_mcs[i][1].tprobs
        logger.debug(f"X0_{i + 1}={X0}")
        logger.debug(f"XA_{i + 1}={XA}")
        # Pr[s_next|s_prev]
        p00, p10 = np.array([1, 0], dtype=np.longdouble) @ XA
        p01, p11 = np.array([0, 1], dtype=np.longdouble) @ XA
        assert 0 <= p00 <= 1
        assert 0 <= p10 <= 1
        assert abs(p00 + p10 - 1) < 1e-6
        assert 0 <= p01 <= 1
        assert 0 <= p11 <= 1
        assert abs(p01 + p11 - 1) < 1e-6
        logger.debug(f"Pr[s_next=0|s_prev=0]={p00}")
        logger.debug(f"Pr[s_next=1|s_prev=0]={p10}")
        logger.debug(f"Pr[s_next=0|s_prev=1]={p01}")
        logger.debug(f"Pr[s_next=1|s_prev=1]={p11}")
        # E[head|s_prev, s_next] =
        # = Pr[head = 1 | s_prev, s_next] =
        # = Pr[head = 1 and s_next | s_prev] / Pr[s_next | s_prev]
        r_0 = np.array([1, 0], dtype=np.longdouble) @ (XA - X0)
        eh_00 = r_0[0] / p00
        eh_01 = r_0[1] / p10
        r_1 = np.array([0, 1], dtype=np.longdouble) @ (XA - X0)
        eh_10 = r_1[0] / p01
        eh_11 = r_1[1] / p11
        assert 0 <= eh_00 <= 1
        assert 0 <= eh_01 <= 1
        assert 0 <= eh_10 <= 1
        assert 0 <= eh_11 <= 1
        logger.debug(f"E[I_{i + 1}|s_prev=0, s_next=0]={eh_00}")
        logger.debug(f"E[I_{i + 1}|s_prev=0, s_next=1]={eh_01}")
        logger.debug(f"E[I_{i + 1}|s_prev=1, s_next=0]={eh_10}")
        logger.debug(f"E[I_{i + 1}|s_prev=1, s_next=1]={eh_11}")

        # Var[head|s_prev, s_next]
        vh_00 = eh_00 * (1 - eh_00)
        vh_01 = eh_01 * (1 - eh_01)
        vh_10 = eh_10 * (1 - eh_10)
        vh_11 = eh_11 * (1 - eh_11)

        logger.debug(f"Var[I_{i + 1}|s_prev=0, s_next=0]={vh_00}")
        logger.debug(f"Var[I_{i + 1}|s_prev=0, s_next=1]={vh_01}")
        logger.debug(f"Var[I_{i + 1}|s_prev=1, s_next=0]={vh_10}")
        logger.debug(f"Var[I_{i + 1}|s_prev=1, s_next=1]={vh_11}")

        # E[head+tail|s_prev] =
        # = sum_{s_next} (E[head|s_prev, s_next] + E[tail|s_next]) * Pr[s_next | s_prev]
        eht_0 = (eh_00 + et_0) * p00 + (eh_01 + et_1) * p10
        eht_1 = (eh_10 + et_0) * p01 + (eh_11 + et_1) * p11
        logger.debug(f"E[I_{i + 1}+...+I_{m}|s_prev=0]={eht_0}")
        logger.debug(f"E[I_{i + 1}+...+I_{m}|s_prev=1]={eht_1}")

        # C1|s_prev = sum_{s_next} (Var[h|s_prev,s_next] + Var[t|s_next]) * Pr[s_next | s_prev]
        c1_0 = (vh_00 + vt_0) * p00 + (vh_01 + vt_1) * p10
        c1_1 = (vh_10 + vt_0) * p01 + (vh_11 + vt_1) * p11
        # C2|s_prev = sum_{s_next} (E[h|s_prev,s_next]+E[t|s_next])^2
        # * Pr[s_next|s_prev] * (1 - Pr[s_next|s_prev])
        c2_0 = (eh_00 + et_0) ** 2 * p00 * (1 - p00) + (eh_01 + et_1) ** 2 * p10 * (1 - p10)
        c2_1 = (eh_10 + et_0) ** 2 * p01 * (1 - p01) + (eh_11 + et_1) ** 2 * p11 * (1 - p11)
        # C3|s_prev = -2
        # * (E[h|s_prev,s_next=0]+E[t|s_next=0]) * (E[h|s_prev,s_next=1]+E[t|s_next=1])
        # * Pr[s_next=0|s_prev] * Pr[s_next=1|s_prev]
        c3_0 = -2 * (eh_00 + et_0) * p00 * (eh_01 + et_1) * p10
        c3_1 = -2 * (eh_10 + et_0) * p01 * (eh_11 + et_1) * p11

        # Var[head+tail|s_prev]
        vht_0 = c1_0 + c2_0 + c3_0
        vht_1 = c1_1 + c2_1 + c3_1
        logger.debug(f"Var[I_{i + 1}+...+I_{m}|s_prev=0]={vht_0}")
        logger.debug(f"Var[I_{i + 1}+...+I_{m}|s_prev=1]={vht_1}")

        # next iteration preparation
        et_0, et_1 = eht_0, eht_1
        vt_0, vt_1 = vht_0, vht_1

    i_0, i_1 = overlap_mcs[0][0].iprobs
    e_total = et_0 * i_0 + et_1 * i_1
    v_total = vt_0 * i_0 + vt_1 * i_1 + \
              et_0 ** 2 * i_0 * (1 - i_0) + et_1 ** 2 * i_1 * (1 - i_1) - \
              2 * et_0 * et_1 * i_0 * i_1
    sd_total = np.sqrt(v_total)

    return e_total, sd_total
