import warnings

import numpy as np
import pytest

from mcdp2.models.overlaps import compute_pvalue


def direct_exp(a, n):
    if n < 0:
        raise ValueError(f"Power should be non-negative, got {n=}!")
    if n == 0:
        return np.identity(a.shape[0], dtype=np.longdouble)
    if n == 1:
        return a
    if n % 2 == 0:
        return direct_exp(a.dot(a), n // 2)
    else:
        return a.dot(direct_exp(a.dot(a), n // 2))


class TDExp:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.T = np.array([[x, 1 - x], [1 - y, y]], dtype=np.longdouble)
        self.D = np.array([[x, 0], [1 - y, 0]], dtype=np.longdouble)
        try:
            self.Q = np.array([[1, (x - 1) / (1 - y)], [1, 1]], dtype=np.longdouble)
            quotient = -2 + x + y
            self.Q_inv = np.array([[(-1 + y) / quotient, (-1 + x) / quotient],
                                   [(1 - y) / quotient, (-1 + y) / quotient]],
                                  dtype=np.longdouble)
        except ZeroDivisionError as e:
            self.exp_t = lambda a: direct_exp(self.T, a)

        self.logx = np.log(x)

    def exp_t(self, a):
        if a < 0:
            raise ValueError(f"Exponent must be nonnegative, got '{a}' instead!")
        elif a == 0:
            return np.identity(2, dtype=np.longdouble)
        elif a == 1:
            return self.T.copy()
        # elif self.y == 1 or self.x == 1 or self.x + self.y == 2:
        #     return direct_exp(self.T, a)
        elif a < 100:
            return direct_exp(self.T, a)
        else:
            l = np.array([[1, 0], [0, np.power(self.x + self.y - 1, a)]], dtype=np.longdouble)
            result = self.Q.dot(l.dot(self.Q_inv))
            return result

    def exp_d(self, a):
        if a < 0:
            raise ValueError(f"Exponent must be nonnegative, got '{a}' instead!")
        elif a == 0:
            return np.identity(2, dtype=np.longdouble)
        elif a == 1:
            return self.D.copy()
        elif a < 50:
            return direct_exp(self.D, a)
        else:
            result = np.array([[np.exp(a * self.logx), 0],
                               [(1 - self.y) * np.exp((a - 1) * self.logx), 0]],
                              dtype=np.longdouble)
            return result


def estimate_mc_weights_simple(chr_size, q):
    if len(q) == 0:
        raise ValueError(f"Query interval set should be non-empty!")
    total_intervals_length = sum(e - b for b, e in q)
    total_gaps_length = chr_size - total_intervals_length
    n = len(q)
    alpha = - n - 1 + total_gaps_length
    beta = -n + total_intervals_length
    x = alpha / (alpha + n)
    y = beta / (beta + n)
    return x, y


def old_pmf_computing(r, q, chr_size):
    if len(q) == 0 or len(r) == 0:
        return [0]

    x, y = estimate_mc_weights_simple(chr_size, q)
    E = TDExp(x, y)

    m = len(r)
    if r[0][0] == 0:
        warnings.warn("First reference interval starts with zero, changing to one!")
        r[0] = (1, r[0][1])
        if r[0][1] - r[0][0] == 0:
            warnings.warn("First reference interval has length 0, removing it!")
            r = r[1:]

    r_augmented = [(-np.inf, 0)] + r + [(chr_size, np.inf)]
    prev_line = np.array([[0, 0] for _ in range(m + 1)],
                         dtype=np.longdouble)
    prev_line[0, 0] = 1
    last_col = np.array([[0, 0] for _ in range(m + 1)],
                        dtype=np.longdouble)

    # zero layer of DP P[*, 0, *] should be calculated in a separate way
    for j in range(1, m + 1):
        g = r_augmented[j][0] - r_augmented[j - 1][1]
        if j == 1:
            g -= 1
        assert g >= 0, f"Expected non-negative `g`, got {g=} instead!"
        l = r_augmented[j][1] - r_augmented[j][0]
        assert l >= 0
        prev_line[j] = prev_line[j - 1].dot(E.exp_t(g).dot(E.exp_d(l)))
    last_col[0] = prev_line[-1].copy()

    next_line = np.array([[0, 0] for _ in range(m + 1)], dtype=np.longdouble)
    for k in range(1, m + 1):
        next_line[k - 1] = [0, 0]
        for j in range(k, m + 1):
            g = r_augmented[j][0] - r_augmented[j - 1][1]  # gap length
            if j == 1:
                g -= 1
            assert g >= 0
            l = r_augmented[j][1] - r_augmented[j][0]  # interval length
            assert l >= 0
            # dont_hit = P[j-1, k] * T^g * D^l
            dont_hit = next_line[j - 1].dot(E.exp_t(g).dot(E.exp_d(l)))
            # hit = P[j-1, k-1] * T^g * (T^l - D^l)
            hit = prev_line[j - 1].dot(E.exp_t(g)).dot(
                E.exp_t(l) - E.exp_d(l))
            # P[j, k] = dont_hit + hit
            next_line[j] = dont_hit + hit
        last_col[k, 0] = next_line[-1, 0]
        last_col[k, 1] = next_line[-1, 1]
        for j in range(m + 1):
            prev_line[j, 0] = next_line[j, 0]
            prev_line[j, 1] = next_line[j, 1]
    probs = [np.log(np.sum(last_col[k, :])) for k in range(m + 1)]
    return probs


@pytest.mark.skip(reason="We already see that the old implementation is different.")
@pytest.mark.parametrize("r,q,chr_size", [
    ([], [], 10),
     ([(100001, 100002), (100004, 100005)], [(100001, 100002)], 100010),
])
def test_compare_pmfs(r, q, chr_size):
    r_full = [('a', b, e) for b, e in r]
    q_full = [('a', b, e) for b, e in q]
    chr_lengths = [('a', chr_size)]
    context = {'a': [('C1', 0, chr_size)]}
    new_result = compute_pvalue(r_full, q_full, chr_lengths, context, threads=1)['_full_dp_pmfs']['total']
    old_result = [np.exp(p) for p in old_pmf_computing(r, q, chr_size)]
    assert new_result == pytest.approx(old_result, rel=1e-3)
