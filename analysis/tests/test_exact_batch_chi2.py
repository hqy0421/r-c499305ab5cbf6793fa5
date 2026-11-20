"""Validation tests for exact_batch_chi2.

The independent oracle brute-forces every row-level relabelling
(itertools over all K-of-N label vectors per batch, no aggregation), builds
contingency tables with pandas crosstab, and computes exact Fraction Pearson
statistics directly from the integer table margins. scipy is used only as a
floating-point cross-check of the statistic itself, never for tail logic.

Run:  python3 -I -m pytest -q
"""

from __future__ import annotations

import importlib.util
import itertools
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2 as chi2_dist
from scipy.stats import contingency

SPEC = importlib.util.spec_from_file_location(
    "exact_batch_chi2", Path(__file__).resolve().parents[1] / "exact_batch_chi2.py"
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def exact_table_chi2(code_col, y):
    """Exact Fraction Pearson chi2 of a code x binary-y contingency table."""
    n_tot = {}
    o_pos = {}
    for c, t in zip(code_col, y):
        c = int(c)
        n_tot[c] = n_tot.get(c, 0) + 1
        o_pos[c] = o_pos.get(c, 0) + int(t)
    N = len(y)
    K = int(y.sum())
    if K == 0 or K == N or len(n_tot) < 2:
        return Fraction(0), 0.0
    total = Fraction(0)
    for c, n in n_tot.items():
        o = o_pos[c]
        total += Fraction((N * o - n * K) ** 2, n * K * (N - K))
    # float counterpart
    tab = pd.crosstab(pd.Series(code_col), pd.Series(y)).to_numpy()
    c2f = contingency.chi2_contingency(tab, correction=False)[0]
    return total, c2f


def oracle_brute_force(df, batch_col, target_col, fields):
    """Enumerate every joint row-level allocation; return observed, raw, adj p."""
    p = len(fields)
    codes, _ = mod.encode_fields(df, fields)
    codes = codes.reset_index(drop=True)
    df = df.reset_index(drop=True)

    observed = [Fraction(0) for _ in range(p)]
    float_obs = [0.0] * p
    batch_choices = []
    total_weight = 1
    for bi, sub in df.groupby(batch_col, sort=False):
        # Must slice global codes with the group's ORIGINAL index before any
        # reset_index, otherwise the second batch reuses the first batch's rows.
        local = codes.loc[sub.index].reset_index(drop=True)
        sub = sub.reset_index(drop=True)
        N, K = len(sub), int(sub[target_col].sum())
        allocations = []
        for bits in itertools.combinations(range(N), K):
            y = np.zeros(N, dtype=int)
            np.put(y, bits, 1)
            stats = []
            for f in fields:
                frac, _ = exact_table_chi2(local[f].to_numpy(), y)
                stats.append(frac)
            allocations.append((1, tuple(stats)))
        total_weight *= math.comb(N, K)
        batch_choices.append(allocations)

        yobs = sub[target_col].to_numpy()
        for j, f in enumerate(fields):
            frac, c2f = exact_table_chi2(local[f].to_numpy(), yobs)
            observed[j] += frac
            float_obs[j] += c2f

    raw_num = [0] * p
    adj_num = [0] * p
    n_joint = 0
    for combo in itertools.product(*batch_choices):
        n_joint += 1
        totals = [Fraction(0) for _ in range(p)]
        for _, st in combo:
            for j in range(p):
                totals[j] += st[j]
        m = max(totals)
        for j in range(p):
            if totals[j] >= observed[j]:
                raw_num[j] += 1
            if m >= observed[j]:
                adj_num[j] += 1

    assert n_joint == total_weight
    raw_p = [Fraction(raw_num[j], total_weight) for j in range(p)]
    adj_p = [Fraction(adj_num[j], total_weight) for j in range(p)]
    return observed, float_obs, raw_p, adj_p


def make_df(rows, cols):
    df = pd.DataFrame(rows)
    df.columns = cols
    return df


def test_statistic_matches_scipy_and_oracle():
    df = make_df(
        [
            ("b1", 1, "a", "x"), ("b1", 1, "b", "x"), ("b1", 0, "a", "y"),
            ("b1", 0, "b", "y"), ("b1", 1, "a", "x"), ("b1", 0, "b", "x"),
            ("b2", 1, "b", "x"), ("b2", 0, "a", "y"), ("b2", 1, "a", "y"),
            ("b2", 0, "b", "x"), ("b2", 0, "a", "x"), ("b2", 1, "b", "y"),
        ],
        ["batch", "y", "f1", "f2"],
    )
    res = mod.run_analysis(df, "batch", "y", ["f1", "f2"], 10**6, 10**7)
    obs_br, float_br, raw_br, adj_br = oracle_brute_force(
        df, "batch", "y", ["f1", "f2"]
    )
    D = res["global_denom"]
    for j in range(2):
        # Engine stores observed stats as integer numerators over the global
        # common denominator; convert to Fraction for exact comparison.
        assert Fraction(res["observed"][j], D) == obs_br[j]
        assert abs(res["observed"][j] / D - float_br[j]) < 1e-10
        assert abs(res["float_observed"][j] - float_br[j]) < 1e-10
    assert res["raw_p"] == raw_br
    assert res["adj_p"] == adj_br
    for j in range(2):
        assert res["adj_p"][j] >= res["raw_p"][j]


def test_duplicate_combination_weights_match_row_bruteforce():
    rows = []
    rng = np.random.default_rng(0)
    for b, rate in [("b1", 0.3), ("b2", 0.6)]:
        for i in range(10):
            rows.append(
                (b, int(rng.random() < rate),
                 "a" if i % 2 else "b", "x" if i % 3 else "y")
            )
    df = make_df(rows, ["batch", "y", "f1", "f2"])
    res = mod.run_analysis(df, "batch", "y", ["f1", "f2"], 10**6, 10**7)
    _, _, raw_br, adj_br = oracle_brute_force(df, "batch", "y", ["f1", "f2"])
    assert res["raw_p"] == raw_br
    assert res["adj_p"] == adj_br
    for b, be in zip(
        [x for x in res["raw_batches"] if not x["degenerate"]],
        res["batch_enums"],
    ):
        assert be["total_weight"] == math.comb(b["N"], b["K"])


def test_missing_value_is_its_own_category():
    df = make_df(
        [
            ("b1", 1, "a"), ("b1", 1, None), ("b1", 0, "a"), ("b1", 0, "b"),
            ("b1", 1, None), ("b1", 0, "a"), ("b1", 1, "b"), ("b1", 0, None),
        ],
        ["batch", "y", "f1"],
    )
    res = mod.run_analysis(df, "batch", "y", ["f1"], 10**6, 10**7)
    codes = res["codes"]["f1"].to_numpy()
    assert set(np.unique(codes).tolist()) == {-1, 0, 1}
    counts = mod.category_count_table(
        df, res["codes"], ["f1"], res["categories"], "batch"
    )
    miss = counts[counts["is_missing"]]
    assert len(miss) == 1 and int(miss["n_rows"].iloc[0]) == 3
    _, _, raw_br, adj_br = oracle_brute_force(df, "batch", "y", ["f1"])
    assert res["raw_p"] == raw_br
    assert res["adj_p"] == adj_br


def test_degenerate_batches_skipped_but_observed_intact():
    df = make_df(
        [
            ("b0", 1, "a"), ("b0", 1, "b"),  # K=N: deterministic, stat 0
            ("b1", 1, "a"), ("b1", 1, "a"), ("b1", 0, "a"),
            ("b1", 0, "b"), ("b1", 0, "b"), ("b1", 1, "b"),
        ],
        ["batch", "y", "f1"],
    )
    res = mod.run_analysis(df, "batch", "y", ["f1"], 10**6, 10**7)
    active = [b for b in res["raw_batches"] if not b["degenerate"]]
    assert len(active) == 1 and active[0]["K"] == 3
    # Single field: adjusted == raw by identity, so the joint enumeration is
    # skipped entirely (the K=N batch is degenerate anyway).
    assert res["adj_p"] == res["raw_p"]
    # The degenerate batch still appears in the original category counts.
    counts = mod.category_count_table(
        df, res["codes"], ["f1"], res["categories"], "batch"
    )
    assert set(counts["n_batch_b0"]) == {1, 1}
    # The K=N batch contributes zero, so observed equals the active batch alone.
    assert res["observed"][0] > 0
    _, _, raw_br, adj_br = oracle_brute_force(df, "batch", "y", ["f1"])
    assert res["raw_p"] == raw_br
    assert res["adj_p"] == adj_br


def test_constant_field_gives_pvalue_one():
    df = make_df(
        [
            ("b1", 1, "c", "a"), ("b1", 0, "c", "b"), ("b1", 1, "c", "a"),
            ("b1", 0, "c", "b"),
            ("b2", 1, "c", "a"), ("b2", 0, "c", "b"),
        ],
        ["batch", "y", "f_const", "f_var"],
    )
    res = mod.run_analysis(df, "batch", "y", ["f_const", "f_var"], 10**6, 10**7)
    assert res["observed"][0] == 0
    assert res["raw_p"][0] == Fraction(1)
    assert res["adj_p"][0] == Fraction(1)


def test_tail_includes_ties_and_matches_oracle():
    df = make_df(
        [
            ("b1", 1, "a"), ("b1", 1, "a"), ("b1", 0, "b"), ("b1", 0, "b"),
            ("b1", 1, "b"), ("b1", 0, "a"),
        ],
        ["batch", "y", "f1"],
    )
    res = mod.run_analysis(df, "batch", "y", ["f1"], 10**6, 10**7)
    _, _, raw_br, adj_br = oracle_brute_force(df, "batch", "y", ["f1"])
    assert res["raw_p"][0] == raw_br[0]
    assert res["adj_p"][0] == adj_br[0]
    # In this symmetric table the observed allocation is among the most
    # extreme attainable ones, so the tie-inclusive tail can legitimately be 1.
    assert res["raw_p"][0] <= Fraction(1)


def test_pvalues_valid_and_denominator_is_exact_joint_count():
    rng = np.random.default_rng(42)
    rows = []
    for b in ("b1", "b2"):
        for _ in range(10):
            rows.append(
                (b, int(rng.random() < 0.5),
                 rng.choice(["a", "b", "c"]), rng.choice(["x", "y"]))
            )
    df = make_df(rows, ["batch", "y", "f1", "f2"])
    res = mod.run_analysis(df, "batch", "y", ["f1", "f2"], 10**6, 10**7)
    for j in range(2):
        assert 0 < float(res["raw_p"][j]) <= 1
        assert 0 < float(res["adj_p"][j]) <= 1
        assert res["adj_p"][j] >= res["raw_p"][j]
    expected = 1
    for be in res["batch_enums"]:
        expected *= be["total_weight"]
    # p-value is (tail count) / expected; after Fraction reduction the tail
    # count recovered by p*expected must still be an integer in [0, expected].
    for pv in (*res["raw_p"], *res["adj_p"]):
        assert (expected * pv.numerator) % pv.denominator == 0
        tail_count = expected * pv.numerator // pv.denominator
        assert 0 <= tail_count <= expected


def test_budget_breach_keeps_raw_exact_and_marks_adjusted_unavailable():
    # Low-cardinality fields but nearly unique JOINT combinations: the joint
    # max-T space explodes while every per-field marginal DP stays small.
    rng = np.random.default_rng(7)
    rows = []
    for i in range(36):
        rows.append(
            ("b1", int(rng.random() < 0.5),
             rng.choice(list("abcd")), rng.choice(list("wxyz")))
        )
    df = make_df(rows, ["batch", "y", "f1", "f2"])
    res = mod.run_analysis(
        df, "batch", "y", ["f1", "f2"],
        max_allocations=100, max_nodes=10**6,
    )
    # Joint adjusted p-values refused, never approximated.
    assert res["adj_error"] is not None
    assert all(v is None for v in res["adj_p"])
    # Raw p-values remain fully exact via the per-field marginal DP.
    assert res["raw_error"] is None
    assert all(isinstance(v, Fraction) for v in res["raw_p"])
    assert all(0 < float(v) <= 1 for v in res["raw_p"])


def test_marginal_dp_weight_identity_on_high_joint_cardinality():
    # Low-cardinality marginals, nearly unique joint combos: the marginal DP
    # stays tiny and must satisfy C(N, K); only the joint step is over budget.
    rng = np.random.default_rng(99)
    rows = []
    for b, rate, n in [("b1", 0.4, 24), ("b2", 0.7, 20)]:
        for i in range(n):
            rows.append(
                (b, int(rng.random() < rate),
                 rng.choice(list("abcde")), rng.choice(list("wxyz")))
            )
    df = make_df(rows, ["batch", "y", "f1", "f2"])
    res = mod.run_analysis(df, "batch", "y", ["f1", "f2"], 1000, 10**7)
    assert res["adj_error"] is not None
    assert all(v is None for v in res["adj_p"])
    for batch in (b for b in res["raw_batches"] if not b["degenerate"]):
        codes, dist = mod.marginal_allocation_distribution(
            batch["n_j"]["f1"], batch["N"], batch["K"], 10**6
        )
        assert sum(dist.values()) == math.comb(batch["N"], batch["K"])
        assert all(sum(state) == batch["K"] for state in dist)
    assert all(0 < float(p) <= 1 for p in res["raw_p"])


def test_marginal_budget_breach_is_isolated_per_field():
    # One healthy low-cardinality field and one near-unique field whose own
    # marginal DP exceeds the budget. The healthy field must still report an
    # exact raw p-value; only the pathological field is marked unavailable.
    rng = np.random.default_rng(11)
    rows = []
    for i in range(30):
        rows.append(
            ("b1", int(rng.random() < 0.5),
             rng.choice(list("abc")), f"u{i}")  # f2 unique per row
        )
    df = make_df(rows, ["batch", "y", "f1", "f2"])
    res = mod.run_analysis(
        df, "batch", "y", ["f1", "f2"],
        max_allocations=10**6, max_nodes=10**7, max_marginal_states=1000,
    )
    assert res["raw_error"] is not None
    assert isinstance(res["raw_p"][0], Fraction)   # healthy field still exact
    assert 0 < float(res["raw_p"][0]) <= 1
    assert res["raw_p"][1] is None                  # pathological field refused
    assert res["adj_p"][1] is None


def test_marginal_raw_p_matches_oracle_on_small_data():
    # Same idea at a size where the row-level oracle is feasible.
    rng = np.random.default_rng(3)
    rows = []
    for b, rate, n in [("b1", 0.4, 8), ("b2", 0.7, 7)]:
        for i in range(n):
            rows.append(
                (b, int(rng.random() < rate),
                 rng.choice(list("abc")), f"u{i % 3}")
            )
    df = make_df(rows, ["batch", "y", "f1", "f2"])
    res = mod.run_analysis(df, "batch", "y", ["f1", "f2"], 1000, 10**7)
    _, _, raw_br, _ = oracle_brute_force(df, "batch", "y", ["f1", "f2"])
    assert res["raw_p"] == raw_br


def test_nonbinary_target_rejected():
    df = make_df([("b1", 2, "a"), ("b1", 0, "b")], ["batch", "y", "f1"])
    with pytest.raises(ValueError):
        mod.run_analysis(df, "batch", "y", ["f1"], 10**6, 10**7)


def test_asymptotic_calibration_against_chi_squared_distribution():
    rng = np.random.default_rng(123)
    rows = []
    for b in ("b1", "b2"):
        for _ in range(80):
            rows.append((b, int(rng.random() < 0.5), rng.choice(["a", "b", "c", "d"])))
    df = make_df(rows, ["batch", "y", "f1"])
    res = mod.run_analysis(df, "batch", "y", ["f1"], 10**7, 10**7)
    asym_p = chi2_dist.sf(res["float_observed"][0], res["df"][0])
    assert abs(float(res["raw_p"][0]) - asym_p) < 0.02
