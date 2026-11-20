"""Exact stratified (per-batch) Pearson chi-squared tests for categorical fields
vs. a binary conversion target, with joint-permutation max-T adjusted p-values.

Design (per the analysis specification):
  * Category codes come from sklearn's *native* OrdinalEncoder. Missing values
    are encoded as their own distinct category (code -1).
  * For every field, the Pearson chi-squared statistic of the field x binary
    target contingency table is computed inside every sampling batch and the
    per-batch statistics are summed across batches.
  * The null distribution conditions on each batch's total number of
    conversions K_b. Within a batch one common relabelling of the K_b positive
    labels is drawn (without replacement) and shared by every field; batches
    are independent.
  * Rows sharing the same joint category combination inside a batch are
    aggregated with weight w_c. An allocation of k_c positives to combination c
    has exact multiplicity prod_c C(w_c, k_c). Every weight is counted in
    original-row units; there is no Monte Carlo / sampling approximation.
  * Raw p-values are per-field tail probabilities P(T_f >= T_f,obs).
  * Adjusted p-values are joint max-T tail probabilities
    P(max_g T_g >= T_f,obs) under the shared allocations.
  * Ties are included in the tails (>=, not >).

Exact arithmetic: for a batch of size N with K positives and per-category
totals n_j / positives o_j,
    chi2 = sum_j (N*o_j - n_j*K)^2 / (n_j * K * (N-K)).
Each (batch, field) has an exact integer denominator L_bf. A single global
common denominator D_all = lcm_bf(L_bf) is used, so every statistic becomes an
exact integer (numerator * D_all/L_bf); all distribution-state keys are plain
integers and cross-batch sums are integer additions. Fractions appear only when
the final p-values are formed. There is no floating point anywhere in the test.
"""

from __future__ import annotations

import argparse
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

MISSING_CODE = -1
MISSING_LABEL = "<MISSING>"


# --------------------------------------------------------------------------- #
# Encoding and contingency preparation
# --------------------------------------------------------------------------- #
def encode_fields(df: pd.DataFrame, fields: list[str]) -> tuple[pd.DataFrame, dict[str, list]]:
    """Fit a native sklearn OrdinalEncoder; missing -> its own category (-1).

    sklearn 1.7 keeps NaN in `categories_` and, for pandas object columns with
    None, does not always apply encoded_missing_value, so missing is normalized
    to NaN before fitting and remapped to MISSING_CODE afterwards.
    """
    encoder = OrdinalEncoder(encoded_missing_value=MISSING_CODE)
    fit_frame = df[fields].astype(object).where(df[fields].notna(), np.nan)
    raw = encoder.fit_transform(fit_frame)
    raw = np.where(np.isnan(raw), MISSING_CODE, raw).astype(np.int64)
    codes = pd.DataFrame(raw, columns=fields, index=df.index)
    categories = {
        field: list(encoder.categories_[i]) for i, field in enumerate(fields)
    }
    return codes, categories


def category_label(code: int, labels: list) -> str:
    if code == MISSING_CODE:
        return MISSING_LABEL
    val = labels[code]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return MISSING_LABEL
    return str(val)


def build_batch(sub: pd.DataFrame, fields: list[str]) -> dict:
    y = sub["__y__"].to_numpy()
    N, K = int(len(sub)), int(y.sum())
    combo_cols = sub[fields].to_numpy()

    # Aggregate duplicate joint category combinations, keeping row weights.
    weights: dict[tuple, int] = {}
    combo_pos: dict[tuple, int] = {}
    for i in range(N):
        key = tuple(int(v) for v in combo_cols[i])
        weights[key] = weights.get(key, 0) + 1
        combo_pos[key] = combo_pos.get(key, 0) + int(y[i])

    n_j: dict[str, dict[int, int]] = {f: {} for f in fields}
    obs: dict[str, dict[int, int]] = {f: {} for f in fields}
    for key, w in weights.items():
        k = combo_pos[key]
        for j, f in enumerate(fields):
            code = key[j]
            n_j[f][code] = n_j[f].get(code, 0) + w
            obs[f][code] = obs[f].get(code, 0) + k

    return {
        "combos": list(weights.items()),
        "n_j": n_j,
        "obs": obs,
        "N": N,
        "K": K,
        "degenerate": (K == 0 or K == N or len(weights) < 2),
    }


# --------------------------------------------------------------------------- #
# Exact integer Pearson chi-squared
# --------------------------------------------------------------------------- #
def batch_denominator(n_j: dict[int, int], N: int, K: int) -> int:
    """L = K*(N-K)*lcm(n_j), so chi2 = integer_numerator / L exactly."""
    lcm = 1
    for n in n_j.values():
        lcm = math.lcm(lcm, n)
    return K * (N - K) * lcm


def batch_stat_numerator(
    positives: dict[int, int], n_j: dict[int, int], N: int, K: int, denom: int
) -> int:
    """Exact integer numerator of the batch Pearson chi-squared over denom."""
    num = 0
    k_nk = K * (N - K)
    for code, n in n_j.items():
        o = positives.get(code, 0)
        d = N * o - n * K
        num += d * d * (denom // (n * k_nk))
    return num


def float_chi2(positives: dict[int, int], n_j: dict[int, int], N: int, K: int) -> float:
    """Standard floating-point Pearson statistic (reporting / cross-check only)."""
    if K == 0 or K == N or len(n_j) < 2:
        return 0.0
    total = 0.0
    for code, n in n_j.items():
        o = positives.get(code, 0)
        e_pos = n * K / N
        e_neg = n - e_pos
        if e_pos > 0:
            total += (o - e_pos) ** 2 / e_pos
        o_neg = n - o
        if e_neg > 0:
            total += (o_neg - e_neg) ** 2 / e_neg
    return total


# --------------------------------------------------------------------------- #
# Exact enumeration within one batch (integer statistics on the global scale)
# --------------------------------------------------------------------------- #
def marginal_allocation_distribution(
    category_weights: dict[int, int],
    N: int,
    K: int,
    max_states: int,
) -> tuple[list[int], dict[tuple[int, ...], int]]:
    """Exact distribution of positive counts over one field's categories.

    Returns (category codes, {tuple of per-category positive counts: exact
    multiplicity prod_c C(w_c, k_c)}). Only depends on this field's category
    weights, so it is independent of the other fields' joint cardinality.
    Raises BudgetExceeded if the merged state space exceeds max_states.
    """
    codes = list(category_weights)
    weights = [category_weights[c] for c in codes]
    dp: dict[tuple[int, ...], int] = {(): 1}
    for c, w in enumerate(weights):
        later = sum(weights[c + 1 :])
        new_dp: dict[tuple[int, ...], int] = {}
        for state, mass in dp.items():
            used = sum(state)
            lo = max(0, K - used - later)
            hi = min(w, K - used)
            for k in range(lo, hi + 1):
                nxt = state + (k,)
                new_dp[nxt] = new_dp.get(nxt, 0) + mass * math.comb(w, k)
        if len(new_dp) > max_states:
            raise BudgetExceeded(
                f"per-field marginal DP exceeded {max_states:,} states "
                f"(field has {len(weights)} categories, N={N}, K={K})"
            )
        dp = new_dp
    if sum(dp.values()) != math.comb(N, K):
        raise AssertionError("marginal DP weight check failed")
    return codes, dp


def enumerate_marginal_stats(
    batch: dict, field: str, global_denom: int, max_states: int
) -> dict[int, int]:
    """{integer statistic (global scale): exact multiplicity} for one field."""
    N, K = batch["N"], batch["K"]
    n_j = batch["n_j"][field]
    if batch["degenerate"] or len(n_j) < 2:
        return {0: math.comb(N, K)}
    L = batch_denominator(n_j, N, K)
    scale = global_denom // L
    codes, dist = marginal_allocation_distribution(n_j, N, K, max_states)
    out: dict[int, int] = {}
    for counts, mass in dist.items():
        positives = {codes[i]: counts[i] for i in range(len(codes))}
        stat = batch_stat_numerator(positives, n_j, N, K, L) * scale
        out[stat] = out.get(stat, 0) + mass
    return out


def enumerate_batch_allocations(
    batch: dict, fields: list[str], global_denom: int, max_allocations: int
) -> dict:
    """Enumerate every exact *joint* label allocation of the batch.

    An allocation assigns k_c positives to each distinct joint combination c
    with multiplicity prod_c C(w_c, k_c), and yields every field's integer
    statistic under that one shared relabelling. Powers the joint max-T
    adjustment; raw tails use the cheaper per-field marginal DP.
    """
    N, K = batch["N"], batch["K"]
    combos = batch["combos"]
    p = len(fields)

    Ls, scales = [], []
    for f in fields:
        n_j = batch["n_j"][f]
        if batch["degenerate"] or len(n_j) < 2:
            Ls.append(None)
            scales.append(0)
        else:
            L = batch_denominator(n_j, N, K)
            Ls.append(L)
            scales.append(global_denom // L)

    run_pos = [dict() for _ in fields]
    stats_acc: list[tuple[int, tuple[int, ...]]] = []
    leaves = {"n": 0}
    combo_codes = [key for key, _ in combos]
    combo_weights = [w for _, w in combos]

    def dfs(c: int, remaining: int, weight: int) -> None:
        if c == len(combos):
            if remaining != 0:
                return
            leaves["n"] += 1
            if leaves["n"] > max_allocations:
                raise BudgetExceeded(
                    f"batch of N={N}, K={K} needs more than "
                    f"{max_allocations:,} distinct joint allocations; raw "
                    "p-values remain exact via marginal DP"
                )
            stats = []
            for j, f in enumerate(fields):
                if Ls[j] is None:
                    stats.append(0)
                else:
                    num = batch_stat_numerator(
                        run_pos[j], batch["n_j"][f], N, K, Ls[j]
                    )
                    stats.append(num * scales[j])
            stats_acc.append((weight, tuple(stats)))
            return

        w = combo_weights[c]
        later_weight = sum(combo_weights[c + 1 :])
        lo = max(0, remaining - later_weight)
        hi = min(w, remaining)
        codes = combo_codes[c]
        for k in range(lo, hi + 1):
            if k:
                for j in range(p):
                    code = codes[j]
                    run_pos[j][code] = run_pos[j].get(code, 0) + k
            dfs(c + 1, remaining - k, weight * math.comb(w, k))
            if k:
                for j in range(p):
                    code = codes[j]
                    v = run_pos[j][code] - k
                    if v:
                        run_pos[j][code] = v
                    else:
                        del run_pos[j][code]

    dfs(0, K, 1)

    total_weight = sum(w for w, _ in stats_acc)
    if total_weight != math.comb(N, K):
        raise AssertionError(
            f"allocation weight check failed: {total_weight} != C({N},{K})"
        )
    max_stat = [0] * p
    for _, st in stats_acc:
        for j, v in enumerate(st):
            if v > max_stat[j]:
                max_stat[j] = v
    return {
        "allocations": stats_acc,
        "total_weight": total_weight,
        "max_stat": max_stat,
        "n_leaves": leaves["n"],
    }


class BudgetExceeded(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Tail probabilities
# --------------------------------------------------------------------------- #
def raw_tail_distributions(
    marginal_stats: list[list[dict[int, int]]],
    observed: list[int],
    total_weight: int,
    p: int,
) -> list[Fraction]:
    """Per-field exact integer convolutions; P(T_f >= T_f,obs), ties included.

    States that cannot reach the observed threshold even after adding the
    maximum attainable statistic of every remaining batch are pruned. This
    removes only outcomes that could never enter the tail, so the result is
    bit-identical to the unpruned convolution.
    """
    B = len(marginal_stats)
    # suffix_max[j][b] = max attainable field-j statistic from batches b..B-1.
    suffix_max = [[0] * (B + 1) for _ in range(p)]
    for b in range(B - 1, -1, -1):
        for j in range(p):
            suffix_max[j][b] = suffix_max[j][b + 1] + max(marginal_stats[b][j])

    dps: list[dict[int, int]] = [{0: 1} for _ in range(p)]
    for b in range(B):
        for j in range(p):
            new_dp: dict[int, int] = {}
            reach_floor = observed[j] - suffix_max[j][b + 1]
            for t0, w0 in dps[j].items():
                for t1, w1 in marginal_stats[b][j].items():
                    key = t0 + t1
                    if key < reach_floor:
                        continue  # cannot reach observed even with max remainder
                    new_dp[key] = new_dp.get(key, 0) + w0 * w1
            dps[j] = new_dp
    return [
        Fraction(sum(w for t, w in dps[j].items() if t >= observed[j]), total_weight)
        for j in range(p)
    ]


def adjusted_tail_distributions(
    batch_enums: list[dict],
    observed: list[int],
    total_weight: int,
    p: int,
    max_nodes: int,
) -> list[Fraction]:
    """Joint max-T tails under the shared per-batch allocation, exact DP.

    For threshold t (= a field's observed integer statistic), each field's
    running statistic is tracked only as min(cumulative, t): crossing t is all
    that matters. That caps and merges states, integrating the joint null
    exactly without the raw cartesian product. Ties are included (>= via cap).
    One pass per distinct observed threshold.
    """
    thresholds: dict[int, list[int]] = {}
    for j, t in enumerate(observed):
        thresholds.setdefault(t, []).append(j)

    B = len(batch_enums)
    suffix_max = [[0] * (B + 1) for _ in range(p)]
    for b in range(B - 1, -1, -1):
        for j in range(p):
            suffix_max[j][b] = suffix_max[j][b + 1] + batch_enums[b]["max_stat"][j]

    pvals: list[Fraction | None] = [None] * p
    for t, members in thresholds.items():
        dp: dict[tuple[int, ...], int] = {tuple(0 for _ in range(p)): 1}
        nodes = 0
        for b in range(B):
            be = batch_enums[b]
            new_dp: dict[tuple[int, ...], int] = {}
            for state, w0 in dp.items():
                nodes += 1
                if nodes > max_nodes:
                    raise BudgetExceeded(
                        f"joint max-T DP exceeded {max_nodes:,} merged states at "
                        f"threshold {t}; refuse to approximate by sampling"
                    )
                # A field that can never reach t even with the max remaining
                # contribution is irrelevant to this threshold; prune the whole
                # state only if no field can reach t anymore.
                reachable = any(
                    state[j] + suffix_max[j][b] >= t for j in range(p)
                )
                if not reachable:
                    continue
                for w_alloc, st in be["allocations"]:
                    key = tuple(
                        t if state[j] + st[j] >= t else state[j] + st[j]
                        for j in range(p)
                    )
                    new_dp[key] = new_dp.get(key, 0) + w0 * w_alloc
            dp = new_dp
        numerator = sum(w for state, w in dp.items() if t in state)
        for j in members:
            pvals[j] = Fraction(numerator, total_weight)
    assert all(v is not None for v in pvals)
    return pvals  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_analysis(
    df: pd.DataFrame,
    batch_col: str,
    target_col: str,
    fields: list[str],
    max_allocations: int,
    max_nodes: int,
    max_marginal_states: int = 2_000_000,
) -> dict:
    if int(df[target_col].isna().sum()):
        raise ValueError(
            f"target {target_col} has missing values; remove or impute them first."
        )
    y = df[target_col]
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(
            f"target {target_col} must be binary 0/1, got {sorted(np.unique(y))}"
        )

    codes, categories = encode_fields(df, fields)
    work = codes.copy()
    work["__y__"] = y.to_numpy()
    work["__batch__"] = df[batch_col].to_numpy()

    raw_batches = [
        build_batch(sub.reset_index(drop=True), fields)
        for _, sub in work.groupby("__batch__", sort=False)
    ]
    p = len(fields)

    # Global common denominator over every (active batch, field).
    active_batches = [b for b in raw_batches if not b["degenerate"]]
    global_denom = 1
    for b in active_batches:
        for f in fields:
            n_j = b["n_j"][f]
            if len(n_j) >= 2:
                global_denom = math.lcm(
                    global_denom, batch_denominator(n_j, b["N"], b["K"])
                )

    # Observed integer statistics (global scale), float stats, and degrees of
    # freedom. Degenerate batches contribute zero statistic.
    observed = [0] * p
    float_observed = [0.0] * p
    degrees_of_freedom = [0] * p
    for b in raw_batches:
        N, K = b["N"], b["K"]
        for j, f in enumerate(fields):
            n_j = b["n_j"][f]
            float_observed[j] += float_chi2(b["obs"][f], n_j, N, K)
            if b["degenerate"] or len(n_j) < 2:
                continue
            L = batch_denominator(n_j, N, K)
            num = batch_stat_numerator(b["obs"][f], n_j, N, K, L)
            observed[j] += num * (global_denom // L)
            degrees_of_freedom[j] += len(n_j) - 1

    adjusted_error: str | None = None
    raw_error: str | None = None
    if not active_batches:
        raw_p: list[Fraction | None] = [Fraction(1) for _ in fields]
        adj_p: list[Fraction | None] = [Fraction(1) for _ in fields]
        marginal_stats: list = []
        batch_enums: list[dict] = []
        total_weight = 1
    else:
        total_weight = 1
        for b in active_batches:
            total_weight *= math.comb(b["N"], b["K"])

        # Raw tails: per-field marginal DP, exact at any joint cardinality.
        # Computed field by field so one pathological (near-unique) field cannot
        # take down the others; an over-budget field is reported unavailable,
        # never approximated.
        raw_p = [None] * p
        for j, f in enumerate(fields):
            if observed[j] == 0:
                raw_p[j] = Fraction(1)
                continue
            try:
                field_marg = [
                    enumerate_marginal_stats(b, f, global_denom, max_marginal_states)
                    for b in active_batches
                ]
                raw_p[j] = raw_tail_distributions(
                    [[d] for d in field_marg], [observed[j]], total_weight, 1
                )[0]
            except BudgetExceeded as exc:
                raw_error = str(exc)
                raw_p[j] = None

        # Adjusted tails. Single active field: max T is T itself, so the
        # adjusted p-value equals the raw p-value (no joint enumeration needed).
        adj_p: list[Fraction | None] = [
            Fraction(1) if raw_p[j] is not None else None for j in range(p)
        ]
        batch_enums = []
        active_fields = [j for j, t in enumerate(observed) if t > 0 and raw_p[j] is not None]
        if not active_fields:
            pass
        elif len(active_fields) == 1:
            adj_p[active_fields[0]] = raw_p[active_fields[0]]
        else:
            try:
                for b in active_batches:
                    batch_enums.append(
                        enumerate_batch_allocations(
                            b, fields, global_denom, max_allocations
                        )
                    )
                sub_enums = [
                    {
                        "allocations": [
                            (w, tuple(st[j] for j in active_fields))
                            for w, st in be["allocations"]
                        ],
                        "total_weight": be["total_weight"],
                        "max_stat": [be["max_stat"][j] for j in active_fields],
                    }
                    for be in batch_enums
                ]
                sub_obs = [observed[j] for j in active_fields]
                sub_adj = adjusted_tail_distributions(
                    sub_enums, sub_obs, total_weight, len(active_fields), max_nodes
                )
                for slot, j in enumerate(active_fields):
                    adj_p[j] = sub_adj[slot]
            except BudgetExceeded as exc:
                # No sampling fallback: report adjusted as unavailable, never
                # guessed; raw p-values above remain fully exact.
                adjusted_error = str(exc)
                adj_p = [None for _ in fields]
                batch_enums = []

    return {
        "fields": fields,
        "categories": categories,
        "codes": codes,
        "raw_batches": raw_batches,
        "batch_enums": batch_enums,
        "observed": observed,
        "global_denom": global_denom,
        "float_observed": float_observed,
        "raw_p": raw_p,
        "adj_p": adj_p,
        "raw_error": raw_error,
        "adj_error": adjusted_error,
        "df": degrees_of_freedom,
        "total_weight": total_weight,
    }


def category_count_table(
    df: pd.DataFrame,
    codes: pd.DataFrame,
    fields: list[str],
    categories: dict[str, list],
    batch_col: str,
) -> pd.DataFrame:
    """Original (unpermuted) category counts; missing is a separate row."""
    batch_values = df[batch_col].to_numpy()
    rows = []
    for f in fields:
        col = codes[f].to_numpy()
        for code in sorted(np.unique(col).tolist()):
            mask = col == code
            label = (
                MISSING_LABEL
                if code == MISSING_CODE
                else category_label(int(code), categories[f])
            )
            row = {
                "field": f,
                "code": int(code),
                "category": label,
                "is_missing": bool(code == MISSING_CODE),
                "n_rows": int(mask.sum()),
            }
            for batch_id in sorted(np.unique(batch_values).tolist()):
                row[f"n_batch_{batch_id}"] = int(
                    np.sum(mask & (batch_values == batch_id))
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV file path")
    parser.add_argument("--batch-col", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument(
        "--fields",
        default="",
        help="comma-separated categorical columns; default: all object/category "
        "columns other than batch/target",
    )
    parser.add_argument("--out-prefix", default="exact_chi2")
    parser.add_argument(
        "--max-allocations", type=int, default=2_000_000,
        help="hard cap on distinct joint allocations enumerated per batch",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=20_000_000,
        help="hard cap on joint max-T DP merged states (no sampling fallback)",
    )
    parser.add_argument(
        "--max-marginal-states", type=int, default=2_000_000,
        help="hard cap on per-field marginal DP states (no sampling fallback)",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    if args.fields:
        fields = [c.strip() for c in args.fields.split(",") if c.strip()]
    else:
        fields = [
            c for c in df.columns
            if c not in (args.batch_col, args.target_col) and df[c].dtype == object
        ]
    missing = [c for c in [args.batch_col, args.target_col, *fields] if c not in df.columns]
    if missing:
        parser.error(f"columns not found in data: {missing}")

    result = run_analysis(
        df, args.batch_col, args.target_col, fields,
        args.max_allocations, args.max_nodes, args.max_marginal_states,
    )

    def _floats(vals):
        return [None if v is None else float(v) for v in vals]

    def _strs(vals):
        return ["UNAVAILABLE" if v is None else str(v) for v in vals]

    prefix = Path(args.out_prefix)
    summary = pd.DataFrame(
        {
            "field": fields,
            "chi2_observed": result["float_observed"],
            "df": result["df"],
            "raw_p": _floats(result["raw_p"]),
            "raw_p_exact": _strs(result["raw_p"]),
            "adj_p_maxT": _floats(result["adj_p"]),
            "adj_p_exact": _strs(result["adj_p"]),
            "n_categories_nonmissing": [
                sum(
                    1
                    for x in result["categories"][f]
                    if not (x is None or (isinstance(x, float) and math.isnan(x)))
                )
                for f in fields
            ],
            "n_missing_rows": [int(df[f].isna().sum()) for f in fields],
        }
    )
    summary.to_csv(f"{prefix}_field_summary.csv", index=False)

    counts = category_count_table(
        df, result["codes"], fields, result["categories"], args.batch_col
    )
    counts.to_csv(f"{prefix}_category_counts.csv", index=False)

    batch_rows = []
    for bi, b in enumerate(result["raw_batches"]):
        if b["degenerate"]:
            continue
        row = {
            "N": b["N"],
            "K_conversions": b["K"],
            "rate": b["K"] / b["N"],
            "distinct_joint_combos": len(b["combos"]),
            "allocations_C(N,K)": math.comb(b["N"], b["K"]),
        }
        if bi < len(result["batch_enums"]):
            row["joint_enumerated_leaves"] = result["batch_enums"][bi]["n_leaves"]
        batch_rows.append(row)
    pd.DataFrame(batch_rows).to_csv(f"{prefix}_batch_summary.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("=== Field-level exact tests (Pearson chi2 summed across batches) ===")
    print(summary.to_string(index=False))
    if result["raw_error"]:
        print(
            "\n[note] Some raw p-values are UNAVAILABLE: a per-field marginal DP\n"
            f"       exceeded its exact budget ({result['raw_error']}). No\n"
            "       sampling approximation was used."
        )
    if result["adj_error"]:
        print(
            "\n[note] Joint max-T adjusted p-values are UNAVAILABLE: the joint\n"
            f"       allocation space exceeded the exact budget ({result['adj_error']}).\n"
            "       Raw p-values above are still fully exact (per-field marginal DP).\n"
            "       No sampling approximation was used. Raise --max-allocations /\n"
            "       --max-nodes or reduce joint-cardinality to obtain adjusted p."
        )
    print("\n=== Active batches ===")
    print(pd.DataFrame(batch_rows).to_string(index=False))
    print("\n=== Original category counts (missing shown as separate row) ===")
    print(counts.to_string(index=False))
    print(
        f"\nWrote: {prefix}_field_summary.csv, {prefix}_category_counts.csv, "
        f"{prefix}_batch_summary.csv"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
