# Exact stratified Pearson χ² with joint max-T adjustment

`exact_batch_chi2.py` tests the association of several categorical business
fields with a binary conversion label, when rows come from sampling batches
with different conversion rates. Everything is exact — there is no Monte Carlo
or asymptotic approximation in the test itself.

## What it does

1. **Native encoding.** Each field is encoded with sklearn's
   `OrdinalEncoder`; missing values are normalized to NaN and given their own
   distinct category code `-1` (reported as `<MISSING>`).
2. **Per-batch Pearson χ².** For every field, the Pearson χ² of the
   field × binary-target contingency table is computed inside each batch and
   summed across batches. Arithmetic uses an exact integer numerator over an
   exact integer denominator (`χ² = Σ (N·o − n·K)² / (n·K·(N−K))`).
3. **Stratified exact null.** Each batch's total number of conversions `K_b`
   is fixed. Within a batch, the same relabelling of the `K_b` positive labels
   is drawn (without replacement) and shared by all fields; batches are
   independent.
4. **Exact weighting, no sampling.** Rows with the same joint category
   combination in a batch are aggregated with row weight `w_c`; an allocation
   of `k_c` positives to combination `c` has multiplicity `∏_c C(w_c, k_c)`.
   Weights are always counted in original-row units. If the exact state space
   exceeds the hard budgets, the run **refuses to approximate** and reports the
   affected p-values as `UNAVAILABLE`.
5. **Two p-values per field, ties included (`>=`):**
   - *raw* — `P(T_field >= T_obs)`, from a per-field marginal exact DP (always
     computable, independent of the other fields' cardinality);
   - *adjusted* — joint max-T `P(max_g T_g >= T_obs_field)` under the shared
     relabelling, from an exact joint DP (the multiplicity correction across
     fields). For a single field the adjusted p-value equals the raw p-value.

A single global common denominator `D = lcm` of every per-(batch, field)
denominator turns every statistic into an exact integer, so all dynamic-program
state keys are plain integers (no per-operation fraction/gcd cost). Fractions
appear only when the final p-values are formed.

## Usage

```bash
python3 -I exact_batch_chi2.py \
  --data path/to/data.csv \
  --batch-col batch \
  --target-col converted \
  --fields city,channel,segment \
  --out-prefix exact_chi2
```

Outputs:

- `<prefix>_field_summary.csv` — per-field observed χ², degrees of freedom,
  raw and max-T adjusted p-values (both float and exact fraction strings);
- `<prefix>_category_counts.csv` — original (unpermuted) category counts per
  field overall and per batch, with the missing category on its own row;
- `<prefix>_batch_summary.csv` — per-batch `N`, conversions, rate, number of
  distinct joint combinations, `C(N,K)` and (when computed) enumerated leaves.

`--max-allocations` and `--max-nodes` are hard exact-state budgets, not sample
sizes. Exceeding them marks adjusted p-values `UNAVAILABLE`; raw p-values stay
exact.

## Validation

`tests/test_exact_batch_chi2.py` builds an independent row-level brute-force
oracle (every `K`-of-`N` relabelling per batch, no aggregation) and checks the
engine's observed statistics, raw and adjusted p-values against it exactly,
including tie handling, missing-as-category, duplicate-combination weights,
degenerate batches, and the no-sampling budget behavior.

```bash
python3 -I -m pytest -q
```

## Scope note

The joint max-T adjustment integrates the full joint permutation distribution,
so its exact state space grows with the number of distinct *joint* category
combinations per batch. With many effectively-unique rows it can exceed any
feasible exact budget; in that regime the raw (per-field) exact p-values remain
available and the adjusted values are reported as unavailable rather than
estimated.
