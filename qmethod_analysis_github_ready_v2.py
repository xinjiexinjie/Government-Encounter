#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible Q-methodology analysis for:
"When Government Contacts You First" / Digital State Reach study

This script reproduces the principal quantitative Q-method steps from the
raw Q-sort CSV exported by the online sorting platform.

Implemented:
1. Data-quality checks and forced-distribution verification
2. Participant-by-participant Q-sort correlation matrix
3. PCA extraction from the participant correlation matrix
4. Varimax rotation
5. Comparison of 3-, 4-, and 5-factor solutions
6. Factor-loading significance threshold: 2.58 / sqrt(number of statements)
   (two-tailed p < .01, a conservative criterion commonly used in Q studies)
7. Two transparent flagging rules:
   a) strict_unique: significant on exactly one factor
   b) qmethod_auto: significant loading AND squared target loading exceeds
      the sum of squared loadings on all other retained factors
8. Majority-positive factor orientation (factor signs are arbitrary)
9. Factor arrays using Brown-style weights: w = f / (1 - f^2)
10. Composite reliability, SE, SED, distinguishing statements, and
    consensus statements
11. CSV/TXT outputs suitable for audit, supplementary materials, and review

Dependencies:
    Python >= 3.9
    numpy
    pandas

Example:
    python qmethod_reproducible_analysis.py "Q Sort (1).csv" --factors 4

Outputs are written to:
    qmethod_outputs/

Important interpretation note:
Negative defining loadings are retained and explicitly marked as the
"negative pole". They must not be described as holding the same viewpoint
as positive defining loaders. Signed factor weights reverse their Q-sort
when constructing the positive-pole factor array, which is the standard
logic of a bipolar factor.

Author note:
This script intentionally uses no hidden/manual rotations and no stochastic
steps, so the results are exactly reproducible from the supplied CSV.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


STATEMENTS: Dict[int, str] = {
    1: "政府能找到我并不奇怪，真正让我在意的是它为什么决定主动联系我。",
    2: "政府主动联系我，会让我首先怀疑自己是不是出了什么问题。",
    3: "如果我正常生活、没有明显风险，政府通常不应该主动来找我。",
    4: "在今天的数字社会，政府掌握我的基本信息几乎不可避免，因此这件事本身不会让我特别不安。",
    5: "如果政府能在损失发生前主动发现风险，我会觉得这种能力是有价值的。",
    6: "如果主动联系确实能够避免严重损失，我会少在意政府具体使用了哪些信息。",
    7: "政府主动联系公民只是履行正常工作流程，并不能说明它有多强的治理能力。",
    8: "有些主动联系可能主要是为了完成行政任务，而不是真正因为公民需要帮助。",
    9: "政府系统里有我的数据很正常，但根据这些数据主动找到我会让我产生完全不同的感觉。",
    10: "真正让我在意的不是政府拥有数据，而是它能把这些数据准确对应到具体的我。",
    11: "只要政府已经拥有我的信息，它能够联系到我就是理所当然的。",
    12: "如果政府真的有很强的数字治理能力，更理想的是在风险形成前阻止问题，而不是等风险已经出现后再联系我。",
    13: "当政府准确说出我近期的具体行为时，我会产生一种自己变得“透明”的感觉。",
    14: "我是否接受政府主动联系，首先取决于它为什么要联系我。",
    15: "风险越严重，政府越有理由在未经我请求的情况下主动介入。",
    16: "为防止诈骗而主动联系，与因为我在网上发表公共事务相关内容而主动联系，在我看来不是一回事。",
    17: "即使一次政府主动联系最终对我有好处，也不能因此忽略它获取和使用信息的方式。",
    18: "政府采取多强的干预，应当与风险本身的严重程度相匹配。",
    19: "政府使用个人数据主动联系我之前，原则上应该得到我的授权。",
    20: "如果不授权就无法正常使用一项服务，我不认为这种授权是真正的选择。",
    21: "在紧急公共安全情况下，政府不应因为缺少事先授权而无法及时行动。",
    22: "即使政府有合理理由联系我，我仍然应该拥有是否继续互动的选择权。",
    23: "如果我已经明确表示不需要帮助，政府一般应该停止继续联系。",
    24: "如果政府说不清为什么联系我，即使出发点是好的，我也会对这次联系保持怀疑。",
    25: "我希望知道是哪一类信息触发了这次政府联系。",
    26: "我最关心的不是政府知道多少，而是这些信息会被用来做什么。",
    27: "如果多个政府部门为了处理同一件事需要共享信息，我可以接受这种共享。",
    28: "为一个目的获得的信息，不应该随意被用于完全不同的目的。",
    29: "只要这些信息留在政府内部、没有给我造成实际损害，我并不太在意事情结束后它是否继续被保存。",
    30: "相比政府知道很多信息，我更担心政府根据错误信息对我采取行动。",
    31: "政府在主动联系个人之前，应尽可能通过多个信息来源核实风险。",
    32: "偶尔的误判可以接受，只要政府能够及时发现并纠正。",
    33: "如果系统无法解释为什么把我判断为风险对象，我会降低对这套系统的信任。",
    34: "一个陌生号码自称政府部门时，即使它知道我的个人信息，我也不会马上相信。",
    35: "只要我能够通过官方渠道独立核验对方身份，我就更愿意接受这种未经请求的政府联系。",
    36: "政府联系公民时采用什么方式、什么态度，会影响我是否愿意配合。",
    37: "对于严重风险，我宁愿政府多联系一次，也不愿它因为担心误判而错过真正的危险。",
    38: "如果政府认为我需要帮助，但我自己并没有这种需要，主动服务也可能成为一种负担。",
    39: "同一件事被多个政府部门重复联系，会让我觉得制度之间缺乏协调。",
    40: "即使这次政府主动联系真的帮助了我，我仍可能因为它暴露了政府对我行为的了解程度而感到不安。",
}


def varimax(phi: np.ndarray, gamma: float = 1.0, max_iter: int = 500,
            tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Orthogonal Varimax rotation."""
    p, k = phi.shape
    rotation = np.eye(k)
    previous = 0.0

    for _ in range(max_iter):
        lam = phi @ rotation
        target = phi.T @ (
            lam ** 3
            - (gamma / p) * lam @ np.diag(np.diag(lam.T @ lam))
        )
        u, s, vh = np.linalg.svd(target)
        rotation = u @ vh
        current = float(s.sum())

        if previous > 0 and current / previous < 1.0 + tol:
            break
        previous = current

    return phi @ rotation, rotation


def pca_varimax(corr: np.ndarray, n_factors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA extraction followed by Varimax rotation."""
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    unrotated = eigvecs[:, :n_factors] * np.sqrt(eigvals[:n_factors])
    rotated, rotation_matrix = varimax(unrotated)
    return eigvals, rotated, rotation_matrix


def significance_threshold(n_statements: int, zcrit: float = 2.58) -> float:
    """Significant loading threshold; default corresponds to p < .01."""
    return zcrit / math.sqrt(n_statements)


def strict_unique_flags(loadings: np.ndarray, threshold: float) -> np.ndarray:
    """
    Strict flagging:
    a Q-sort is defining only when it is significant on exactly one factor.
    """
    sig = np.abs(loadings) >= threshold
    return sig & (sig.sum(axis=1, keepdims=True) == 1)


def qmethod_auto_flags(loadings: np.ndarray, threshold: float) -> np.ndarray:
    """
    Automatic Q-method flagging used in the qmethod literature:
      1) loading is statistically significant; AND
      2) squared loading on target factor exceeds the sum of squared
         loadings on all other retained factors.
    """
    n_people, n_factors = loadings.shape
    flags = np.zeros((n_people, n_factors), dtype=bool)

    for i in range(n_people):
        sumsq = float(np.sum(loadings[i] ** 2))
        for j in range(n_factors):
            target_sq = float(loadings[i, j] ** 2)
            if (
                abs(loadings[i, j]) >= threshold
                and target_sq > (sumsq - target_sq)
            ):
                flags[i, j] = True
    return flags


def orient_factors(loadings: np.ndarray, flags: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
    """
    Factor sign is mathematically arbitrary. Orient each factor so the
    majority of flagged/defining Q-sorts load positively.
    """
    oriented = loadings.copy()
    flips: List[bool] = []

    for j in range(oriented.shape[1]):
        vals = oriented[flags[:, j], j]
        should_flip = int(np.sum(vals < 0)) > int(np.sum(vals > 0))
        if should_flip:
            oriented[:, j] *= -1
        flips.append(should_flip)

    return oriented, flips


def infer_grid_distribution(x: np.ndarray) -> np.ndarray:
    """
    Infer the discrete forced distribution from the first Q-sort, after
    verifying that every participant has exactly the same score counts.
    """
    reference_values, reference_counts = np.unique(x[0], return_counts=True)
    reference = dict(zip(reference_values.tolist(), reference_counts.tolist()))

    for i in range(1, x.shape[0]):
        values, counts = np.unique(x[i], return_counts=True)
        current = dict(zip(values.tolist(), counts.tolist()))
        if current != reference:
            raise ValueError(
                f"Participant row {i + 1} does not match the forced distribution. "
                f"Expected {reference}; found {current}."
            )

    return np.sort(x[0].copy())


def factor_arrays(
    x: np.ndarray,
    loadings: np.ndarray,
    flags: np.ndarray,
    grid_distribution: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct factor z-scores using signed Brown weights:
        w = f / (1 - f^2)

    Negative defining loaders therefore contribute the reverse of their
    Q-sort to the positive-pole factor array.

    Q-sorts are standardized across statements before aggregation.
    Factor z-scores are standardized again across statements.
    Discrete factor scores are assigned by rank to the original forced grid.
    """
    row_mean = x.mean(axis=1, keepdims=True)
    row_sd = x.std(axis=1, ddof=1, keepdims=True)
    z_person = (x - row_mean) / row_sd

    n_factors = loadings.shape[1]
    n_statements = x.shape[1]
    z_arrays = np.zeros((n_factors, n_statements), dtype=float)
    score_arrays = np.zeros((n_factors, n_statements), dtype=int)

    for j in range(n_factors):
        mask = flags[:, j]
        if mask.sum() < 2:
            raise ValueError(
                f"Factor {j + 1} has fewer than two flagged Q-sorts; "
                "factor array is not considered stable enough to construct."
            )

        f = loadings[mask, j]
        weights = f / (1.0 - f ** 2)
        composite = np.average(z_person[mask], axis=0, weights=weights)
        composite = (composite - composite.mean()) / composite.std(ddof=1)
        z_arrays[j] = composite

        rank_order = np.argsort(composite)
        assigned = np.empty(n_statements, dtype=int)
        assigned[rank_order] = grid_distribution.astype(int)
        score_arrays[j] = assigned

    return z_arrays, score_arrays


def factor_reliability_and_se(
    n_flagged: np.ndarray,
    grid_sd: float,
    assumed_qsort_reliability: float = 0.80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conventional Q-method composite reliability and standard error:
        r_f = (r * p) / [1 + (p - 1)r]
        SE_f = SD_grid * sqrt(1 - r_f)
    """
    p = n_flagged.astype(float)
    r = assumed_qsort_reliability
    reliability = (r * p) / (1.0 + (p - 1.0) * r)
    se = grid_sd * np.sqrt(1.0 - reliability)
    return reliability, se


def distinguishing_and_consensus(
    z_arrays: np.ndarray,
    factor_scores: np.ndarray,
    factor_se: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Distinguishing statement:
      z-score for one factor differs from EVERY other factor at p < .05.
      A p < .01 indicator is also supplied.

    Consensus statement:
      no pairwise factor difference is significant at p < .05.
    """
    n_factors, n_statements = z_arrays.shape
    sed = np.sqrt(factor_se[:, None] ** 2 + factor_se[None, :] ** 2)

    dist_rows = []
    consensus_rows = []

    for s in range(n_statements):
        pair_sig_05 = {}
        pair_sig_01 = {}

        for a in range(n_factors):
            for b in range(a + 1, n_factors):
                diff = abs(z_arrays[a, s] - z_arrays[b, s])
                pair_sig_05[(a, b)] = diff > 1.96 * sed[a, b]
                pair_sig_01[(a, b)] = diff > 2.58 * sed[a, b]

        if not any(pair_sig_05.values()):
            row = {
                "statement": s + 1,
                "text": STATEMENTS.get(s + 1, ""),
            }
            for j in range(n_factors):
                row[f"F{j+1}_z"] = z_arrays[j, s]
                row[f"F{j+1}_score"] = factor_scores[j, s]
            consensus_rows.append(row)

        for j in range(n_factors):
            sig05_all = []
            sig01_all = []
            for k in range(n_factors):
                if k == j:
                    continue
                a, b = sorted((j, k))
                sig05_all.append(pair_sig_05[(a, b)])
                sig01_all.append(pair_sig_01[(a, b)])

            if all(sig05_all):
                row = {
                    "factor": f"F{j+1}",
                    "statement": s + 1,
                    "text": STATEMENTS.get(s + 1, ""),
                    "factor_z": z_arrays[j, s],
                    "factor_score": factor_scores[j, s],
                    "distinguishing_p05": True,
                    "distinguishing_p01": all(sig01_all),
                }
                for k in range(n_factors):
                    row[f"F{k+1}_z"] = z_arrays[k, s]
                    row[f"F{k+1}_score"] = factor_scores[k, s]
                dist_rows.append(row)

    return pd.DataFrame(dist_rows), pd.DataFrame(consensus_rows)


def classify_loadings(
    session_ids: List[str],
    loadings: np.ndarray,
    flags: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for i, sid in enumerate(session_ids):
        flagged = np.where(flags[i])[0]
        significant = np.where(np.abs(loadings[i]) >= threshold)[0]

        if len(flagged) == 0:
            status = "unflagged"
        elif len(flagged) == 1:
            j = int(flagged[0])
            pole = "positive" if loadings[i, j] > 0 else "negative"
            status = f"defining_F{j+1}_{pole}_pole"
        else:
            # Should be rare/impossible under squared-loading dominance,
            # but retained for transparency.
            status = "multiple_flags_" + "_".join(f"F{x+1}" for x in flagged)

        row = {
            "session_id": sid,
            "status": status,
            "n_significant_loadings": len(significant),
            "n_flags": len(flagged),
        }
        for j in range(loadings.shape[1]):
            row[f"F{j+1}_loading"] = loadings[i, j]
            row[f"F{j+1}_significant"] = bool(abs(loadings[i, j]) >= threshold)
            row[f"F{j+1}_flagged"] = bool(flags[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def solution_diagnostics(
    corr: np.ndarray,
    n_statements: int,
    factors_to_compare: List[int],
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    threshold = significance_threshold(n_statements)
    rows = []
    loadings_store: Dict[int, np.ndarray] = {}

    for k in factors_to_compare:
        eigvals, raw_rotated, _ = pca_varimax(corr, k)

        raw_flags = qmethod_auto_flags(raw_rotated, threshold)
        rotated, flips = orient_factors(raw_rotated, raw_flags)
        flags = qmethod_auto_flags(rotated, threshold)
        strict = strict_unique_flags(rotated, threshold)
        loadings_store[k] = rotated

        rotated_variance = (rotated ** 2).sum(axis=0) / corr.shape[0] * 100.0
        total_explained = float(eigvals[:k].sum() / corr.shape[0] * 100.0)

        row = {
            "n_factors": k,
            "total_explained_variance_pct": total_explained,
            "unrotated_eigenvalues_retained": "; ".join(
                f"{v:.6f}" for v in eigvals[:k]
            ),
            "qmethod_auto_flagged_counts": "; ".join(
                str(int(v)) for v in flags.sum(axis=0)
            ),
            "strict_unique_counts": "; ".join(
                str(int(v)) for v in strict.sum(axis=0)
            ),
            "unflagged_qsorts_qmethod_auto": int((flags.sum(axis=1) == 0).sum()),
            "factor_sign_flips": "; ".join(str(bool(v)) for v in flips),
            "rotated_variance_pct_by_factor": "; ".join(
                f"{v:.6f}" for v in rotated_variance
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows), loadings_store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible PCA/Varimax Q-methodology analysis."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="Q Sort (1).csv",
        help="Input CSV. First column must be participant/session ID; remaining columns are statements.",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=4,
        help="Primary number of retained factors (default: 4).",
    )
    parser.add_argument(
        "--outdir",
        default="qmethod_outputs",
        help="Output directory (default: qmethod_outputs).",
    )
    args = parser.parse_args()

    input_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    if data.shape[1] < 3:
        raise ValueError("CSV must contain one ID column and multiple Q-statement columns.")

    id_col = data.columns[0]
    session_ids = data[id_col].astype(str).tolist()

    q = data.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
    x = q.to_numpy(dtype=float)
    n_people, n_statements = x.shape

    if n_statements != 40:
        print(
            f"NOTE: Found {n_statements} statements rather than 40. "
            "Analysis will proceed; embedded statement text may be incomplete."
        )

    if q.isna().any().any():
        raise ValueError("Missing values detected in Q-sort columns.")

    grid = infer_grid_distribution(x)
    threshold = significance_threshold(n_statements)

    # Participant-by-participant Q correlations.
    corr = np.corrcoef(x)
    corr_df = pd.DataFrame(corr, index=session_ids, columns=session_ids)
    corr_df.to_csv(outdir / "participant_qsort_correlations.csv", encoding="utf-8-sig")

    # Eigenvalues from full participant correlation matrix.
    eigvals_full = np.linalg.eigvalsh(corr)[::-1]
    pd.DataFrame({
        "component": np.arange(1, len(eigvals_full) + 1),
        "eigenvalue": eigvals_full,
        "variance_pct": eigvals_full / n_people * 100.0,
        "cumulative_variance_pct": np.cumsum(eigvals_full) / n_people * 100.0,
    }).to_csv(outdir / "eigenvalues.csv", index=False, encoding="utf-8-sig")

    # Compare 3, 4, and 5 factor solutions unless impossible.
    compare = [k for k in [3, 4, 5] if k < n_people]
    diagnostics, loadings_store = solution_diagnostics(
        corr, n_statements, compare
    )
    diagnostics.to_csv(
        outdir / "factor_solution_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save loading matrices for each compared solution.
    for k, loadings in loadings_store.items():
        flags = qmethod_auto_flags(loadings, threshold)
        strict = strict_unique_flags(loadings, threshold)

        temp = pd.DataFrame(
            loadings,
            index=session_ids,
            columns=[f"F{i+1}" for i in range(k)],
        )
        for j in range(k):
            temp[f"F{j+1}_auto_flag"] = flags[:, j]
            temp[f"F{j+1}_strict_unique"] = strict[:, j]
        temp.index.name = "session_id"
        temp.to_csv(
            outdir / f"loadings_{k}factor.csv",
            encoding="utf-8-sig",
        )

    # Primary solution.
    k = args.factors
    eigvals, raw_rotated, _ = pca_varimax(corr, k)
    initial_flags = qmethod_auto_flags(raw_rotated, threshold)
    loadings, flips = orient_factors(raw_rotated, initial_flags)
    auto_flags = qmethod_auto_flags(loadings, threshold)
    strict_flags = strict_unique_flags(loadings, threshold)

    classification = classify_loadings(
        session_ids, loadings, auto_flags, threshold
    )
    classification.to_csv(
        outdir / f"participant_classification_{k}factor.csv",
        index=False,
        encoding="utf-8-sig",
    )

    strict_classification = classify_loadings(
        session_ids, loadings, strict_flags, threshold
    )
    strict_classification.to_csv(
        outdir / f"participant_classification_{k}factor_STRICT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    z_arrays, factor_scores = factor_arrays(
        x, loadings, auto_flags, grid
    )

    array_rows = []
    for s in range(n_statements):
        row = {
            "statement": s + 1,
            "text": STATEMENTS.get(s + 1, ""),
        }
        for j in range(k):
            row[f"F{j+1}_z"] = z_arrays[j, s]
            row[f"F{j+1}_score"] = factor_scores[j, s]
        array_rows.append(row)

    pd.DataFrame(array_rows).to_csv(
        outdir / f"factor_arrays_{k}factor.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Factor-array correlations.
    fa_corr = np.corrcoef(z_arrays)
    pd.DataFrame(
        fa_corr,
        index=[f"F{i+1}" for i in range(k)],
        columns=[f"F{i+1}" for i in range(k)],
    ).to_csv(
        outdir / f"factor_array_correlations_{k}factor.csv",
        encoding="utf-8-sig",
    )

    # Reliability, SE, SED.
    p_counts = auto_flags.sum(axis=0)
    grid_sd = float(np.std(grid, ddof=1))
    reliability, se = factor_reliability_and_se(p_counts, grid_sd)
    sed = np.sqrt(se[:, None] ** 2 + se[None, :] ** 2)

    factor_characteristics = pd.DataFrame({
        "factor": [f"F{i+1}" for i in range(k)],
        "n_flagged_qsorts": p_counts,
        "n_positive_flagged": [
            int(np.sum(auto_flags[:, j] & (loadings[:, j] > 0)))
            for j in range(k)
        ],
        "n_negative_flagged": [
            int(np.sum(auto_flags[:, j] & (loadings[:, j] < 0)))
            for j in range(k)
        ],
        "rotated_variance_pct": (loadings ** 2).sum(axis=0) / n_people * 100.0,
        "composite_reliability": reliability,
        "standard_error_factor_scores": se,
        "factor_sign_flipped": flips,
    })
    factor_characteristics.to_csv(
        outdir / f"factor_characteristics_{k}factor.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        sed,
        index=[f"F{i+1}" for i in range(k)],
        columns=[f"F{i+1}" for i in range(k)],
    ).to_csv(
        outdir / f"SED_matrix_{k}factor.csv",
        encoding="utf-8-sig",
    )

    # Distinguishing / consensus statements.
    distinguishing, consensus = distinguishing_and_consensus(
        z_arrays, factor_scores, se
    )
    distinguishing.to_csv(
        outdir / f"distinguishing_statements_{k}factor.csv",
        index=False,
        encoding="utf-8-sig",
    )
    consensus.to_csv(
        outdir / f"consensus_statements_{k}factor.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Data-quality summary.
    unique_scores, counts = np.unique(grid, return_counts=True)
    grid_text = ", ".join(
        f"{int(v)}:{int(c)}" for v, c in zip(unique_scores, counts)
    )

    offdiag = corr[np.triu_indices_from(corr, k=1)]
    total_explained = float(eigvals[:k].sum() / n_people * 100.0)

    summary_lines = [
        "Q-METHODOLOGY REPRODUCIBLE ANALYSIS SUMMARY",
        "=" * 56,
        f"Input file: {input_path}",
        f"Participants: {n_people}",
        f"Statements: {n_statements}",
        f"Missing Q-sort values: {int(np.isnan(x).sum())}",
        f"Forced grid counts: {grid_text}",
        f"Loading significance threshold (p < .01): {threshold:.6f}",
        "",
        "Participant Q-sort correlations (off-diagonal):",
        f"  mean = {offdiag.mean():.6f}",
        f"  median = {np.median(offdiag):.6f}",
        f"  min = {offdiag.min():.6f}",
        f"  max = {offdiag.max():.6f}",
        "",
        f"Primary solution: {k} factors, PCA + Varimax",
        f"Total unrotated variance explained by retained factors: {total_explained:.6f}%",
        f"Auto-flagged counts: {p_counts.tolist()}",
        f"Strict-unique counts: {strict_flags.sum(axis=0).tolist()}",
        f"Positive flagged counts: {factor_characteristics['n_positive_flagged'].tolist()}",
        f"Negative flagged counts: {factor_characteristics['n_negative_flagged'].tolist()}",
        f"Rotated variance by factor (%): "
        + ", ".join(f"{v:.6f}" for v in factor_characteristics["rotated_variance_pct"]),
        "",
        "IMPORTANT:",
        "Negative defining loadings indicate an opposing pole of a bipolar factor.",
        "Do not describe negative-pole participants as sharing the positive-pole viewpoint.",
        "",
        "See factor_solution_comparison.csv for the 3/4/5-factor comparison.",
    ]

    (outdir / "analysis_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n".join(summary_lines))
    print(f"\nOutputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
