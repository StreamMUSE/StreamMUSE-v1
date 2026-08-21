from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_BANDWIDTH_GRID = [0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]


@dataclass(frozen=True)
class ScaleFit:
    center: float
    scale: float
    method: str

    def transform(self, values: Sequence[float]) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.center) / self.scale


def average_rank(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    ranks = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return ranks
    valid = values[mask]
    order = np.argsort(valid, kind="mergesort")
    sorted_valid = valid[order]
    ranked = np.empty(valid.size, dtype=float)
    start = 0
    while start < valid.size:
        end = start + 1
        while end < valid.size and sorted_valid[end] == sorted_valid[start]:
            end += 1
        ranked[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    ranks[mask] = ranked
    return ranks


def spearman_nan(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    xr = average_rank(x[mask])
    yr = average_rank(y[mask])
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = float(np.sqrt(np.dot(xr, xr) * np.dot(yr, yr)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(xr, yr) / denom)


def robust_train_scaler(values: Sequence[float]) -> ScaleFit:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return ScaleFit(center=0.0, scale=1.0, method="constant")
    center = float(np.median(values))
    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr > 0.0:
        return ScaleFit(center=center, scale=iqr, method="iqr")
    mad = float(np.median(np.abs(values - center)))
    if mad > 0.0:
        return ScaleFit(center=center, scale=1.4826 * mad, method="mad")
    std = float(np.std(values))
    if std > 0.0:
        return ScaleFit(center=center, scale=std, method="std")
    return ScaleFit(center=center, scale=1.0, method="constant")


def gaussian_kernel(u: Sequence[float]) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.exp(-0.5 * np.square(u))


def nadaraya_watson_1d(
    x_train: Sequence[float],
    y_train: Sequence[float],
    x_eval: Sequence[float],
    bandwidth: float,
    sample_weight: Sequence[float] | None = None,
    prior: float | None = None,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_eval = np.asarray(x_eval, dtype=float).reshape(-1)
    if x_train.size != y_train.size:
        raise ValueError("x_train and y_train must have the same length")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if sample_weight is None:
        weight = np.ones(y_train.shape, dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weight.size != y_train.size:
            raise ValueError("sample_weight must match y_train length")
    if x_train.size == 0:
        fallback = 0.0 if prior is None else float(prior)
        return np.full(x_eval.shape, fallback, dtype=float)
    weight_sum = float(weight.sum())
    if weight_sum > 0.0:
        fallback = float(np.dot(weight, y_train) / weight_sum)
    else:
        fallback = float(np.mean(y_train))
    kernel = gaussian_kernel((x_eval[:, None] - x_train[None, :]) / float(bandwidth))
    kernel *= weight[None, :]
    denom = kernel.sum(axis=1)
    num = kernel @ y_train
    pred = np.full(x_eval.shape, fallback, dtype=float)
    np.divide(num, denom, out=pred, where=denom > 0.0)
    return pred


def equal_group_mse(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[object],
) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    groups = np.asarray(groups, dtype=object).reshape(-1)
    if not (y_true.size == y_pred.size == groups.size):
        raise ValueError("y_true, y_pred, and groups must have the same length")
    labels = list(dict.fromkeys(groups.tolist()))
    if not labels:
        return float("nan")
    return float(
        np.mean([np.mean((y_true[groups == label] - y_pred[groups == label]) ** 2) for label in labels])
    )


def nested_logo_cv(
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence[object],
    bandwidth_grid: Sequence[float] = DEFAULT_BANDWIDTH_GRID,
    sample_weight: Sequence[float] | None = None,
    min_groups: int = 3,
) -> dict:
    x, y, groups, sample_weight, _, valid_index = _prepare_valid_arrays(x, y, groups, sample_weight=sample_weight)
    invalid = _invalid_reason(x, y, groups, min_groups, valid_index)
    if invalid is not None:
        return invalid
    labels = _labels(groups)
    pred = np.empty(y.shape, dtype=float)
    null_pred = np.empty(y.shape, dtype=float)
    outer_audit = []
    for label in labels:
        test_mask = groups == label
        train_mask = ~test_mask
        bandwidth, _, _ = _select_logo_bandwidth(
            x[train_mask],
            y[train_mask],
            groups[train_mask],
            bandwidth_grid,
            None if sample_weight is None else sample_weight[train_mask],
        )
        fit = robust_train_scaler(x[train_mask])
        train_weight = None if sample_weight is None else sample_weight[train_mask]
        prior = _weighted_mean(y[train_mask], train_weight)
        pred[test_mask] = nadaraya_watson_1d(
            fit.transform(x[train_mask]),
            y[train_mask],
            fit.transform(x[test_mask]),
            bandwidth,
            sample_weight=train_weight,
            prior=prior,
        )
        null_pred[test_mask] = prior
        outer_audit.append(
            {
                "group": label,
                "bandwidth": float(bandwidth),
                "center": float(fit.center),
                "scale": float(fit.scale),
                "scale_method": fit.method,
                "prior": float(prior),
            }
        )
    cv_mse = equal_group_mse(y, pred, groups)
    null_mse = equal_group_mse(y, null_pred, groups)
    return {
        "valid": True,
        "reason": None,
        "n_groups": len(labels),
        "cv_mse": float(cv_mse),
        "null_mse": float(null_mse),
        "skill": float(1.0 - cv_mse / null_mse) if null_mse > 0.0 else float("nan"),
        "predictions": pred,
        "null_predictions": null_pred,
        "original_valid_indices": valid_index,
        "outer_audit": outer_audit,
    }


def select_logo_bandwidth_and_curve(
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence[object],
    bandwidth_grid: Sequence[float] = DEFAULT_BANDWIDTH_GRID,
    sample_weight: Sequence[float] | None = None,
    min_groups: int = 3,
    curve_x: Sequence[float] | None = None,
    n_curve: int = 256,
) -> dict:
    x, y, groups, sample_weight, _, valid_index = _prepare_valid_arrays(x, y, groups, sample_weight=sample_weight)
    invalid = _invalid_reason(x, y, groups, min_groups, valid_index)
    if invalid is not None:
        return invalid
    bandwidth, cv_mse, bandwidth_mse = _select_logo_bandwidth(x, y, groups, bandwidth_grid, sample_weight)
    fit = robust_train_scaler(x)
    prior = _weighted_mean(y, sample_weight)
    if curve_x is None:
        lo = float(np.min(x))
        hi = float(np.max(x))
        curve_x = np.linspace(lo, hi, int(n_curve))
    else:
        curve_x = np.asarray(curve_x, dtype=float).reshape(-1)
    curve_y = nadaraya_watson_1d(
        fit.transform(x),
        y,
        fit.transform(curve_x),
        bandwidth,
        sample_weight=sample_weight,
        prior=prior,
    )
    pred, null_pred, audit, selected_cv_mse, null_mse = _logo_cv_predictions(
        x, y, groups, bandwidth, sample_weight
    )
    return {
        "valid": True,
        "reason": None,
        "n_groups": len(_labels(groups)),
        "bandwidth": float(bandwidth),
        "bandwidth_mse": bandwidth_mse,
        "cv_mse": float(cv_mse if np.isfinite(cv_mse) else selected_cv_mse),
        "null_mse": float(null_mse),
        "skill": float(1.0 - selected_cv_mse / null_mse) if null_mse > 0.0 else float("nan"),
        "curve_x": np.asarray(curve_x, dtype=float),
        "curve_y": curve_y,
        "predictions": pred,
        "null_predictions": null_pred,
        "outer_audit": audit,
        "center": float(fit.center),
        "scale": float(fit.scale),
        "scale_method": fit.method,
        "prior": float(prior),
        "original_valid_indices": valid_index,
    }


def group_block_permutation_skill(
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence[object],
    bandwidth_grid: Sequence[float] = DEFAULT_BANDWIDTH_GRID,
    sample_weight: Sequence[float] | None = None,
    min_groups: int = 3,
    n_perm: int = 1000,
    seed: int = 0,
    within_group_order: Sequence[float] | None = None,
) -> dict:
    observed = nested_logo_cv(
        x,
        y,
        groups,
        bandwidth_grid=bandwidth_grid,
        sample_weight=sample_weight,
        min_groups=min_groups,
    )
    if not observed["valid"]:
        return {
            "valid": False,
            "reason": observed["reason"],
            "observed_skill": float("nan"),
            "perm_skills": np.array([], dtype=float),
            "p_value": float("nan"),
        }
    x, y, groups, sample_weight, within_group_order, _ = _prepare_valid_arrays(
        x,
        y,
        groups,
        sample_weight=sample_weight,
        within_group_order=within_group_order,
    )
    labels, positions, blocks, block_size = _group_blocks(y, groups, within_group_order)
    rng = np.random.default_rng(seed)
    perm_skills = np.empty(int(n_perm), dtype=float)
    for idx in range(int(n_perm)):
        perm = rng.permutation(len(labels))
        y_perm = np.empty_like(y)
        for dest, src in enumerate(perm):
            y_perm[positions[dest]] = blocks[src]
        perm_result = nested_logo_cv(
            x,
            y_perm,
            groups,
            bandwidth_grid=bandwidth_grid,
            sample_weight=sample_weight,
            min_groups=min_groups,
        )
        perm_skills[idx] = perm_result["skill"]
    observed_skill = float(observed["skill"])
    exceed = int(np.sum(perm_skills >= observed_skill))
    return {
        "valid": True,
        "reason": None,
        "observed_skill": observed_skill,
        "perm_skills": perm_skills,
        "p_value": float((exceed + 1.0) / (int(n_perm) + 1.0)),
        "seed": int(seed),
        "n_perm": int(n_perm),
        "block_size": int(block_size),
    }


def group_bootstrap_curve_bands(
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence[object],
    bandwidth_grid: Sequence[float] = DEFAULT_BANDWIDTH_GRID,
    sample_weight: Sequence[float] | None = None,
    min_groups: int = 3,
    curve_x: Sequence[float] | None = None,
    n_curve: int = 256,
    bandwidth: float | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    fit = select_logo_bandwidth_and_curve(
        x,
        y,
        groups,
        bandwidth_grid=bandwidth_grid,
        sample_weight=sample_weight,
        min_groups=min_groups,
        curve_x=curve_x,
        n_curve=n_curve,
    )
    if not fit["valid"]:
        return {
            "valid": False,
            "reason": fit["reason"],
            "curve_x": np.array([], dtype=float),
            "curve_y": np.array([], dtype=float),
            "lower": np.array([], dtype=float),
            "upper": np.array([], dtype=float),
            "boot_curves": np.empty((0, 0), dtype=float),
        }
    x, y, groups, sample_weight, _, _ = _prepare_valid_arrays(x, y, groups, sample_weight=sample_weight)
    labels = _labels(groups)
    group_rows = [np.flatnonzero(groups == label) for label in labels]
    curve_x = np.asarray(fit["curve_x"], dtype=float)
    bandwidth = float(fit["bandwidth"] if bandwidth is None else bandwidth)
    rng = np.random.default_rng(seed)
    curves = np.empty((int(n_boot), curve_x.size), dtype=float)
    for boot_idx in range(int(n_boot)):
        picks = rng.integers(0, len(labels), size=len(labels))
        xb = np.concatenate([x[group_rows[pick]] for pick in picks])
        yb = np.concatenate([y[group_rows[pick]] for pick in picks])
        if sample_weight is None:
            wb = None
        else:
            wb = np.concatenate([sample_weight[group_rows[pick]] for pick in picks])
        fit_b = robust_train_scaler(xb)
        prior_b = _weighted_mean(yb, wb)
        curves[boot_idx] = nadaraya_watson_1d(
            fit_b.transform(xb),
            yb,
            fit_b.transform(curve_x),
            bandwidth,
            sample_weight=wb,
            prior=prior_b,
        )
    return {
        "valid": True,
        "reason": None,
        "curve_x": curve_x,
        "curve_y": np.asarray(fit["curve_y"], dtype=float),
        "lower": np.quantile(curves, alpha / 2.0, axis=0),
        "upper": np.quantile(curves, 1.0 - alpha / 2.0, axis=0),
        "boot_curves": curves,
        "bandwidth": bandwidth,
        "seed": int(seed),
        "n_boot": int(n_boot),
    }


def bh_qvalues(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float).reshape(-1)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    mask = np.isfinite(p_values)
    if not mask.any():
        return q_values
    valid = np.clip(p_values[mask], 0.0, 1.0)
    order = np.argsort(valid)
    ranked = valid[order]
    m = ranked.size
    adjusted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_valid = np.empty_like(adjusted)
    q_valid[order] = adjusted
    q_values[mask] = q_valid
    return q_values


def _prepare_valid_arrays(
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence[object],
    sample_weight: Sequence[float] | None = None,
    within_group_order: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    groups = np.asarray(groups, dtype=object).reshape(-1)
    if not (x.size == y.size == groups.size):
        raise ValueError("x, y, and groups must have the same length")
    mask = np.isfinite(x) & np.isfinite(y) & np.array([_valid_group(value) for value in groups], dtype=bool)
    if sample_weight is None:
        weight = None
    else:
        weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weight.size != x.size:
            raise ValueError("sample_weight must match x length")
        mask &= np.isfinite(weight) & (weight >= 0.0)
    if within_group_order is None:
        order = None
    else:
        order = np.asarray(within_group_order, dtype=float).reshape(-1)
        if order.size != x.size:
            raise ValueError("within_group_order must match x length")
        mask &= np.isfinite(order)
    valid_index = np.flatnonzero(mask)
    return (
        x[mask],
        y[mask],
        groups[mask],
        None if weight is None else weight[mask],
        None if order is None else order[mask],
        valid_index,
    )


def _invalid_reason(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    min_groups: int,
    valid_index: np.ndarray,
) -> dict | None:
    n_groups = len(_labels(groups))
    if n_groups < int(min_groups):
        return {
            "valid": False,
            "reason": {"code": "insufficient_groups", "n_groups": n_groups, "min_groups": int(min_groups)},
            "n_groups": n_groups,
            "cv_mse": float("nan"),
            "null_mse": float("nan"),
            "skill": float("nan"),
            "predictions": np.array([], dtype=float),
            "null_predictions": np.array([], dtype=float),
            "original_valid_indices": valid_index,
            "outer_audit": [],
        }
    if x.size == 0 or float(np.max(x) - np.min(x)) == 0.0:
        value = float(x[0]) if x.size else float("nan")
        return {
            "valid": False,
            "reason": {"code": "constant_feature", "value": value, "n_groups": n_groups},
            "n_groups": n_groups,
            "cv_mse": float("nan"),
            "null_mse": float("nan"),
            "skill": float("nan"),
            "predictions": np.array([], dtype=float),
            "null_predictions": np.array([], dtype=float),
            "original_valid_indices": valid_index,
            "outer_audit": [],
        }
    if y.size == 0 or float(np.max(y) - np.min(y)) == 0.0:
        value = float(y[0]) if y.size else float("nan")
        return {
            "valid": False,
            "reason": {"code": "constant_target", "value": value, "n_groups": n_groups},
            "n_groups": n_groups,
            "cv_mse": float("nan"),
            "null_mse": float("nan"),
            "skill": float("nan"),
            "predictions": np.array([], dtype=float),
            "null_predictions": np.array([], dtype=float),
            "original_valid_indices": valid_index,
            "outer_audit": [],
        }
    return None


def _labels(groups: np.ndarray) -> list[object]:
    return list(dict.fromkeys(np.asarray(groups, dtype=object).tolist()))


def _weighted_mean(values: np.ndarray, sample_weight: np.ndarray | None) -> float:
    if values.size == 0:
        return 0.0
    if sample_weight is None:
        return float(np.mean(values))
    total = float(sample_weight.sum())
    if total > 0.0:
        return float(np.dot(sample_weight, values) / total)
    return float(np.mean(values))


def _logo_cv_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    bandwidth: float,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[dict], float, float]:
    pred = np.empty(y.shape, dtype=float)
    null_pred = np.empty(y.shape, dtype=float)
    audit = []
    for label in _labels(groups):
        test_mask = groups == label
        train_mask = ~test_mask
        fit = robust_train_scaler(x[train_mask])
        train_weight = None if sample_weight is None else sample_weight[train_mask]
        prior = _weighted_mean(y[train_mask], train_weight)
        pred[test_mask] = nadaraya_watson_1d(
            fit.transform(x[train_mask]),
            y[train_mask],
            fit.transform(x[test_mask]),
            bandwidth,
            sample_weight=train_weight,
            prior=prior,
        )
        null_pred[test_mask] = prior
        audit.append(
            {
                "group": label,
                "bandwidth": float(bandwidth),
                "center": float(fit.center),
                "scale": float(fit.scale),
                "scale_method": fit.method,
                "prior": float(prior),
            }
        )
    return pred, null_pred, audit, equal_group_mse(y, pred, groups), equal_group_mse(y, null_pred, groups)


def _select_logo_bandwidth(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    bandwidth_grid: Sequence[float],
    sample_weight: np.ndarray | None,
) -> tuple[float, float, dict]:
    best_bandwidth = None
    best_mse = float("inf")
    bandwidth_mse = {}
    for bandwidth in bandwidth_grid:
        _, _, _, mse, _ = _logo_cv_predictions(x, y, groups, float(bandwidth), sample_weight)
        bandwidth_mse[float(bandwidth)] = float(mse)
        if mse < best_mse:
            best_mse = float(mse)
            best_bandwidth = float(bandwidth)
    if best_bandwidth is None:
        raise ValueError("bandwidth_grid must be non-empty")
    return best_bandwidth, best_mse, bandwidth_mse


def _group_blocks(
    values: np.ndarray,
    groups: np.ndarray,
    within_group_order: np.ndarray | None,
) -> tuple[list[object], list[np.ndarray], list[np.ndarray], int]:
    labels = _labels(groups)
    positions = []
    blocks = []
    for label in labels:
        pos = np.flatnonzero(groups == label)
        if within_group_order is not None:
            pos = pos[np.argsort(within_group_order[pos], kind="mergesort")]
        positions.append(pos)
        blocks.append(values[pos].copy())
    sizes = {block.size for block in blocks}
    if len(sizes) != 1:
        raise ValueError("group blocks must have equal size or provide within_group_order alignment")
    block_size = int(next(iter(sizes)))
    return labels, positions, blocks, block_size


def _valid_group(value: object) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return True
