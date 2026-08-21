import numpy as np
import pytest

from experiments.lekai_failcase_analysis.kde_model import (
    DEFAULT_BANDWIDTH_GRID,
    bh_qvalues,
    equal_group_mse,
    group_block_permutation_skill,
    group_bootstrap_curve_bands,
    nested_logo_cv,
    robust_train_scaler,
    select_logo_bandwidth_and_curve,
    spearman_nan,
)


def _u_shape_dataset():
    base = np.linspace(-3.0, 3.0, 13)
    x = np.concatenate([base + (g - 2.5) * 0.005 for g in range(6)])
    y = x**2
    groups = np.repeat(np.arange(6), base.size)
    order = np.tile(np.arange(base.size), 6)
    return x, y, groups, order


def test_grouped_u_shape_has_positive_skill_while_spearman_is_near_zero():
    x, y, groups, _ = _u_shape_dataset()

    rho = spearman_nan(x, y)
    result = nested_logo_cv(x, y, groups)
    curve = select_logo_bandwidth_and_curve(x, y, groups, n_curve=81)

    assert abs(rho) < 0.2
    assert result["valid"] is True
    assert result["n_groups"] == 6
    assert result["skill"] > 0.75
    assert len(result["predictions"]) == len(result["original_valid_indices"]) == x.size
    assert len(result["outer_audit"]) == 6
    assert curve["valid"] is True
    assert curve["bandwidth"] in DEFAULT_BANDWIDTH_GRID
    mid = curve["curve_y"][len(curve["curve_y"]) // 2]
    assert mid == pytest.approx(np.min(curve["curve_y"]))
    assert curve["curve_y"][0] > mid
    assert curve["curve_y"][-1] > mid


def test_nested_logo_audit_excludes_held_group_from_scaler_and_prior():
    x = np.array([0.0, 0.1, 1.0, 1.1, 100.0, 100.1])
    y = np.array([0.0, 0.0, 1.0, 1.0, 10.0, 10.0])
    groups = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)

    result = nested_logo_cv(x, y, groups)
    audit_c = next(item for item in result["outer_audit"] if item["group"] == "c")

    assert result["valid"] is True
    assert audit_c["center"] == pytest.approx(0.55)
    assert audit_c["prior"] == pytest.approx(0.5)


def test_constant_feature_returns_structured_invalid_reason():
    scale = robust_train_scaler([2.0, 2.0, 2.0])
    constant = nested_logo_cv(
        x=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        y=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        groups=["a", "a", "b", "b", "c", "c"],
    )

    assert scale.method == "constant"
    assert constant["valid"] is False
    assert constant["reason"] == {"code": "constant_feature", "value": 2.0, "n_groups": 3}


def test_insufficient_groups_after_nan_filter_and_constant_target_are_invalid():
    constant_target = nested_logo_cv(
        x=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        y=[7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        groups=["a", "a", "b", "b", "c", "c"],
    )
    insufficient = nested_logo_cv(
        x=[0.0, np.nan, 2.0],
        y=[1.0, 2.0, 3.0],
        groups=["a", "b", "c"],
        min_groups=3,
    )

    assert constant_target["valid"] is False
    assert constant_target["reason"]["code"] == "constant_target"
    assert constant_target["reason"]["value"] == pytest.approx(7.0)
    assert insufficient["valid"] is False
    assert insufficient["reason"] == {"code": "insufficient_groups", "n_groups": 2, "min_groups": 3}
    assert insufficient["original_valid_indices"].tolist() == [0, 2]


def test_equal_group_loss_uses_exact_group_weighting():
    group_mse = equal_group_mse(
        y_true=[0.0, 0.0, 10.0, 10.0, 10.0],
        y_pred=[0.0, 1.0, 10.0, 10.0, 0.0],
        groups=["small", "small", "big", "big", "big"],
    )

    assert group_mse == pytest.approx((0.5 + (100.0 / 3.0)) / 2.0)


def test_block_permutation_preserves_group_structure_and_is_deterministic():
    x, y, groups, order = _u_shape_dataset()
    perm1 = group_block_permutation_skill(x, y, groups, n_perm=32, seed=11, within_group_order=order)
    perm2 = group_block_permutation_skill(x, y, groups, n_perm=32, seed=11, within_group_order=order)

    assert perm1["valid"] is True
    assert perm1["block_size"] == 13
    assert perm1["observed_skill"] > 0.75
    assert np.array_equal(perm1["perm_skills"], perm2["perm_skills"])
    assert perm1["p_value"] == pytest.approx(perm2["p_value"])
    assert perm1["p_value"] == pytest.approx(1.0 / 33.0)


def test_bootstrap_bands_are_deterministic_and_well_formed():
    x, y, groups, _ = _u_shape_dataset()

    bands1 = group_bootstrap_curve_bands(x, y, groups, n_curve=51, n_boot=24, seed=17)
    bands2 = group_bootstrap_curve_bands(x, y, groups, n_curve=51, n_boot=24, seed=17)

    assert bands1["valid"] is True
    assert np.array_equal(bands1["curve_x"], bands2["curve_x"])
    assert np.allclose(bands1["curve_y"], bands2["curve_y"])
    assert np.allclose(bands1["lower"], bands2["lower"])
    assert np.allclose(bands1["upper"], bands2["upper"])
    assert bands1["boot_curves"].shape == (24, 51)
    assert np.all(bands1["lower"] <= bands1["upper"])


def test_bh_qvalues_match_expected_monotone_adjustment():
    q = bh_qvalues([0.04, 0.01, 0.03, np.nan, 0.20])

    assert np.isnan(q[3])
    assert q[:3].tolist() == pytest.approx([0.05333333333333334, 0.04, 0.05333333333333334])
    assert q[4] == pytest.approx(0.2)
