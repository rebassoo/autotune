"""
Tests for the generic N-snapshot preprocessing pipeline.

Tests 1-3 require no real data files — all inputs are synthetic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune_gp.config import SnapshotCfg
from preprocessing.pipeline import drop_nan_zrg_features_generic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshots(labels):
    """Minimal SnapshotCfg objects — only .label is used by drop_nan_zrg_features_generic."""
    return [SnapshotCfg(label=l, weight=1.0, sim_dir="", obs_dir="") for l in labels]


def _make_zrg_result(n_members, n_zonal, n_regions, n_snaps, var_names, fill=1.0):
    """
    Build a minimal zrg_result dict that drop_nan_zrg_features_generic expects.

    Layout per variable: n_snaps blocks of (n_zonal + n_regions + 1) columns.
    zrg_obs layout: n_vars blocks of n_feat columns (single obs row).
    """
    n_per_snap = n_zonal + n_regions + 1
    n_feat = n_per_snap * n_snaps
    members = [f"m{i:03d}" for i in range(n_members)]
    result = {}
    per_var_dfs = []
    for var in var_names:
        df = pd.DataFrame(
            np.full((n_members, n_feat), fill),
            index=members,
        )
        result[f"{var}_zrg_ppedataset"] = df
        per_var_dfs.append(df)

    result["zrg_ppedataset"] = pd.concat(per_var_dfs, axis=1)
    result["zrg_obs"] = pd.DataFrame(
        np.full((1, len(var_names) * n_feat), fill),
        index=["obs"],
    )
    return result


# ---------------------------------------------------------------------------
# Test 1: calendar-weighted mean differs from unweighted for non-uniform data
# ---------------------------------------------------------------------------

class TestCalendarWeightedMean:
    """
    Directly tests the averaging logic used in load_and_mask_generic.

    The pipeline does:
        weights = ds.time.dt.days_in_month.astype(float)
        ds = ds.weighted(weights).mean(dim="time")

    With two months of unequal length and different values, the weighted and
    unweighted results must differ, and the weighted result must match the
    analytically expected value.
    """

    def _make_two_month_dataset(self):
        """January (31 days, value=0) and February non-leap (28 days, value=1)."""
        times = pd.date_range("2019-01-01", periods=2, freq="MS")
        data = xr.DataArray(
            [0.0, 1.0],
            dims=["time"],
            coords={"time": times},
        )
        return xr.Dataset({"field": data})

    def test_weighted_differs_from_unweighted(self):
        ds = self._make_two_month_dataset()
        unweighted = float(ds["field"].mean(dim="time"))
        weights = ds.time.dt.days_in_month.astype(float)
        weighted = float(ds["field"].weighted(weights).mean(dim="time"))
        assert unweighted != pytest.approx(weighted), (
            "Weighted and unweighted means should differ for months of unequal length"
        )

    def test_weighted_mean_matches_expected(self):
        ds = self._make_two_month_dataset()
        weights = ds.time.dt.days_in_month.astype(float)
        weighted = float(ds["field"].weighted(weights).mean(dim="time"))
        # Jan=31 days * 0.0 + Feb=28 days * 1.0 / (31+28) = 28/59
        expected = 28.0 / 59.0
        assert weighted == pytest.approx(expected, rel=1e-6)

    def test_unweighted_mean_is_half(self):
        ds = self._make_two_month_dataset()
        unweighted = float(ds["field"].mean(dim="time"))
        assert unweighted == pytest.approx(0.5)

    def test_uniform_data_gives_same_result(self):
        """When all values are equal, weighted and unweighted must agree."""
        times = pd.date_range("2019-01-01", periods=12, freq="MS")
        ds = xr.Dataset({"field": xr.DataArray(
            np.ones(12), dims=["time"], coords={"time": times}
        )})
        weights = ds.time.dt.days_in_month.astype(float)
        weighted = float(ds["field"].weighted(weights).mean(dim="time"))
        unweighted = float(ds["field"].mean(dim="time"))
        assert weighted == pytest.approx(unweighted, rel=1e-10)


# ---------------------------------------------------------------------------
# Test 2: drop_nan_zrg_features_generic — symmetric dropping across snapshots
# ---------------------------------------------------------------------------

class TestDropNanSymmetric:
    """
    If a zonal-band column is all-NaN in one snapshot, the same positional
    column must be dropped from all other snapshots too.
    """

    N_ZONAL = 3
    N_REGIONS = 2
    N_SNAPS = 2
    VAR_NAMES = ["VAR1", "VAR2"]

    def _make_result_with_nan_col(self, nan_col_idx):
        """Set column nan_col_idx of VAR1 to NaN (all members)."""
        result = _make_zrg_result(
            n_members=4,
            n_zonal=self.N_ZONAL,
            n_regions=self.N_REGIONS,
            n_snaps=self.N_SNAPS,
            var_names=self.VAR_NAMES,
        )
        result["VAR1_zrg_ppedataset"].iloc[:, nan_col_idx] = np.nan
        return result

    def test_n_zonal_decrements(self):
        # Column 0 = zonal band 0 in snapshot 0 → should reduce n_zonal by 1
        result = self._make_result_with_nan_col(0)
        snaps = _make_snapshots(["S1", "S2"])
        _, new_n_zonal = drop_nan_zrg_features_generic(
            result, self.VAR_NAMES, self.N_ZONAL, self.N_REGIONS,
            ["r0", "r1"], snaps,
        )
        assert new_n_zonal == self.N_ZONAL - 1

    def test_mirror_column_also_dropped(self):
        """Column 0 is all-NaN in S1 → column n_per_snap (S2 mirror) must also be dropped."""
        n_per_snap = self.N_ZONAL + self.N_REGIONS + 1   # 6
        result = self._make_result_with_nan_col(0)
        snaps = _make_snapshots(["S1", "S2"])
        updated, _ = drop_nan_zrg_features_generic(
            result, self.VAR_NAMES, self.N_ZONAL, self.N_REGIONS,
            ["r0", "r1"], snaps,
        )
        n_feat_after = updated["VAR1_zrg_ppedataset"].shape[1]
        n_feat_before = self.N_SNAPS * n_per_snap
        assert n_feat_after == n_feat_before - 2   # one column removed from each snapshot

    def test_all_variables_drop_same_columns(self):
        """The column drop must be applied identically to every variable."""
        n_per_snap = self.N_ZONAL + self.N_REGIONS + 1
        result = self._make_result_with_nan_col(0)
        snaps = _make_snapshots(["S1", "S2"])
        updated, _ = drop_nan_zrg_features_generic(
            result, self.VAR_NAMES, self.N_ZONAL, self.N_REGIONS,
            ["r0", "r1"], snaps,
        )
        shapes = {v: updated[f"{v}_zrg_ppedataset"].shape[1] for v in self.VAR_NAMES}
        assert len(set(shapes.values())) == 1, f"Variable shapes differ after drop: {shapes}"

    def test_zrg_obs_columns_also_dropped(self):
        """zrg_obs must have the same columns removed as the per-variable DataFrames."""
        n_per_snap = self.N_ZONAL + self.N_REGIONS + 1
        n_feat = n_per_snap * self.N_SNAPS
        result = self._make_result_with_nan_col(0)
        snaps = _make_snapshots(["S1", "S2"])
        updated, _ = drop_nan_zrg_features_generic(
            result, self.VAR_NAMES, self.N_ZONAL, self.N_REGIONS,
            ["r0", "r1"], snaps,
        )
        n_vars = len(self.VAR_NAMES)
        expected_obs_cols = n_vars * (n_feat - 2)
        assert updated["zrg_obs"].shape[1] == expected_obs_cols

    def test_no_drop_when_no_nans(self):
        """No columns should be dropped if nothing is all-NaN."""
        n_per_snap = self.N_ZONAL + self.N_REGIONS + 1
        result = _make_zrg_result(
            n_members=4, n_zonal=self.N_ZONAL, n_regions=self.N_REGIONS,
            n_snaps=self.N_SNAPS, var_names=self.VAR_NAMES,
        )
        snaps = _make_snapshots(["S1", "S2"])
        updated, new_n_zonal = drop_nan_zrg_features_generic(
            result, self.VAR_NAMES, self.N_ZONAL, self.N_REGIONS,
            ["r0", "r1"], snaps,
        )
        assert new_n_zonal == self.N_ZONAL
        assert updated["VAR1_zrg_ppedataset"].shape == result["VAR1_zrg_ppedataset"].shape


# ---------------------------------------------------------------------------
# Test 3: drop_nan_zrg_features_generic — explicit zonal band dropping
# ---------------------------------------------------------------------------

class TestDropNanExplicit:
    """
    Bands listed in explicit_drop_zonal must be removed from all snapshots
    even if the data is not all-NaN.
    """

    N_ZONAL = 3
    N_REGIONS = 2
    N_SNAPS = 2
    VAR_NAMES = ["VAR1"]

    def _band_centre(self, idx):
        """Centre latitude of zonal band idx for N_ZONAL bands."""
        edges = np.linspace(-90, 90, self.N_ZONAL + 1)
        return float((edges[idx] + edges[idx + 1]) / 2)

    def test_explicit_band_dropped_despite_no_nans(self):
        """Requesting a band by centre latitude must drop it even if all values are finite."""
        n_per_snap = self.N_ZONAL + self.N_REGIONS + 1
        result = _make_zrg_result(
            n_members=4, n_zonal=self.N_ZONAL, n_regions=self.N_REGIONS,
            n_snaps=self.N_SNAPS, var_names=self.VAR_NAMES,
        )
        snaps = _make_snapshots(["S1", "S2"])
        centre = self._band_centre(0)
        updated, new_n_zonal = drop_nan_zrg_features_generic(
            result, self.VAR_NAMES, self.N_ZONAL, self.N_REGIONS,
            ["r0", "r1"], snaps,
            explicit_drop_zonal=[centre],
        )
        assert new_n_zonal == self.N_ZONAL - 1
        # Both snapshots lose 1 column each
        assert updated["VAR1_zrg_ppedataset"].shape[1] == self.N_SNAPS * n_per_snap - 2

    def test_explicit_drop_is_symmetric(self):
        """Explicitly dropped band must be removed from both snapshots, not just the first."""
        n_per_snap = self.N_ZONAL + self.N_REGIONS + 1
        result = _make_zrg_result(
            n_members=4, n_zonal=self.N_ZONAL, n_regions=self.N_REGIONS,
            n_snaps=self.N_SNAPS, var_names=self.VAR_NAMES,
        )
        snaps = _make_snapshots(["S1", "S2"])
        centre = self._band_centre(1)   # middle band
        updated, _ = drop_nan_zrg_features_generic(
            result, self.VAR_NAMES, self.N_ZONAL, self.N_REGIONS,
            ["r0", "r1"], snaps,
            explicit_drop_zonal=[centre],
        )
        # After dropping: S1 loses col 1, S2 loses col n_per_snap+1
        # Remaining columns = 2*n_per_snap - 2
        assert updated["VAR1_zrg_ppedataset"].shape[1] == 2 * n_per_snap - 2

    def test_multiple_explicit_bands(self):
        """Two explicit bands must each be removed symmetrically."""
        n_per_snap = self.N_ZONAL + self.N_REGIONS + 1
        result = _make_zrg_result(
            n_members=4, n_zonal=self.N_ZONAL, n_regions=self.N_REGIONS,
            n_snaps=self.N_SNAPS, var_names=self.VAR_NAMES,
        )
        snaps = _make_snapshots(["S1", "S2"])
        centres = [self._band_centre(0), self._band_centre(2)]
        updated, new_n_zonal = drop_nan_zrg_features_generic(
            result, self.VAR_NAMES, self.N_ZONAL, self.N_REGIONS,
            ["r0", "r1"], snaps,
            explicit_drop_zonal=centres,
        )
        assert new_n_zonal == self.N_ZONAL - 2
        assert updated["VAR1_zrg_ppedataset"].shape[1] == 2 * n_per_snap - 4
