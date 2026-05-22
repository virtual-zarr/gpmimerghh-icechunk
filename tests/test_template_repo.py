"""Tests for ``template_repo.initialize_repo`` using a tiny on-disk HDF5
fixture (no S3, no network).

The fixture mirrors the GPM IMERG HH layout (a ``Grid`` group with the same
variable names and attribute structure) but at a tiny grid size so the test
runs in well under a second.

To run:
    pytest tests/test_template_repo.py -v
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry

from tests.hdf5_fixtures import LON_ATTRS, LAT_ATTRS, ROOT_ATTRS, _build_fixture
from notebooks import helpers
from template_repo import _is_initialized, initialize_repo

# Test cube dimensions — small enough to be fast, large enough to actually
# exercise multi-chunk lon-chunking.
TEST_NLON = 12        # → 2 chunks of size 6 for the 24-chunk vars
TEST_NLAT = 6
TEST_N_TIME = 48      # one synthetic "day" of half-hours
TEST_T0 = datetime(1998, 1, 1)
TEST_TIME_CHUNK = TEST_N_TIME  # one shard


@pytest.fixture
def fixture_file(tmp_path: Path) -> Path:
    """Path to a freshly-built HDF5 fixture for this test."""
    path = tmp_path / "fixture.HDF5"
    _build_fixture(path)
    return path



@pytest.fixture
def sample(fixture_file: Path, local_registry: ObjectStoreRegistry) -> xr.Dataset:
    """The fixture opened via ``helpers.open_vds_with_coords`` — i.e. exactly
    the shape ``initialize_repo`` consumes in production.
    """
    return helpers.open_vds_with_coords(
        f"file://{fixture_file}",
        registry=local_registry,
    )


@pytest.fixture
def test_repo(tmp_path: Path) -> icechunk.Repository:
    """A fresh icechunk repo in a tmp dir, with a small manifest split size
    and a no-credentials virtual chunk container so nothing tries to reach
    out to S3.
    """
    virtual_chunk_url = f"file://{tmp_path}/"
    return helpers.open_or_create_repo(
        storage=icechunk.local_filesystem_storage(path=str(tmp_path / "repo")),
        manifest_split_size=TEST_N_TIME,
        virtual_chunk_url=virtual_chunk_url,
        virtual_chunk_store=icechunk.local_filesystem_store(str(tmp_path)),
        # Local filesystem needs no credentials, but the container must still
        # be present in the auth map so icechunk will resolve chunks from it.
        virtual_chunk_credentials={virtual_chunk_url: None},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initialize_repo_writes_coords_and_attrs(
    test_repo: icechunk.Repository, sample: xr.Dataset
) -> None:
    """The single happy-path test: init the repo and verify the coord arrays,
    root attrs, data array shapes, and per-data-variable attrs / fill values
    were all written correctly.
    """
    initialize_repo(
        repo=test_repo,
        sample=sample,
        n_time=TEST_N_TIME,
        t0=TEST_T0,
        time_chunk=TEST_TIME_CHUNK,
    )

    # Re-open read-only and inspect what landed in the repo.
    read = test_repo.readonly_session("main")
    root = zarr.open_group(store=read.store, mode="r")

    # ---- time --------------------------------------------------------
    time_arr = root["time"]
    assert time_arr.shape == (TEST_N_TIME,)
    assert time_arr.dtype == np.int64
    assert time_arr.chunks == (TEST_TIME_CHUNK,)
    # First value == t0 expressed in ns since the unix epoch.
    expected_ns = np.datetime64(TEST_T0, "ns").view("int64")
    assert int(time_arr[0]) == int(expected_ns)
    # Spacing is 30 minutes.
    half_hour_ns = np.timedelta64(30, "m").astype("timedelta64[ns]").view("int64")
    assert int(time_arr[1]) - int(time_arr[0]) == int(half_hour_ns)
    # CF decode attrs are present.
    assert time_arr.attrs["units"] == "nanoseconds since 1970-01-01"
    assert time_arr.attrs["calendar"] == "proleptic_gregorian"

    # ---- lon / lat ---------------------------------------------------
    np.testing.assert_array_equal(root["lon"][:], sample.lon.values)
    np.testing.assert_array_equal(root["lat"][:], sample.lat.values)
    assert root["lon"].shape == (TEST_NLON,)
    assert root["lat"].shape == (TEST_NLAT,)
    # Attrs survive the round-trip.
    for k, v in LON_ATTRS.items():
        assert root["lon"].attrs[k] == v
    for k, v in LAT_ATTRS.items():
        assert root["lat"].attrs[k] == v

    # ---- root / global attrs ----------------------------------------
    for k, v in ROOT_ATTRS.items():
        assert root.attrs[k] == v

    # ---- four data variables exist with the expected shape ----------
    expected_vars = {
        "precipitation": ("float32", np.float32(0.0)),
        "randomError": ("float32", np.float32(0.0)),
        "precipitationQualityIndex": ("float32", np.float32(0.0)),
        "probabilityLiquidPrecipitation": ("int16", np.int16(0)),
    }
    for name, (dtype_str, fill) in expected_vars.items():
        arr = root[name]
        assert arr.shape == (TEST_N_TIME, TEST_NLON, TEST_NLAT), name
        assert str(arr.dtype) == dtype_str, name
        # chunks[0] is always 1 (one timestep per chunk).
        assert arr.chunks[0] == 1, name
        # lon-chunk sizes track the fixture: 24-chunk vars → 6 (nlon/2),
        # the 12-chunk var → nlon.
        if name == "probabilityLiquidPrecipitation":
            assert arr.chunks[1] == TEST_NLON, name
        else:
            assert arr.chunks[1] == TEST_NLON // 2, name
        # Fill value carried through from the HDF5 dataset.
        # Use numpy comparison so dtype subtleties are handled.
        assert np.array(arr.fill_value).astype(arr.dtype) == np.array(fill).astype(arr.dtype), name
        # A couple of representative attrs.
        assert arr.attrs["DimensionNames"] == "time,lon,lat", name


def test_initialize_repo_is_idempotent(
    test_repo: icechunk.Repository, sample: xr.Dataset
) -> None:
    """Calling ``initialize_repo`` twice on the same repo must be a no-op the
    second time — VDP's GC lambda also calls it.
    """
    initialize_repo(
        repo=test_repo,
        sample=sample,
        n_time=TEST_N_TIME,
        t0=TEST_T0,
        time_chunk=TEST_TIME_CHUNK,
    )
    assert _is_initialized(test_repo)

    # Snapshot id of the latest commit on main.
    head_before = next(iter(test_repo.ancestry(branch="main"))).id

    # Second call should detect "already initialized" and return without
    # creating a new commit.
    initialize_repo(
        repo=test_repo,
        sample=sample,
        n_time=TEST_N_TIME,
        t0=TEST_T0,
        time_chunk=TEST_TIME_CHUNK,
    )
    head_after = next(iter(test_repo.ancestry(branch="main"))).id
    assert head_before == head_after, "second initialize_repo created a new commit"
