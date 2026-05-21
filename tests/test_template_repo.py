"""Tests for ``template_repo.initialize_repo`` using a tiny on-disk HDF5
fixture (no S3, no network).

The fixture mirrors the GPM IMERG HH layout (a ``Grid`` group with the same
variable names and attribute structure) but at a tiny grid size so the test
runs in well under a second.

To run:
    pytest tests/test_template_repo.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import h5py
import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from notebooks import helpers
from template_repo import _is_initialized, initialize_repo

# Test cube dimensions — small enough to be fast, large enough to actually
# exercise multi-chunk lon-chunking.
TEST_NLON = 12        # → 2 chunks of size 6 for the 24-chunk vars
TEST_NLAT = 6
TEST_N_TIME = 48      # one synthetic "day" of half-hours
TEST_T0 = datetime(1998, 1, 1)
TEST_TIME_CHUNK = TEST_N_TIME  # one shard


# ---------------------------------------------------------------------------
# Fixture HDF5 file
# ---------------------------------------------------------------------------

# Attribute payloads we plant in the fixture and assert on round-trip.
ROOT_ATTRS = {
    "Title": "Test fixture mimicking GPM IMERG HH",
    "DOI": "10.5067/GPM/IMERG/3B-HH/test",
    "AlgorithmID": "3IMERGHH",
    "ProductionTime": "2024-01-01T00:00:00Z",
}
TIME_ATTRS = {
    "units": "seconds since 1970-01-01 00:00:00 UTC",
    "calendar": "julian",
    "standard_name": "time",
}
LON_ATTRS = {
    "units": "degrees_east",
    "standard_name": "longitude",
    "axis": "X",
}
LAT_ATTRS = {
    "units": "degrees_north",
    "standard_name": "latitude",
    "axis": "Y",
}
PRECIP_ATTRS = {
    "units": "mm/hr",
    "DimensionNames": "time,lon,lat",
    "CodeMissingValue": "-9999.9",
}


def _build_fixture(path: Path, *, nlon: int = TEST_NLON, nlat: int = TEST_NLAT) -> None:
    """Write an HDF5 file laid out like a single GPM IMERG HH granule.

    The variables, group structure, and attribute names follow the real product
    (``Grid/precipitation``, ``Grid/lon``, ``Grid/Intermediate``, etc.) so that
    ``helpers.open_vds_with_coords`` can parse it identically to production.

    HDF5 dimension scales are set on all coord datasets and attached to every
    data variable so VirtualiZarr can resolve dimension names without falling
    back to ``phony_dim_N`` placeholders.

    Global attributes are placed on the ``Grid`` group because
    ``helpers.HDFParser(group='Grid')`` reads attrs from that group only.
    """
    chunk_lon = max(1, nlon // 2)  # 24-chunk vars get 2 chunks here
    plp_chunk_lon = nlon           # 12-chunk var: chunked twice as coarsely

    with h5py.File(path, "w") as f:
        grid = f.create_group("Grid")
        # Global attrs live on Grid — that's what HDFParser(group='Grid') reads.
        for k, v in ROOT_ATTRS.items():
            grid.attrs[k] = v

        # --- coord dimension scales -------------------------------------
        lon_values = np.linspace(-179.95, 179.95, nlon, dtype="float32")
        lat_values = np.linspace(-89.95, 89.95, nlat, dtype="float32")

        time_v = grid.create_dataset("time", data=np.array([0], dtype="int32"))
        for k, v in TIME_ATTRS.items():
            time_v.attrs[k] = v
        time_v.make_scale("time")

        lon_v = grid.create_dataset("lon", data=lon_values)
        for k, v in LON_ATTRS.items():
            lon_v.attrs[k] = v
        lon_v.make_scale("lon")

        lat_v = grid.create_dataset("lat", data=lat_values)
        for k, v in LAT_ATTRS.items():
            lat_v.attrs[k] = v
        lat_v.make_scale("lat")

        # --- bounds (nv is a dim scale; bounds attach their two scales) -
        nv_v = grid.create_dataset("nv", data=np.arange(2, dtype="int32"))
        nv_v.make_scale("nv")

        time_bnds = grid.create_dataset(
            "time_bnds", data=np.array([[0, 1799]], dtype="int32")
        )
        time_bnds.dims[0].attach_scale(time_v)
        time_bnds.dims[1].attach_scale(nv_v)

        lon_edges = np.linspace(-180.0, 180.0, nlon + 1, dtype="float32")
        lon_bnds = grid.create_dataset(
            "lon_bnds",
            data=np.column_stack([lon_edges[:-1], lon_edges[1:]]),
        )
        lon_bnds.dims[0].attach_scale(lon_v)
        lon_bnds.dims[1].attach_scale(nv_v)

        lat_edges = np.linspace(-90.0, 90.0, nlat + 1, dtype="float32")
        lat_bnds = grid.create_dataset(
            "lat_bnds",
            data=np.column_stack([lat_edges[:-1], lat_edges[1:]]),
        )
        lat_bnds.dims[0].attach_scale(lat_v)
        lat_bnds.dims[1].attach_scale(nv_v)

        # --- dropped: aux dim vars + Intermediate subgroup --------------
        grid.create_dataset("lonv", data=np.arange(2, dtype="int32"))
        grid.create_dataset("latv", data=np.arange(2, dtype="int32"))
        intermediate = grid.create_group("Intermediate")
        intermediate.create_dataset("ignored", data=np.zeros(3, dtype="float32"))

        # --- data variables (virtual) -----------------------------------
        def _add_data(name: str, dtype: str, chunk_lon: int,
                      fillvalue, extra_attrs: dict | None = None):
            ds = grid.create_dataset(
                name,
                shape=(1, nlon, nlat),
                dtype=dtype,
                chunks=(1, chunk_lon, nlat),
                # todo(aimee): this is to mimic the missing fill value in GPM IMERG HH files
                fillvalue=None,
            )
            attrs = dict(PRECIP_ATTRS)
            if extra_attrs:
                attrs.update(extra_attrs)
            attrs["_FillValue"] = fillvalue
            for k, v in attrs.items():
                ds.attrs[k] = v
            # Attach dimension scales so VirtualiZarr resolves dim names.
            ds.dims[0].attach_scale(time_v)
            ds.dims[1].attach_scale(lon_v)
            ds.dims[2].attach_scale(lat_v)

        _add_data("precipitation", "float32", chunk_lon, np.float32(-9999.9))
        _add_data("randomError", "float32", chunk_lon, np.float32(-9999.9))
        _add_data("precipitationQualityIndex", "float32", chunk_lon, np.float32(-9999.9))
        _add_data(
            "probabilityLiquidPrecipitation",
            "int16",
            plp_chunk_lon,
            np.int16(-9999),
            extra_attrs={"units": "percent"},
        )


@pytest.fixture
def fixture_file(tmp_path: Path) -> Path:
    """Path to a freshly-built HDF5 fixture for this test."""
    path = tmp_path / "fixture.HDF5"
    _build_fixture(path)
    return path


@pytest.fixture
def local_registry(tmp_path: Path) -> ObjectStoreRegistry:
    """An ObjectStoreRegistry that resolves file:// URLs anywhere under
    ``tmp_path`` to a LocalStore.
    """
    return ObjectStoreRegistry({f"file://{tmp_path}": LocalStore()})


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
