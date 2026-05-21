"""Tests for ``write_day.process_file`` using a tiny on-disk HDF5 fixture
(no S3, no network).

The fixture mirrors the GPM IMERG HH layout but at a tiny grid size, and —
unlike the template_repo fixture — it has *real* chunk data in every data
variable so we can assert round-trip equality after ``process_file`` writes
its virtual refs into a region of the cube.

What this test exercises:
  * ``process_file`` opens the granule with all coords + bounds excluded
    (drops never happen post-hoc; they're filtered at the HDF parser).
  * The virtual refs land at the time index implied by the timestamp.
  * Reading the cube back returns the fixture's data at the written index.
  * The Stage-0 coord arrays (``time``, ``lon``, ``lat``) are *not* touched
    by the region write — this is the regression check for the "drop all
    coords" correctness fix.
  * ``time_index_for`` rejects timestamps that aren't 30-minute aligned.

To run:
    pytest tests/test_write_day.py -v
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
from template_repo import initialize_repo
from write_day import process_file, time_index_for

# Small cube to keep tests sub-second.
WD_NLON = 12          # → 2 lon-chunks of size 6 for the 24-chunk vars
WD_NLAT = 6
WD_N_TIME = 48        # one synthetic "day" of half-hours
WD_T0 = datetime(1998, 1, 1)


def _build_fixture_with_data(
    path: Path, *, nlon: int = WD_NLON, nlat: int = WD_NLAT
) -> dict[str, np.ndarray]:
    """Write an IMERG-shaped HDF5 fixture and *populate* the data variables.

    Returns the dict of expected values keyed by variable name (each shaped
    ``(1, nlon, nlat)``) so the round-trip assertions can compare against it.

    Production-shape details preserved: ``Grid`` group, dim scales on
    ``time``/``lon``/``lat`` + ``nv``, bounds attach their scales, the four
    data vars are chunked (24-chunk vars chunk lon in half, the int16 var
    spans the full lon axis to mimic the 12-chunk layout).
    """
    rng = np.random.default_rng(seed=0)
    chunk_lon = max(1, nlon // 2)
    plp_chunk_lon = nlon

    expected: dict[str, np.ndarray] = {
        "precipitation": rng.uniform(0.0, 50.0, size=(1, nlon, nlat)).astype("float32"),
        "randomError": rng.uniform(0.0, 5.0, size=(1, nlon, nlat)).astype("float32"),
        "precipitationQualityIndex": rng.uniform(0.0, 1.0, size=(1, nlon, nlat)).astype("float32"),
        "probabilityLiquidPrecipitation": rng.integers(
            0, 100, size=(1, nlon, nlat), dtype="int16"
        ),
    }

    with h5py.File(path, "w") as f:
        grid = f.create_group("Grid")
        grid.attrs["Title"] = "write_day test fixture"

        # ---- coord dim scales ------------------------------------------
        time_v = grid.create_dataset("time", data=np.array([0], dtype="int32"))
        time_v.attrs["units"] = "seconds since 1970-01-01 00:00:00 UTC"
        time_v.attrs["calendar"] = "julian"
        time_v.make_scale("time")

        lon_v = grid.create_dataset(
            "lon", data=np.linspace(-179.95, 179.95, nlon, dtype="float32")
        )
        lon_v.attrs["units"] = "degrees_east"
        lon_v.make_scale("lon")

        lat_v = grid.create_dataset(
            "lat", data=np.linspace(-89.95, 89.95, nlat, dtype="float32")
        )
        lat_v.attrs["units"] = "degrees_north"
        lat_v.make_scale("lat")

        nv_v = grid.create_dataset("nv", data=np.arange(2, dtype="int32"))
        nv_v.make_scale("nv")

        # ---- bounds (loaded natively, not part of process_file's writes)
        time_bnds = grid.create_dataset(
            "time_bnds", data=np.array([[0, 1799]], dtype="int32")
        )
        time_bnds.dims[0].attach_scale(time_v)
        time_bnds.dims[1].attach_scale(nv_v)

        lon_edges = np.linspace(-180.0, 180.0, nlon + 1, dtype="float32")
        lon_bnds = grid.create_dataset(
            "lon_bnds", data=np.column_stack([lon_edges[:-1], lon_edges[1:]])
        )
        lon_bnds.dims[0].attach_scale(lon_v)
        lon_bnds.dims[1].attach_scale(nv_v)

        lat_edges = np.linspace(-90.0, 90.0, nlat + 1, dtype="float32")
        lat_bnds = grid.create_dataset(
            "lat_bnds", data=np.column_stack([lat_edges[:-1], lat_edges[1:]])
        )
        lat_bnds.dims[0].attach_scale(lat_v)
        lat_bnds.dims[1].attach_scale(nv_v)

        # ---- dropped: aux dim vars + Intermediate subgroup -------------
        grid.create_dataset("lonv", data=np.arange(2, dtype="int32"))
        grid.create_dataset("latv", data=np.arange(2, dtype="int32"))
        intermediate = grid.create_group("Intermediate")
        intermediate.create_dataset("ignored", data=np.zeros(3, dtype="float32"))

        # ---- data variables — populated with real values ---------------
        def _add_data(name: str, dtype: str, chunk_lon: int, fillvalue):
            ds = grid.create_dataset(
                name,
                shape=(1, nlon, nlat),
                dtype=dtype,
                chunks=(1, chunk_lon, nlat),
                fillvalue=fillvalue,
            )
            ds[...] = expected[name]
            ds.attrs["_FillValue"] = fillvalue
            ds.attrs["DimensionNames"] = "time,lon,lat"
            ds.attrs["units"] = "mm/hr" if "precip" in name else "1"
            ds.dims[0].attach_scale(time_v)
            ds.dims[1].attach_scale(lon_v)
            ds.dims[2].attach_scale(lat_v)

        _add_data("precipitation", "float32", chunk_lon, np.float32(-9999.9))
        _add_data("randomError", "float32", chunk_lon, np.float32(-9999.9))
        _add_data("precipitationQualityIndex", "float32", chunk_lon, np.float32(-9999.9))
        _add_data("probabilityLiquidPrecipitation", "int16", plp_chunk_lon, np.int16(-9999))

    return expected


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wd_fixture(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Fixture file + dict of expected per-variable data."""
    path = tmp_path / "fixture.HDF5"
    expected = _build_fixture_with_data(path)
    return path, expected


@pytest.fixture
def wd_registry(tmp_path: Path) -> ObjectStoreRegistry:
    """Registry that resolves file:// URLs under ``tmp_path`` to a LocalStore."""
    return ObjectStoreRegistry({f"file://{tmp_path}": LocalStore()})


@pytest.fixture
def wd_initialized_repo(
    tmp_path: Path,
    wd_fixture: tuple[Path, dict[str, np.ndarray]],
    wd_registry: ObjectStoreRegistry,
) -> icechunk.Repository:
    """A repo with the Stage-0 template committed, ready for region writes.

    Uses the same fixture file as the sample so the cube shape, dtypes,
    chunk layout, and fill values all match the granule we'll then write.
    """
    fixture_file, _ = wd_fixture
    virtual_chunk_url = f"file://{tmp_path}/"
    repo = helpers.open_or_create_repo(
        storage=icechunk.local_filesystem_storage(path=str(tmp_path / "repo")),
        manifest_split_size=WD_N_TIME,
        virtual_chunk_url=virtual_chunk_url,
        virtual_chunk_store=icechunk.local_filesystem_store(str(tmp_path)),
        # Local filesystem needs no credentials, but the container must still
        # be present in the auth map so icechunk will resolve chunks from it.
        virtual_chunk_credentials={virtual_chunk_url: None},
    )
    sample = helpers.open_vds_with_coords(
        f"file://{fixture_file}",
        registry=wd_registry,
    )
    initialize_repo(
        repo=repo,
        sample=sample,
        n_time=WD_N_TIME,
        t0=WD_T0,
        time_chunk=WD_N_TIME,
    )
    return repo


# ---------------------------------------------------------------------------
# process_file tests
# ---------------------------------------------------------------------------

def test_process_file_writes_at_time_zero(
    wd_initialized_repo: icechunk.Repository,
    wd_fixture: tuple[Path, dict[str, np.ndarray]],
    wd_registry: ObjectStoreRegistry,
) -> None:
    """``process_file`` with ``t == T0`` puts the granule into time index 0;
    reading it back returns the fixture's data values exactly.
    """
    fixture_file, expected = wd_fixture

    session = wd_initialized_repo.writable_session("main")
    ok = process_file(
        f"file://{fixture_file}", session, t=WD_T0, registry=wd_registry
    )
    assert ok is True
    session.commit(f"wrote {WD_T0.isoformat()}")

    read = wd_initialized_repo.readonly_session("main")
    root = zarr.open_group(store=read.store, mode="r")

    for name, data in expected.items():
        arr = root[name]
        # data has shape (1, nlon, nlat); the cube has (N_TIME, nlon, nlat)
        # so cube[0] returns shape (nlon, nlat).
        np.testing.assert_array_equal(
            np.asarray(arr[0, :, :]),
            data[0],
            err_msg=f"{name}: round-trip mismatch at time index 0",
        )


def test_process_file_writes_at_nonzero_time(
    wd_initialized_repo: icechunk.Repository,
    wd_fixture: tuple[Path, dict[str, np.ndarray]],
    wd_registry: ObjectStoreRegistry,
) -> None:
    """A timestamp 5 half-hours after T0 lands at time index 5 — not 0 —
    and the surrounding indices remain at fill.
    """
    fixture_file, expected = wd_fixture
    t = WD_T0 + timedelta(minutes=30 * 5)
    expected_idx = time_index_for(t)
    assert expected_idx == 5

    session = wd_initialized_repo.writable_session("main")
    process_file(f"file://{fixture_file}", session, t=t, registry=wd_registry)
    session.commit(f"wrote {t.isoformat()}")

    read = wd_initialized_repo.readonly_session("main")
    root = zarr.open_group(store=read.store, mode="r")

    precip = root["precipitation"]
    # Index 5: fixture data round-trips.
    np.testing.assert_array_equal(
        np.asarray(precip[5, :, :]),
        expected["precipitation"][0],
    )
    # Index 0 is still empty → reads return the fill value everywhere.
    fill = np.float32(precip.fill_value)
    np.testing.assert_array_equal(
        np.asarray(precip[0, :, :]),
        np.full((WD_NLON, WD_NLAT), fill, dtype="float32"),
    )


def test_process_file_does_not_touch_coords(
    wd_initialized_repo: icechunk.Repository,
    wd_fixture: tuple[Path, dict[str, np.ndarray]],
    wd_registry: ObjectStoreRegistry,
) -> None:
    """Regression for the 'drop all coords' fix.

    Snapshot the Stage-0 coord arrays before the region write, then run
    ``process_file`` and confirm none of them changed — even though the
    fixture file itself contains its own (single-timestep) ``time``, ``lon``,
    ``lat``, and bounds.
    """
    fixture_file, _ = wd_fixture

    pre = wd_initialized_repo.readonly_session("main")
    pre_root = zarr.open_group(store=pre.store, mode="r")
    time_before = np.asarray(pre_root["time"][:])
    lon_before = np.asarray(pre_root["lon"][:])
    lat_before = np.asarray(pre_root["lat"][:])

    session = wd_initialized_repo.writable_session("main")
    process_file(f"file://{fixture_file}", session, t=WD_T0, registry=wd_registry)
    session.commit("region write")

    post = wd_initialized_repo.readonly_session("main")
    post_root = zarr.open_group(store=post.store, mode="r")
    np.testing.assert_array_equal(np.asarray(post_root["time"][:]), time_before)
    np.testing.assert_array_equal(np.asarray(post_root["lon"][:]), lon_before)
    np.testing.assert_array_equal(np.asarray(post_root["lat"][:]), lat_before)


# ---------------------------------------------------------------------------
# time_index_for unit tests
# ---------------------------------------------------------------------------

def test_time_index_for_aligned() -> None:
    assert time_index_for(WD_T0) == 0
    assert time_index_for(WD_T0 + timedelta(minutes=30)) == 1
    assert time_index_for(WD_T0 + timedelta(days=1)) == 48
    # Same answer regardless of seconds → expressed via aligned datetime.
    assert time_index_for(datetime(1998, 1, 2, 12, 30)) == 73


def test_time_index_for_misaligned() -> None:
    with pytest.raises(ValueError, match="30-minute"):
        time_index_for(WD_T0 + timedelta(minutes=15))
    with pytest.raises(ValueError, match="30-minute"):
        time_index_for(WD_T0 + timedelta(seconds=30))


def test_time_index_for_before_epoch_raises() -> None:
    with pytest.raises(ValueError, match="before the cube epoch"):
        time_index_for(WD_T0 - timedelta(minutes=30))
