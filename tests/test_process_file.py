"""Tests for ``process_file`` using a tiny on-disk HDF5 fixture
(no S3, no network).

What this test exercises:
  * The virtual refs land at the time index implied by the timestamp.
  * Reading the cube back returns the fixture's data at the written index.
  * The coord arrays (``time``, ``lon``, ``lat``) are *not* touched
    by the region write.
  * ``time_index_for`` rejects timestamps that aren't 30-minute aligned.

To run:
    pytest tests/test_process_file.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import icechunk
import numpy as np
import pytest
import zarr
from obspec_utils.registry import ObjectStoreRegistry

from tests.hdf5_fixtures import _build_fixture
from notebooks import helpers
from template_repo import initialize_repo
from process_file import process_file, time_index_for

# Small cube to keep tests sub-second.
NLON = 12          # → 2 lon-chunks of size 6 for the 24-chunk vars
NLAT = 6
N_TIME = 48        # one synthetic "day" of half-hours
helpers.T0 = datetime(1998, 1, 1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Fixture file + dict of expected per-variable data."""
    path = tmp_path / "fixture.HDF5"
    expected = _build_fixture(path, nlon=NLON, nlat=NLAT, populate_data=True)
    return path, expected


@pytest.fixture
def initialized_repo(
    tmp_path: Path,
    fixture: tuple[Path, dict[str, np.ndarray]],
    local_registry: ObjectStoreRegistry,
) -> icechunk.Repository:
    """A repo with the Stage-0 template committed, ready for region writes.

    Uses the same fixture file as the sample so the cube shape, dtypes,
    chunk layout, and fill values all match the granule we'll then write.
    """
    fixture_file, _ = fixture
    virtual_chunk_url = f"file://{tmp_path}/"
    repo = helpers.open_or_create_repo(
        storage=icechunk.local_filesystem_storage(path=str(tmp_path / "repo")),
        manifest_split_size=N_TIME,
        virtual_chunk_url=virtual_chunk_url,
        virtual_chunk_store=icechunk.local_filesystem_store(str(tmp_path)),
        # Local filesystem needs no credentials, but the container must still
        # be present in the auth map so icechunk will resolve chunks from it.
        virtual_chunk_credentials={virtual_chunk_url: None},
    )
    sample = helpers.open_vds_with_coords(
        f"file://{fixture_file}",
        registry=local_registry,
    )
    initialize_repo(
        repo=repo,
        sample=sample,
        n_time=N_TIME,
        t0=helpers.T0,
        time_chunk=N_TIME,
    )
    return repo


# ---------------------------------------------------------------------------
# process_file tests
# ---------------------------------------------------------------------------

def test_process_file_writes_at_time_zero(
    initialized_repo: icechunk.Repository,
    fixture: tuple[Path, dict[str, np.ndarray]],
    local_registry: ObjectStoreRegistry,
) -> None:
    """``process_file`` with ``t == T0`` puts the granule into time index 0;
    reading it back returns the fixture's data values exactly.
    """
    fixture_file, expected = fixture

    session = initialized_repo.writable_session("main")
    ok = process_file(
        f"file://{fixture_file}", session, t=helpers.T0, registry=local_registry
    )
    assert ok is True
    session.commit(f"wrote {helpers.T0.isoformat()}")

    read = initialized_repo.readonly_session("main")
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
    initialized_repo: icechunk.Repository,
    fixture: tuple[Path, dict[str, np.ndarray]],
    local_registry: ObjectStoreRegistry,
) -> None:
    """A timestamp 5 half-hours after T0 lands at time index 5 — not 0 —
    and the surrounding indices remain at fill.
    """
    fixture_file, expected = fixture
    t = helpers.T0 + timedelta(minutes=30 * 5)
    expected_idx = time_index_for(t)
    assert expected_idx == 5

    session = initialized_repo.writable_session("main")
    process_file(f"file://{fixture_file}", session, t=t, registry=local_registry)
    session.commit(f"wrote {t.isoformat()}")

    read = initialized_repo.readonly_session("main")
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
        np.full((NLON, NLAT), fill, dtype="float32"),
    )


def test_process_file_does_not_touch_coords(
    initialized_repo: icechunk.Repository,
    fixture: tuple[Path, dict[str, np.ndarray]],
    local_registry: ObjectStoreRegistry,
) -> None:
    """
    Store the coord arrays as in-memory variables before the region write,
    then run ``process_file`` and confirm none of them changed — even though the
    fixture file itself contains its own (single-timestep) ``time``, ``lon``,
    ``lat``, and bounds.
    """
    fixture_file, _ = fixture

    pre = initialized_repo.readonly_session("main")
    pre_root = zarr.open_group(store=pre.store, mode="r")
    time_before = np.asarray(pre_root["time"][:])
    lon_before = np.asarray(pre_root["lon"][:])
    lat_before = np.asarray(pre_root["lat"][:])

    session = initialized_repo.writable_session("main")
    process_file(f"file://{fixture_file}", session, t=helpers.T0, registry=local_registry)
    session.commit("region write")

    post = initialized_repo.readonly_session("main")
    post_root = zarr.open_group(store=post.store, mode="r")
    np.testing.assert_array_equal(np.asarray(post_root["time"][:]), time_before)
    np.testing.assert_array_equal(np.asarray(post_root["lon"][:]), lon_before)
    np.testing.assert_array_equal(np.asarray(post_root["lat"][:]), lat_before)


# ---------------------------------------------------------------------------
# time_index_for unit tests
# ---------------------------------------------------------------------------

def test_time_index_for_aligned() -> None:
    assert time_index_for(helpers.T0) == 0
    assert time_index_for(helpers.T0 + timedelta(minutes=30)) == 1
    assert time_index_for(helpers.T0 + timedelta(days=1)) == 48
    # Same answer regardless of seconds → expressed via aligned datetime.
    assert time_index_for(datetime(1998, 1, 2, 12, 30)) == 73


def test_time_index_for_misaligned() -> None:
    with pytest.raises(ValueError, match="30-minute"):
        time_index_for(helpers.T0 + timedelta(minutes=15))
    with pytest.raises(ValueError, match="30-minute"):
        time_index_for(helpers.T0 + timedelta(seconds=30))


def test_time_index_for_before_epoch_raises() -> None:
    with pytest.raises(ValueError, match="before the cube epoch"):
        time_index_for(helpers.T0 - timedelta(minutes=30))
