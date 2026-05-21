"""Initialize the GPM IMERG HH virtual icechunk store."""
from __future__ import annotations

from datetime import datetime, timedelta
from itertools import islice

import icechunk as ic
import numpy as np
import xarray as xr
import zarr

from notebooks import helpers

# ---------------------------------------------------------------------------
# Target time axis
# ---------------------------------------------------------------------------
N_TIME = 486_480
T0 = datetime(1998, 1, 1)

# Coord chunk size = one year of half-hours. Matches the manifest split
# size below so "one year of coord reads" and "one year of data reads" both
# resolve to a single chunk / a single shard respectively.
TIME_CHUNK = 48 * 365  # 17_520


def _native_chunks(var) -> tuple:
    """Best-effort chunk-shape lookup for a virtualizarr-loaded variable."""
    enc_chunks = var.encoding.get("chunks") or var.encoding.get("preferred_chunks")
    if enc_chunks:
        return tuple(enc_chunks)
    if hasattr(var.data, "chunks") and var.data.chunks is not None:
        return tuple(var.data.chunks)
    raise ValueError("Couldn't determine native chunk shape for variable")


def _is_initialized(repo: ic.Repository) -> bool:
    """Has the template commit been made yet?

    A freshly created Icechunk repo has exactly one ancestor (the root /
    "Repository initialized" commit). Once the full empty arrays have been committed
    there are at least two ancestors. We use ``islice(..., 2)`` so we never
    walk the full history — the same trick VDP's sample processor uses.
    """
    return len(list(islice(repo.ancestry(branch="main"), 2))) > 1


def initialize_repo(
    repo: ic.Repository | None = None,
    sample: xr.Dataset | None = None,
    *,
    n_time: int = N_TIME,
    t0: datetime = T0,
    time_chunk: int = TIME_CHUNK,
) -> ic.Repository:
    """Open or create the repo and, on first call only, write the coordinate
    template.

    Returns the open ``Repository``. Safe to call repeatedly — subsequent
    calls just return the existing repo.

    Parameters
    ----------
    repo:
        Pre-opened repository. If ``None``, ``helpers.open_or_create_repo()``
        is called with default (production) arguments.
    sample:
        Sample granule dataset, as returned by ``helpers.open_vds_with_coords``.
        Used to pull coord values, attrs, and per-data-variable chunk shapes /
        fill values / codecs. If ``None``, one is opened from S3 for ``t0``.
    n_time:
        Number of timesteps in the target cube. Defaults to ``N_TIME``.
    t0:
        Start of the target time axis. Defaults to ``T0`` (1998-01-01).
    time_chunk:
        Chunk size along the time axis for the native ``time`` coord.
    """
    if repo is None:
        repo = helpers.open_or_create_repo()

    if _is_initialized(repo):
        print("Repo already initialized; skipping template write.")
        return repo

    time = np.array(
        [t0 + i * timedelta(minutes=30) for i in range(n_time)],
        dtype="datetime64[ns]",
    )

    if sample is None:
        # Pull metadata from one sample file. open_vds_with_coords loads
        # coords + bounds natively (via loadable_variables) so we can read
        # their values here.
        sample = helpers.open_vds_with_coords(helpers.url_for(t0))
    nlon = sample.sizes["lon"]
    nlat = sample.sizes["lat"]

    # Write everything directly via zarr — one open_group, no xarray
    session = repo.writable_session("main")
    root = zarr.open_group(store=session.store, mode="w-")

    # time: int64 nanoseconds since epoch; CF attrs let xarray decode on read
    root.create_array(
        "time",
        shape=(n_time,),
        dtype="int64",
        chunks=(time_chunk,),
        dimension_names=("time",),
    )
    root["time"][:] = time.view("int64")
    root["time"].attrs.update({
        "units": "nanoseconds since 1970-01-01",
        "calendar": "proleptic_gregorian",
    })

    # lon / lat — small enough to write in one shot
    lon_data = sample.lon.values
    root.create_array("lon", shape=lon_data.shape, dtype=lon_data.dtype,
                      chunks=(nlon,), dimension_names=("lon",))
    root["lon"][:] = lon_data
    root["lon"].attrs.update(dict(sample.lon.attrs))

    lat_data = sample.lat.values
    root.create_array("lat", shape=lat_data.shape, dtype=lat_data.dtype,
                      chunks=(nlat,), dimension_names=("lat",))
    root["lat"][:] = lat_data
    root["lat"].attrs.update(dict(sample.lat.attrs))

    # Global attributes
    root.attrs.update(dict(sample.attrs))

    # Data variables — metadata + fill value only, no chunk data written.
    # Bounds (time_bnds, lon_bnds, lat_bnds) are 2-D and may appear in
    # data_vars; skip anything that isn't a 3-D (time, lon, lat) array.
    for name, var in sample.data_vars.items():
        src_chunks = _native_chunks(var)
        if len(src_chunks) != 3:
            continue
        chunks = (1, src_chunks[1], src_chunks[2])
        # import pdb; pdb.set_trace()
        arr = root.create_array(
            name=name,
            shape=(n_time, nlon, nlat),
            chunks=chunks,
            dtype=var.dtype,
            fill_value=var.data.metadata.fill_value,
            dimension_names=("time", "lon", "lat"),
            serializer=var.data.metadata.codecs[0],
            compressors=var.data.metadata.codecs[1:],
        )
        arr.attrs.update(dict(var.attrs))

    session.commit("init: full shape + coords + native chunk grid (no dask, no xarray)")
    print("Committed initial template.")
    return repo


if __name__ == "__main__":
    repo = initialize_repo()

    # Verify: re-open in a fresh read session and dump shape/chunks/dtype/fill
    read_session = repo.readonly_session("main")
    root_r = zarr.open_group(store=read_session.store, mode="r")

    print()
    for name in sorted(root_r.array_keys()):
        arr = root_r[name]
        fill = arr.fill_value
        print(f"  {name}: shape={arr.shape} chunks={arr.chunks} dtype={arr.dtype} fill={fill}")
