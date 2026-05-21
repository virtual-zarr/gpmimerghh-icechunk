"""Initialize the GPM IMERG HH virtual icechunk store.

Stage 0 of the pipeline described in design.md: create empty arrays at the
full target shape with the native HDF5 chunk grid, plus the small native
coords. No dask, no xarray — everything goes through zarr's lower-level API.

After this commits, Stage 1 Lambdas can write virtual refs into
``region={"time": slice(t, t+1)}`` and the chunk grids align 1:1 with the
source files.

This script is for a fresh init. To re-init, delete the local
``gpmimerg_hh_07`` directory first.
"""
import numpy as np
import zarr
import icechunk as ic
from datetime import datetime, timedelta
from notebooks import helpers

# ---------------------------------------------------------------------------
# Target time axis
# ---------------------------------------------------------------------------
N_TIME = 486_480
t0 = datetime(1998, 1, 1)
time = np.array(
    [t0 + i * timedelta(minutes=30) for i in range(N_TIME)],
    dtype="datetime64[ns]",
)

# Coord chunk size = one leap-year of half-hours. Matches the manifest split
# size below so "one year of coord reads" and "one year of data reads" both
# resolve to a single chunk / a single shard respectively.
TIME_CHUNK = 48 * 366  # 17_568

# ---------------------------------------------------------------------------
# Repo + manifest splitting config
# ---------------------------------------------------------------------------
repo = helpers.open_or_create_repo()

# ---------------------------------------------------------------------------
# Pull metadata from one sample file
# ---------------------------------------------------------------------------
sample = helpers.open_vds(helpers.url_for(t0))
nlon = sample.sizes["lon"]
nlat = sample.sizes["lat"]


def _native_chunks(var) -> tuple:
    """Best-effort chunk-shape lookup for a virtualizarr-loaded variable."""
    enc_chunks = var.encoding.get("chunks") or var.encoding.get("preferred_chunks")
    if enc_chunks:
        return tuple(enc_chunks)
    if hasattr(var.data, "chunks") and var.data.chunks is not None:
        return tuple(var.data.chunks)
    raise ValueError(f"Couldn't determine native chunk shape for variable")

# ---------------------------------------------------------------------------
# Write everything directly via zarr — one open_group, no xarray
# ---------------------------------------------------------------------------
session = repo.writable_session("main")
root = zarr.open_group(store=session.store, mode="w-")

# time: int64 nanoseconds since epoch; CF attrs let xarray decode on read
root.create_array(
    "time",
    shape=(N_TIME,),
    dtype="int64",
    chunks=(TIME_CHUNK,),
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

# Data variables — metadata + fill value only, no chunk data written
for name, var in sample.data_vars.items():
    src_chunks = _native_chunks(var)
    assert len(src_chunks) == 3, f"{name}: expected 3-D chunks, got {src_chunks}"
    chunks = (1, src_chunks[1], src_chunks[2])
    arr = root.create_array(
        name=name,
        shape=(N_TIME, nlon, nlat),
        chunks=chunks,
        dtype=var.dtype,
        # TODO: need to check this works
        fill_value=var.manifest.data.metadata.fillvalue,
        dimension_names=("time", "lon", "lat"),
        serializer=var.data.metadata.codecs[0],
        compressors=var.data.metadata.codecs[1:],
    )
    arr.attrs.update(dict(var.attrs))

session.commit("init: full shape + coords + native chunk grid (no dask, no xarray)")
print("Committed initial template.")

# ---------------------------------------------------------------------------
# Verify: re-open in a fresh read session and dump shape/chunks/dtype/fill
# ---------------------------------------------------------------------------
read_session = repo.readonly_session("main")
root_r = zarr.open_group(store=read_session.store, mode="r")

print()
for name in sorted(root_r.array_keys()):
    arr = root_r[name]
    fill = arr.fill_value
    print(f"  {name}: shape={arr.shape} chunks={arr.chunks} dtype={arr.dtype} fill={fill}")
