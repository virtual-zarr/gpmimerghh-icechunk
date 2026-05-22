"""Shared HDF5 fixture helpers for the test suite."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

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


def _build_fixture(
    path: Path,
    *,
    nlon: int = 12,
    nlat: int = 6,
    populate_data: bool = False,
) -> dict[str, np.ndarray] | None:
    """Write an HDF5 file laid out like a single GPM IMERG HH granule.

    When ``populate_data=True``, fills every data variable with random values
    and returns them as a ``{name: array}`` dict for round-trip assertions.
    Otherwise the datasets are left empty and ``None`` is returned.

    ``fillvalue=None`` is used on all data datasets to match real GPM IMERG HH
    files, which carry the fill value only as a ``_FillValue`` attribute.
    """
    chunk_lon = max(1, nlon // 2)
    plp_chunk_lon = nlon

    expected: dict[str, np.ndarray] | None = None
    if populate_data:
        rng = np.random.default_rng(seed=0)
        expected = {
            "precipitation": rng.uniform(0.0, 50.0, size=(1, nlon, nlat)).astype("float32"),
            "randomError": rng.uniform(0.0, 5.0, size=(1, nlon, nlat)).astype("float32"),
            "precipitationQualityIndex": rng.uniform(0.0, 1.0, size=(1, nlon, nlat)).astype("float32"),
            "probabilityLiquidPrecipitation": rng.integers(
                0, 100, size=(1, nlon, nlat), dtype="int16"
            ),
        }

    with h5py.File(path, "w") as f:
        grid = f.create_group("Grid")
        for k, v in ROOT_ATTRS.items():
            grid.attrs[k] = v

        time_v = grid.create_dataset("time", data=np.array([0], dtype="int32"))
        for k, v in TIME_ATTRS.items():
            time_v.attrs[k] = v
        time_v.make_scale("time")

        lon_v = grid.create_dataset(
            "lon", data=np.linspace(-179.95, 179.95, nlon, dtype="float32")
        )
        for k, v in LON_ATTRS.items():
            lon_v.attrs[k] = v
        lon_v.make_scale("lon")

        lat_v = grid.create_dataset(
            "lat", data=np.linspace(-89.95, 89.95, nlat, dtype="float32")
        )
        for k, v in LAT_ATTRS.items():
            lat_v.attrs[k] = v
        lat_v.make_scale("lat")

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

        grid.create_dataset("lonv", data=np.arange(2, dtype="int32"))
        grid.create_dataset("latv", data=np.arange(2, dtype="int32"))
        intermediate = grid.create_group("Intermediate")
        intermediate.create_dataset("ignored", data=np.zeros(3, dtype="float32"))

        def _add_data(name: str, dtype: str, chunk_lon: int,
                      fillvalue, extra_attrs: dict | None = None):
            ds = grid.create_dataset(
                name,
                shape=(1, nlon, nlat),
                dtype=dtype,
                chunks=(1, chunk_lon, nlat),
                fillvalue=None,
            )
            attrs = dict(PRECIP_ATTRS)
            if extra_attrs:
                attrs.update(extra_attrs)
            attrs["_FillValue"] = fillvalue
            for k, v in attrs.items():
                ds.attrs[k] = v
            ds.dims[0].attach_scale(time_v)
            ds.dims[1].attach_scale(lon_v)
            ds.dims[2].attach_scale(lat_v)
            if expected is not None:
                ds[...] = expected[name]

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

    return expected
