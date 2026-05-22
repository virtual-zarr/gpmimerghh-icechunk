"""Per-file region write into the GPM IMERG HH virtual icechunk store.

Each half-hour granule is opened via ``helpers.open_vds_data_only`` so the
HDF parser only emits the 4 data variables — every coordinate and bounds
variable is excluded *before* the parser reads it. The resulting vds is
written straight into ``region={"time": slice(time_idx, time_idx + 1)}`` with
no further filtering needed.

The ``process_file`` function matches the shape expected by
``virtualizarr_processor.typing.VirtualizarrProcessor.process_file`` and is
the unit of work VDP will batch + commit.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from icechunk import Session

from notebooks import helpers

if TYPE_CHECKING:
    from obspec_utils.registry import ObjectStoreRegistry


def time_index_for(t: datetime) -> int:
    """Half-hour offset from the cube's epoch (1998-01-01 00:00:00 UTC)."""
    delta = t - helpers.T0
    if delta < timedelta(0):
        raise ValueError(f"{t!r} is before the cube epoch {helpers.T0!r}")
    seconds = int(delta.total_seconds())
    if seconds % 1800 != 0:
        raise ValueError(f"{t!r} is not aligned to a 30-minute boundary")
    return seconds // 1800


def process_file(
    file_url: str,
    session: Session,
    *,
    t: datetime | None = None,
    registry: ObjectStoreRegistry | None = None,
) -> bool:
    """Write one half-hour granule into its region of the store.

    Parameters
    ----------
    file_url : str
        Full ``s3://`` URL of the source HDF5 granule.
    session : icechunk.Session
        Writable session. The caller is responsible for committing — this
        function only stages writes, matching the VDP processor contract.
    t : datetime, optional
        The granule's timestamp. If omitted it is parsed from ``file_url``.
    registry : ObjectStoreRegistry, optional
        Forwarded to ``helpers.open_vds_data_only``. Production callers leave
        this ``None`` (default GES DISC S3 registry); tests pass a
        ``LocalStore``-backed registry pointing at a fixture file.
    """
    if t is None:
        t = _timestamp_from_url(file_url)
    time_idx = time_index_for(t)

    # data-only opener: coords + bounds are excluded inside the HDF parser
    # itself, so there's nothing to drop after the fact.
    vds = helpers.open_vds_data_only(file_url, registry=registry)
    vds.vz.to_icechunk(
        session.store,
        region={"time": slice(time_idx, time_idx + 1)},
    )
    return True


def _timestamp_from_url(file_url: str) -> datetime:
    """Parse the start timestamp out of a 3B-HHR... filename.

    Example filename:
      3B-HHR.MS.MRG.3IMERG.20250930-S233000-E235959.1410.V07B.HDF5
    """
    filename = file_url.rsplit("/", 1)[-1]
    # token "20250930-S233000" → date + start-of-half-hour
    date_part, start_part = filename.split(".")[4].split("-")[:2]
    return datetime.strptime(date_part + start_part[1:], "%Y%m%d%H%M%S")
