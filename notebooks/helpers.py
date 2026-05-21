import icechunk
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from typing import Dict
from datetime import datetime
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import S3Store
import earthaccess

from datetime import datetime, timedelta
from obstore.auth.earthdata import NasaEarthdataCredentialProvider

BASE = "s3://gesdisc-cumulus-prod-protected/GPM_L3/GPM_3IMERGHH.07"

# Auxiliary variables that we never want in the analysis-ready cube. The
# `Grid` group plus these dimension/bounds variables are dropped on *every*
# read.
AUX_DROP_VARIABLES = ["Intermediate", "nv", "lonv", "latv"]

# Coordinate + bounds variables. In Stage 0 (template init) these are loaded
# natively so we can extract their values; in Stage 1 (region writes) these
# are dropped because they're already written in the store.
COORD_VARIABLES = ["time", "lon", "lat", "time_bnds", "lon_bnds", "lat_bnds"]

# Back-compat aliases for any external code / notebooks still importing these.
drop_variables = AUX_DROP_VARIABLES
all_coords = ["time", "lon", "lat"]
coord_bnds = ("time_bnds", "lon_bnds", "lat_bnds")
group = "Grid"

example_link = 's3://gesdisc-cumulus-prod-protected/GPM_L3/GPM_3IMERGHH.07/2025/273/3B-HHR.MS.MRG.3IMERG.20250930-S233000-E235959.1410.V07B.HDF5'
data_prefix_url, filename = example_link.rsplit("/", 1)
print(f"Data prefix url: {data_prefix_url}")
bucket = data_prefix_url.split("/")[2]
credentials_url = "https://data.gesdisc.earthdata.nasa.gov/s3credentials"
# Since no NASA Earthdata credentials are specified in this example,
# environment variables or netrc will be used to locate them in order to
# obtain S3 credentials from the URL.
cp = NasaEarthdataCredentialProvider(credentials_url)

def url_for(t: datetime) -> str:
    end = t + timedelta(minutes=29, seconds=59)
    midnight = datetime(t.year, t.month, t.day)
    minutes_since = (t - midnight) // timedelta(minutes=1)
    name = (
        "3B-HHR.MS.MRG.3IMERG."
        + t.strftime("%Y%m%d") + "-S" + t.strftime("%H%M%S")
        + "-E" + end.strftime("%H%M%S")
        + f".{minutes_since:04d}.V07B.HDF5"
    )
    return f"{BASE}/{t.year:04d}/{t.strftime('%j')}/{name}"


def _open_vds(data_url: str, *, drop: list[str], load: list[str]):
    """Internal: open a granule with explicit drop_variables / loadable_variables.

    All public openers below are thin wrappers around this so the parser /
    registry / credential setup lives in exactly one place.
    """
    data_prefix_url, _ = data_url.rsplit("/", 1)
    store = S3Store.from_url(data_prefix_url, credential_provider=cp)
    registry = ObjectStoreRegistry({f"s3://{bucket}": store})
    parser = HDFParser(group="Grid", drop_variables=drop)
    return open_virtual_dataset(
        url=data_url,
        parser=parser,
        registry=registry,
        loadable_variables=load,
        # decode_times=False
    )


def open_vds_with_coords(data_url: str):
    """Stage 0 / exploratory: returns a vds with coords + bounds loaded natively.

    Use this when you need to *read* the coordinate values — e.g. to extract
    time/lon/lat into the Stage 0 template, or when poking at a granule in a
    notebook. Coords come back as concrete numpy arrays; data variables come
    back as VirtualiZarr ManifestArrays.
    """
    return _open_vds(
        data_url,
        drop=AUX_DROP_VARIABLES,
        load=COORD_VARIABLES,
    )


def open_vds_data_only(data_url: str):
    """Stage 1 / region writes: returns a vds with **only** the 4 data variables.

    Every coordinate and bounds variable is added to `drop_variables` so the
    HDF parser never reads them, and `loadable_variables` is empty so nothing
    gets materialised. The result can be written straight into
    ``region={"time": slice(t, t+1)}`` without any post-hoc ``drop_vars``.
    """
    return _open_vds(
        data_url,
        drop=AUX_DROP_VARIABLES + COORD_VARIABLES,
        load=[],
    )


# Back-compat alias. Existing callers (the notebook) get coords + bounds, same
# as before. New code should pick one of the two functions above.
open_vds = open_vds_with_coords

def get_prefix_from_url(url: str) -> str:
    """Extract prefix from URL for icechunk."""
    return url.rsplit("/", 3)[0] + '/'

def get_icechunk_creds(daac: str = 'GES_DISC') -> icechunk.S3StaticCredentials:
    """Get refreshable earthdata credentials for icechunk."""
    auth = earthaccess.login()
    if not auth.authenticated:
        raise PermissionError("Could not authenticate using environment variables")
    creds = auth.get_s3_credentials(daac=daac)
    return icechunk.S3StaticCredentials(
        access_key_id=creds["accessKeyId"],
        secret_access_key=creds["secretAccessKey"],
        expires_after=datetime.fromisoformat(creds["expiration"]),
        session_token=creds["sessionToken"],
    )

def get_container_credentials(
    file_url: str
) -> Dict[str, icechunk.AnyCredential]:
    """Get container credentials for icechunk."""
    return icechunk.containers_credentials(
        {
            get_prefix_from_url(
                file_url
            ): icechunk.s3_refreshable_credentials(
                get_credentials=get_icechunk_creds
            )
        }
    )

def open_or_create_test_icechunk_repo(url: str) -> None:
    """Open an existing icechunk repository."""
    storage = icechunk.local_filesystem_storage(
        path='gpmimerg_hh_07',
    )
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            get_prefix_from_url(url),
            icechunk.s3_store(region="us-west-2"),
        )
    )
    repo = icechunk.Repository.open_or_create(
        config=config,
        storage=storage,
        authorize_virtual_chunk_access=get_container_credentials(
            url
        ),
    )
    return repo

def open_or_create_repo():
    config = icechunk.RepositoryConfig.default()
    manifest_split_size = 48 * 366  # 48 half-hours/day x 366 (leap-year safe)
    time_split_size = {
        icechunk.config.ManifestSplitDimCondition.DimensionName("time"): manifest_split_size
    }
    config.manifest = icechunk.ManifestConfig(
        splitting=icechunk.ManifestSplittingConfig.from_dict({
            icechunk.config.ManifestSplitCondition.name_matches("precipitation"): time_split_size,
            icechunk.config.ManifestSplitCondition.name_matches("randomError"): time_split_size,
            icechunk.config.ManifestSplitCondition.name_matches("precipitationQualityIndex"): time_split_size,
            icechunk.config.ManifestSplitCondition.name_matches("probabilityLiquidPrecipitation"): time_split_size,
        }),
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0),
    )
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            get_prefix_from_url(example_link),
            icechunk.s3_store(region="us-west-2"),
        )
    )

    local_storage = icechunk.local_filesystem_storage(path="gpmimerg_hh_07")
    repo = icechunk.Repository.open_or_create(
        config=config,
        storage=local_storage,
        authorize_virtual_chunk_access=get_container_credentials(example_link),
    )
    repo.save_config()
    return repo