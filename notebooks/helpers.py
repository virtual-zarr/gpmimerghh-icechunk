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
drop_variables = ["Intermediate", "nv", "lonv", "latv"]
all_coords = ["time", "lon", "lat"]
coord_bnds = "time_bnds", "lon_bnds", "lat_bnds"
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


def open_vds(data_url: str):
    data_prefix_url, filename = data_url.rsplit("/", 1)
    store = S3Store.from_url(data_prefix_url, credential_provider=cp)
    registry = ObjectStoreRegistry({f"s3://{bucket}": store})    
    parser = HDFParser(
        group="Grid",
        drop_variables=drop_variables,
    )
    return open_virtual_dataset(
      url=data_url,
      parser=parser,
      registry=registry,
      loadable_variables=all_coords + coord_bnds,
      #decode_times=False
    )

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