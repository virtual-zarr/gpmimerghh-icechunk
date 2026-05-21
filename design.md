# GPM IMERG HH Virtual Icechunk Store Design Doc

## Overivew

This document describes the design for a cloud-based data pipeline to deliver a virtual Icechunk store covering the full [GPM IMERG Final Precipitation L3 Half Hourly 0.1 degree x 0.1 degree V07 (GPM_3IMERGHH)](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary) record.

## Goals

- Produce a single analysis-ready cloud-optimized (ARCO) data cube spanning the full GPM_3IMERGHH dataset record (1998-01-01 to 2025-09-30).
- Use `virtualizarr-data-pipelines` (SQS + Lambda) for parallel virtual-reference generation + commit to icechunk.
- Keep individual chunk manifests under ~500 MB by using [Icechunk manifest splitting](https://icechunk.io/en/stable/guides/performance/#splitting-manifests) to make `xr.open_zarr(...)` cheap regardless of total dataset size.
- Use region writing to enable unsequenced parallelism to virtualize the entire dataset.

## Non-goals

At time of writing, it is _not_ a goal to create an ongoing icechunk store using the late or early run GPM IMERG product. This may change if it is determined such an ongoing dataset would be useful to users.

The [first FAQ on this page](https://gpm.nasa.gov/data/imerg) describes the different (final, early, and late) GPM IMERG products.

## About the dataset

- Official name: **GPM IMERG Final Precipitation L3 Half Hourly 0.1° × 0.1° V07 (GPM_3IMERGHH)** at GES DISC.
- S3 bucket: `s3://gesdisc-cumulus-prod-protected/GPM_L3/GPM_3IMERGHH.07/` (us-west-2, NASA requester-pays via Earthdata STS).
- One HDF5 file per 30-min interval, 48 files/day, 486,480 files through September 2025.

## Single granule structure

![Single granule cube](diagrams/single_file_cube.svg)

Each HDF5 file contains a `Grid` group (plus a `Grid/Intermediate` group that is dropped). The Grid group has dimensions `(time=1, lon=3600, lat=1800)` and the variables below. The cube view above shows the lon-chunking of the 24-chunk variables on the front face — the 12-chunk array (`probabilityLiquidPrecipitation`) chunks the same axis half as finely.

| variable | dtype | shape | chunk shape | num chunks |
|---|---|---|---|---|
| precipitation | float32 | (1, 3600, 1800) | (1, 145, 1800) | 24 |
| randomError | float32 | (1, 3600, 1800) | (1, 145, 1800) | 24 |
| precipitationQualityIndex | float32 | (1, 3600, 1800) | (1, 145, 1800) | 24 |
| probabilityLiquidPrecipitation | int16 | (1, 3600, 1800) | (1, 291, 1800) | 12 |
| time | int32 | (1,) | (32,) | (loaded natively) |
| lon | float32 | (3600,) | (3600,) | (loaded natively) |
| lat | float32 | (1800,) | (1800,) | (loaded natively) |
| time_bnds, lon_bnds, lat_bnds | — | small | small | (loaded natively) |

**Per file:** `24 + 24 + 24 + 12 = 84 virtual chunks`, 6 coordinate / bounds arrays.

**Fill Value Issue:**

There are 2 fill value concepts for HDF5 virtual zarr datasets. They are well-detailed [in this VirtualiZarr documentation](https://virtualizarr.readthedocs.io/en/stable/custom_parsers.html#fill-values). The first concept, the "value for uninitialized chunks - (e.g., Zarr fill_value)", is typically parsed from the HDF5 `fill_value` attribute. This attribute is not set on GPM IMERG HH files. A fallback has been introduced in VirtualiZarr but not yet released. That is why, at time of writing, this repository uses the `virtualizarr[hdf] @ git+https://github.com/zarr-developers/virtualizarr.git@fix/problem_fillvalues` branch of VirtualiZarr.

The second fill value concept, the "sentinel value - (e.g., CF _FillValue ))" is present on the HDF5 datasets via its attributes. For example, `_FillValue` and `CodeMissingValue` are present on the `precipitation` HDF5 dataset as `-9999.9`.

## Drop variables, load variables

The `Intermediate` group plus the auxiliary `nv`, `lonv`, `latv` dimension variables are always dropped — they aren't useful at the analysis-ready cube level.

Coordinates (`time`, `lon`, `lat`) and bounds (`time_bnds`, `lon_bnds`, `lat_bnds`) are handled **differently in Stage 0 vs Stage 1** because we use region writes. The store has to be initialized with full-length, non-virtual coord arrays *before* any region is written; per-file region writes must then leave those coords alone.

**Stage 0 — initialize repo (runs once).** A single granule is opened with `loadable_variables=["time","lon","lat","time_bnds","lon_bnds","lat_bnds"]` so the coordinate values are materialised in memory. We then write the *full-length* coord arrays (`time: 486480`, `lon: 3600`, `lat: 1800`) directly via the zarr API. See [`template_repo.py`](template_repo.py). This is the non-VirtualiZarr initialization step that region writes require: the coord arrays exist at their final shape before any region is written.

**TODO:** Test this script returns the expected empty icechunk store.

**Stage 1 — per-file region writes.** Each worker opens its granule with **no coords or bounds at all** and writes the resulting vds straight into `region={"time": slice(t, t+1)}`. If the per-file `time`, `lon`, or `lat` were left in the dataset, the region write would attempt to overwrite the Stage 0 coord arrays for every file — either clobbering values cell-by-cell or raising a conflict. Bounds would do the same. So the per-file vds passes `loadable_variables=[]` and `drop_variables=["Intermediate","nv","lonv","latv","time","lon","lat","time_bnds","lon_bnds","lat_bnds"]` to `open_virtual_dataset` — the HDF parser never even reads the coord bytes, and there's no post-hoc `drop_vars` step. This is encapsulated by `helpers.open_vds_data_only`, which is what [`write_day.py`](write_day.py) uses; the Stage 0 equivalent is `helpers.open_vds_with_coords`.

## Final virtual Icechunk store

![Final store long cube](diagrams/full_store_cube.svg)

Conceptually the store is one root group with:

- **Four virtual data arrays**, each of shape `(time: 486480, lon: 3600, lat: 1800)`. Chunks are virtual referents to a byte range of an HDF5 file on GES DISC's S3.
- **Native coordinate arrays.** `time` is the only sizable one — 486,480 int64 values, rechunked to 17,520 per chunk (one year per chunk) so coord reads stay cheap. `lon` and `lat` are single small chunks.

Per data array:

```
number of chunks for each precipitation, randomError, precipitationQualityIndex = 486,480 × 24 = 11,675,520
number of chunks for probabilityLiquidPrecipitation = 486,480 × 12 = 5,679,936

total virtual chunks = 40,864,320
```

## Manifest sharding strategy

This is a lesson [from the store created with icechunk v1 over a year ago](https://github.com/earth-mover/icechunk-nasa/blob/main/design-docs/icechunk-stores.md): a single monolithic manifest does not scale. At 11 years of data, v1 produced a ~3 GB manifest that had to be fully downloaded on every open and every append.

**Strategy.** Split manifests 1 per year per array. Use the chunk position along `time`, with one shard per year. NB: This split will not be perfectly aligned with years because of leap years.

```python
import icechunk as ic

splitting = ic.ManifestSplittingConfig(
    split_sizes={
        "precipitation":                  {"time": 17520},
        "randomError":                    {"time": 17520},
        "precipitationQualityIndex":      {"time": 17520},
        "probabilityLiquidPrecipitation": {"time": 17520},
    }
)
preload = ic.ManifestPreloadConfig(max_total_refs=0)  # don't eagerly load data manifests

config = ic.RepositoryConfig.default()
config.manifest = ic.ManifestConfig(splitting=splitting, preload=preload)
```

This produces:

**TODO:** Verify the size estimates

- **28 shards per array × 4 arrays = 112 data manifests.**
- ~420,480 refs per shard (17520 timesteps × 24 lon-chunks).
- Roughly **80–200 MB per shard** in Icechunk 2.x's Arrow-style manifest format.
- A small, separately-stored coordinate manifest (<10 MB).

**Why this works.** Opening the store with `xr.open_zarr` only needs the array metadata and the coordinate manifest. Reading a slice loads exactly the shard(s) covering that time range, in parallel. Appending a new year touches one shard per array, not the whole record.

**Critical**: Splitting must be set on the `RepositoryConfig` *before the first write*. If you ever need to retrofit, `rewrite_manifests` lets you re-split an existing repo at the cost of one rewrite.

# Cloud architecture: `virtualizarr-data-pipelines`

## Why `virtualizarr-data-pipelines`

Generating the GPM IMERG half-hourly virtual icechunk store will require writing references for 40 million chunks. This number of chunk writes introduces the potential for various issues which `virtualizarr-data-pipelines` (VDP) is designed to solve:

1. One (1) commit per day of data will generate about 10,000 snapshots (28 years * 365 days / year). While the final configuration of chunks/files-per-commit is TBD, with `virtualizarr-data-pipelines` we will be able to garbage collect snapshots we no longer need.
2. Given the large number of chunks, we need to be able to set concurrency limits and batch the processing of files + associated commits. Batching will reduce the number of total commits and batching + concurrency limits will reduce the likelihood of conflicts. VDP supports setting a concurrency limit and batch processing.
3. Given there may still be conflicts, VDP supports retrying through a dead-letter queue with retry configuration error handling through a dead-letter queue redrive and logging.

## Knobs

* `GARBAGE_COLLECTION_FREQUENCY`: What should this be?
* `BATCH_SIZE`: The batch size will determine how many commits + associated snapshots are created. Perhaps 50 is a good starting point?
* 

## Why region writes (instead of append)

IMERG filenames are **deterministic** (see [`notebooks/helpers.py#url_for`](./notebooks/helpers.py)) and each file maps to exactly one time slice. Any worker can compute a file's time index from its filename alone, so writes parallelize across all files without opening them and without commit collisions. Region writes are safer since they are idempotent and serially appending has the added risk and complication of ensuring the append is happening in the right order (not skipping or duplicating existing indices).

That collapses the build pipeline to: initialize the repo with complete coordinates --> write all regions in parallel. 

## Three-stage architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 0 — Initialize repo (one process, runs once)                  │
│   • Compute full time index                                         │
│   • Open or create repo with ManifestSplittingConfig set            │
│   • Initialize empty arrays at final shape using example file       │
│   • Write coord arrays + commit                                     │
│   • Idempotent: subsequent calls (e.g. from VDP's GC lambda) no-op  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Dispatch messages to the SQS queue (runs once)            │
│   • Enumerate every (year, day, half-hour) in the target range      │
│   • Build the corresponding s3:// URL via `helpers.url_for`         │
│   • send_message_batch(10) onto VDP's input queue, in the format    │
│     VDP's `process_notification` handler expects                    │
│   • No bucket listing / S3 inventory needed — filenames are         │
│     deterministic                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2 — Region writes (VDP, runs at SQS scale)                    │
│   • VDP polls SQS in batches of SQS_BATCH_SIZE                      │
│   • Authenticate to Earthdata, fetch short-lived S3 creds (once     │
│     per Lambda cold start, refresh via NasaEarthdataCredentialProvider) │
│   • For each file in the batch:                                     │
│       open_vds_data_only(url)   # coords/bounds excluded at parser  │
│       vds.vz.to_icechunk(session.store, region={"time": slice(...)})│
│   • One commit per batch (not per file)                             │
│   • Failures → DLQ for retry/redrive                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 0 — Initialize

VDP includes a [trigger once](https://github.com/developmentseed/virtualizarr-data-pipelines/blob/main/cdk/stack.py#L157-L178) custom resource for the lambda code [here](https://github.com/developmentseed/virtualizarr-data-pipelines/tree/main/lambda/initialize). Since the `initialize_repo` function also gets called by the `process_messages` lambda, we will setup the repo in a separate function that will also get called by the initialize lambda (but not by the `process_messages` lambda).

**TODO:**

See [`template_repo.py`](template_repo.py) for the sample code for initializing the repo. This will be a new function in `virtualizarr_processor.processor` that will be imported and called in the [initialize handler](https://github.com/developmentseed/virtualizarr-data-pipelines/blob/main/lambda/initialize/handler.py).

### Stage 1 - Dispatch messages to the queue

A script will generate a list of files to process and send those files as messages to the queue. The current VDP [`process_notification`](https://github.com/developmentseed/virtualizarr-data-pipelines/blob/main/lambda/process_messages/handler.py#L21-L42) function expects messages with a certain structure, so the script will push messages which match that structure. 

NB: We cannot enable an inventory since we are not the bucket owners.

**TODOs:** Write and test this new inventory-to-queue script

### Stage 2 — Batching + Region writes

Each file can be written to a region and then committed as part of a "batch", enabled by the [`BatchProcessor`](https://fiserv.github.io/aws-lambda-powertools-python/develop/utilities/batch/)  ([see usage in VDS](https://github.com/developmentseed/virtualizarr-data-pipelines/blob/main/lambda/process_messages/handler.py#L17)) class and configured using `SQS_BATCH_SIZE`. With region writing, the files within a batch do not need to be coordinated.

**TODOS:**
* Integrate the code in [write_day.py](./write_day.py) as the `process_file` function in VDP.

## Additional Requirements

### Earthdata Auth

The NASA bucket needs short-lived S3 credentials via `https://data.gesdisc.earthdata.nasa.gov/s3credentials`, which authorizes via Earthdata Login credentisl. Inside the lambda:

- Store `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` in Lambda env vars (sourced from Secrets Manager or SSM Parameter Store at deploy time).
- Each Lambda calls `NasaEarthdataCredentialProvider(credentials_url)` to fetch its own STS creds. Don't pass STS creds in as task arguments — they expire in ~1 hour and the full job will run longer than that.
- Use the *icechunk-side* `s3_refreshable_credentials(get_credentials=...)` so refreshes happen automatically inside icechunk too.

### Failure handling

Region writes are *idempotent*. If you re-run any set of files, the functionality will just overwrite the same chunk refs with the same byte ranges. So the failure protocol is to collect failed executions and retry them.

### Validation

Validate after building by scanning for any timesteps that have an average of the Zarr fill value.

### Cost (order of magnitude)

**TODO:** Fact check these cost estimates

- ~10,000 Lambda invocations × ~30 s average × 2 GB ≈ a few dollars of Lambda runtime.
- S3 requests: ~473k HEAD + small GETs against `gesdisc-cumulus-prod-protected` — within same region, free egress, requests are sub-1 cent.
- Icechunk store storage: the manifests come to ~20–40 GB total; coords + metadata are small. < $1/month at S3 standard.
- Total: low double digits of dollars for the whole build.

## Memory budget

| stage | workload | peak memory per worker |
|---|---|---|
| Stage 0 — Initialize Repo | One process, builds the 486,480-element time array, opens one sample granule for metadata, writes coord arrays | < 500 MB |
| Stage 1 — Dispatch | Iterates ~486k timestamps, builds URLs, sends SQS messages in batches of 10. No HDF5 reads, no virtual refs in memory. | < 128 MB (Lambda min is comfortable) |
| Stage 2 — Region writes (VDP) | `SQS_BATCH_SIZE` files × 84 virtual refs each, plus icechunk session state for the batch's commit. At batch size 48 ≈ 4,032 refs. | < 1 GB (2 GB Lambda is comfortable) |

## Fallback: staged + serial commits

If `virtualizarr.to_icechunk(..., region=...)` doesn't work as advertised, fall back to writing data serially, using `virtualizarr.to_icechunk(..., append_dim="time")`.

## Implementation sequence

Following the TODOs listed above:

- [ ] **Single-Lambda dry run:** One Lambda writes one day's 48 refs into a fresh repo with splitting configured. Open with `xr.open_zarr` and verify.
- [ ] **Year-scale:** run all of 1998 (~365 Lambdas), measure per-shard manifest size, validate read latency on a random slice.
- [ ] **Concurrency stress:** run 5,000 Lambdas concurrently (5 years) and confirm the merge step holds up.
- [ ] **Full build:** all 473,328 files.
- [ ] **Validation:** scan for fill-value-heavy slices indicating failed writes; spot-check 100 random chunks against original HDF5 byte ranges.
- [ ] **Read-performance benchmark:** time-series at a point, global mean at a single timestep, regional subset over 1 year. Compare vs. opening individual HDF5 files.
- [ ] **(Future)** Batch rechunk virtual → native Icechunk for read-heavy use cases. Use the virtual store as the source.

## References

- VirtualiZarr: https://github.com/zarr-developers/VirtualiZarr
- Icechunk: https://icechunk.io
- Proof of concept: [notebooks/test-imerghh-virtualization.ipynb](./notebooks/test-imerghh-virtualization.ipynb)
- [Prior design doc (icechunk 1.x): `icechunk-stores.md`](https://github.com/earth-mover/icechunk-nasa/blob/main/design-docs/icechunk-stores.md)
- [`virtualizarr-data-pipelines`](https://github.com/developmentseed/virtualizarr-data-pipelines)