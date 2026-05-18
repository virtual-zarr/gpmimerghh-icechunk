# GPM IMERG HH → Virtual Icechunk Store — Design Doc

A virtual Icechunk store covering the full GPM IMERG Half-Hourly (HH) Final Precipitation record, built with Lithops + AWS Lambda on top of Icechunk 2.x.

## Goals

- Produce a single ARCO data cube spanning the full GPM IMERG HH record (1998-01-01 → 2024-12-31).
- Use serverless compute (Lithops on AWS Lambda) for parallel virtual-reference generation — no persistent cluster.
- Keep individual chunk manifests under ~500 MB by using Icechunk 2.x manifest splitting from the first write.
- Make `xr.open_zarr(...)` cheap regardless of total dataset size — no full-manifest load on open.
- Stay close enough to Icechunk + VirtualiZarr primitives that "rechunk to native Icechunk later" is straightforward.

## About the dataset

- Official name: **GPM IMERG Final Precipitation L3 Half Hourly 0.1° × 0.1° V07 (GPM_3IMERGHH)** at GES DISC.
- S3 bucket: `s3://gesdisc-cumulus-prod-protected/GPM_L3/GPM_3IMERGHH.07/` (us-west-2, NASA requester-pays via Earthdata STS).
- One HDF5 file per 30-min interval, 48 files/day, 473,328 files through end of 2024.
- Filenames are **deterministic** from a Python `datetime`: see [`notebooks/helpers.py#url_for`](./notebooks/helpers.py)

This determinism is load-bearing: it lets every worker compute the time index of any file from its filename alone, without ever opening the file. That's what makes region writes viable (see §6).

## Single granule structure

![Single granule cube](diagrams/single_file_cube.svg)

Each HDF5 file contains a `Grid` group (plus a `Grid/Intermediate` group we drop). The Grid group has dimensions `(time=1, lon=3600, lat=1800)` and the variables below. The cube view above shows the lon-chunking of the 24-chunk variables on the front face — the 12-chunk array (`probabilityLiquidPrecipitation`) chunks the same axis half as finely.

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

**Quirks worth knowing:**

- `_FillValue` is not present as an attribute on the HDF5 dataset; it lives in `CodeMissingValue`. VirtualiZarr's HDF parser promotes this for Zarr v3 encoding only on the `fix/no-fillvalue-attribute` branch (or release ≥ whatever lands first).
- `drop_variables = ["Intermediate", "nv", "lonv", "latv"]`.
- Use `HDFParser(group="Grid")` and pass all coordinate names as `loadable_variables` so they're materialised natively in Icechunk and not stored as virtual refs.

## Final virtual Icechunk store

![Final store long cube](diagrams/full_store_cube.svg)

Conceptually the store is one root group with:

- **Four virtual data arrays**, each of shape `(473328, 3600, 1800)`. Every chunk is a virtual ref pointing at a byte range inside an HDF5 file on GES DISC's S3.
- **Native coordinate arrays.** `time` is the only sizable one — 473,328 int64 values, rechunked to 17,520 per chunk (one year per chunk) so coord reads stay cheap. `lon` and `lat` are single small chunks.
- **No `Intermediate` group, no `nv/lonv/latv`.**

Per data array:

```
chunks(precipitation, randomError, precipitationQualityIndex)
  = 473,328 × 24 = 11,359,872 each
chunks(probabilityLiquidPrecipitation)
  = 473,328 × 12 = 5,679,936

total virtual chunks = 39,759,552
```

## Manifest sharding strategy

This is the lesson from the v1 attempt: a single monolithic manifest does not scale. At 11 years of data, v1 produced a ~3 GB manifest that had to be fully downloaded on every open and every append. We have to split.

**Strategy.** Split per-array, by chunk position along `time`, with one shard per year (`17,520` half-hours).

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

- **27 shards per array × 4 arrays = 108 data manifests.**
- ~420,480 refs per shard (17,520 timesteps × 24 lon-chunks).
- Roughly **80–200 MB per shard** in Icechunk 2.x's Arrow-style manifest format.
- A small, separately-stored coordinate manifest (≪ 10 MB).

**Why this works.** Opening the store with `xr.open_zarr` only needs the array metadata and the coordinate manifest. Reading a slice loads exactly the shard(s) covering that time range, in parallel. Appending a new year touches one shard per array, not the whole record.

**Critical**: splitting must be set on the `RepositoryConfig` *before the first write*. Setting it after produces a monolithic manifest first, then a separately-split set, which defeats the point. If you ever need to retrofit, `rewrite_manifests` lets you re-split an existing repo at the cost of one rewrite.

## Build architecture: Lithops + region writes

### Why region writes (instead of concat + append)

Two facts about IMERG make region writes a much cleaner architecture than the staged-tree-reduce + serial-append pattern from the v1 plan:

1. **The time coordinate is deterministic from the filename.** Worker N can compute its own time index without reading anything.
2. **Each file maps to exactly one time slice with no overlap.** File at time `t` writes to `region={"time": slice(t, t+1)}`; workers never collide.

That collapses the build pipeline to: initialize once → write all regions in parallel → merge sessions → commit once. No intermediate S3 staging, no tree-reduce, no 27 serial yearly commits, no growing-manifest-on-append problem.

### Three-stage architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 0 — Driver (one process, runs once)                           │
│   • Compute full time index (473,328 deterministic values)          │
│   • Open or create repo with ManifestSplittingConfig set            │
│   • Initialize empty arrays at final shape (one virtual placeholder)│
│   • Write coord arrays + commit                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Lithops (fully parallel, ~ thousands of Lambdas)          │
│   • Each Lambda owns a time range (e.g. one day = 48 files)         │
│   • Authenticates to Earthdata, fetches short-lived S3 STS creds    │
│   • For each file: open_virtual_dataset → write to its region       │
│       in a forked icechunk session                                  │
│   • Returns the session's diff (small)                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2 — Driver (one process, once)                                │
│   • Collect all session diffs from Lithops futures                  │
│   • Repository.merge_sessions(diffs)                                │
│   • One commit                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 0 — initialize

See `template_repo.py`

The "init from a single file then `reindex(time=...)`" trick gives you the right dtypes / attrs / encoding without doing the full per-file open. Alternative: pre-build the metadata by hand.

### Stage 1 — Lithops region writes

```python
import lithops
from datetime import datetime, timedelta

def write_day(day_idx: int) -> bytes:
    """Run on AWS Lambda. Writes 48 virtual refs into the icechunk store."""
    repo = open_repo()  # reads RepositoryConfig from object storage
    session = repo.distributed_writable_session("main")

    base_time = datetime(1998, 1, 1) + timedelta(days=day_idx)
    for k in range(48):
        t = base_time + timedelta(minutes=30 * k)
        time_idx = day_idx * 48 + k

        vds = open_virtual_dataset(
            url_for(t),
            parser=HDFParser(group="Grid", drop_variables=DROP),
            registry=registry,
            loadable_variables=[],  # coords already written in Stage 0
        )
        vds.vz.to_icechunk(
            session.store,
            region={"time": slice(time_idx, time_idx + 1)},
        )

    return session.fork()  # serialized diff, ~tens of KB

fexec = lithops.FunctionExecutor(
    runtime_memory=2048,
    runtime_timeout=900,
    region="us-west-2",
)
N_DAYS = 9_860  # ≈ 27 × 365.25
futures = fexec.map(write_day, range(N_DAYS))
diffs = fexec.get_result(futures)
```

Tuning knobs:

- **Files per Lambda.** A day (48 files) is a sensible default. Smaller batches = more Lambdas but lower per-Lambda risk; bigger batches = fewer, longer-running Lambdas. Stay well under the 15-min Lambda ceiling.
- **`runtime_memory`.** 2 GB is plenty for 48 files; bumping to 4 GB just buys faster CPUs.
- **`max_workers`.** Cap if Lambda concurrency limits would otherwise be hit (default account limit is 1,000 concurrent — request increase if you want to land all 9,860 Lambdas in one wave).

### Stage 2 — merge & commit

```python
session = repo.writable_session("main")
for diff in diffs:
    session.merge(diff)
session.commit(
    f"build: full GPM IMERG HH record, {N_DAYS} day-batches"
)
```

The merge is cheap — each forked session knows only the chunks *it* wrote, so the coordinator processes ~84 refs per worker, ~830k total. The actual byte writes already happened during Stage 1; this step is metadata bookkeeping.

## Operational concerns

### Earthdata auth inside Lambda

The NASA bucket needs short-lived STS credentials via `https://data.gesdisc.earthdata.nasa.gov/s3credentials`, which itself needs Earthdata Login. Inside Lambda:

- Store `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` in Lambda env vars (sourced from Secrets Manager or SSM Parameter Store at deploy time).
- Each Lambda calls `NasaEarthdataCredentialProvider(credentials_url)` to fetch its own STS creds. Don't pass STS creds in as task arguments — they expire in ~1 hour and the full job will run longer than that.
- Use the *icechunk-side* `s3_refreshable_credentials(get_credentials=...)` so refreshes happen automatically inside icechunk too.

### Lithops runtime image

The default Lithops runtime won't have `virtualizarr`, `icechunk`, `obstore`, `earthaccess`, `obspec_utils`. Build a custom runtime once:

```bash
lithops runtime build -f Dockerfile gpm-imerg-runtime
lithops runtime deploy gpm-imerg-runtime --memory 2048
```

Pin everything: `icechunk==2.x.y`, `virtualizarr==fix/no-fillvalue-attribute or release`, `obstore`, `earthaccess`.

### Failure handling

Region writes are *idempotent* if you re-run the same Lambda — re-running write_day(k) just overwrites the same chunk refs with the same byte ranges. So the failure protocol is simply: collect failed futures, retry them. No special undo logic.

The one exception: a Lambda that crashed mid-way leaves the time slices it didn't get to as "empty" chunks (which Zarr fills with the configured fill value). A read of that slice returns the fill value, not an error. Validate after the build by scanning for fill-value-heavy regions.

### Cost (order of magnitude)

- ~10,000 Lambda invocations × ~30 s average × 2 GB ≈ a few dollars of Lambda runtime.
- S3 requests: ~473k HEAD + small GETs against `gesdisc-cumulus-prod-protected` — within same region, free egress, requests are sub-cent.
- Icechunk store storage: the manifests come to ~20–40 GB total; coords + metadata are small. < $1/month at S3 standard.
- Total: low double digits of dollars for the whole build.

## Memory budget

| stage | workload | peak memory per worker |
|---|---|---|
| Stage 0 init | One driver process, builds 473k-element time array, writes coords | < 500 MB |
| Stage 1 Lambda | 48 files × 84 virtual refs each ≈ 4,032 refs in memory, plus icechunk session state | < 1 GB (2 GB Lambda is comfortable) |
| Stage 2 merge | Driver processes ~10k tiny diffs; sequential merge into one session | < 2 GB |

This compares favourably to the v1 attempt, which hit OOM on append once the in-memory manifest passed ~3 GB at year 11.

## Fallback: staged + serial commits

If `virtualizarr.to_icechunk(..., region=...)` or `distributed_writable_session` don't work as advertised in the versions we pin (see §10), fall back to the staged plan:

1. **Stage 1 (Lithops, parallel):** one Lambda per day. Each Lambda concats its 48 virtual datasets into a one-day VDS, serializes (Arrow / pickle of `ManifestArray`s), writes to `s3://staging/day/YYYY-MM-DD.parquet`.
2. **Stage 2 (Lithops, parallel):** one Lambda per year. Reads 365 day-VDS blobs, concats into a year-VDS, writes to `s3://staging/year/YYYY.parquet`.
3. **Stage 3 (driver, serial):** for each year, read year-VDS, `vds.vz.to_icechunk(repo, append_dim="time")`, commit.

This is the architecture that would have worked in icechunk 1.x had manifest splitting been available. It's still memory-bounded (workers never hold more than one year), but it requires a serial Stage 3 (since icechunk commits to a branch ref are serialized). Wall-clock is bounded by 27 × per-year-commit time instead of `max(Lambda)` + merge.

## Validation / open questions

Before locking in the region-write architecture, verify:

- **`virtualizarr.DataArray.vz.to_icechunk(region=...)` accepts a `region` kwarg** in the pinned version. If not: drop to `icechunk.Session.set_virtual_ref(array_path, [time_idx, lon_idx, lat_idx], ref)` directly and skip the virtualizarr write helper. The VirtualiZarr-built `vds` still gives you the refs; just iterate them.
- **`repo.distributed_writable_session(...)` + `session.fork()` + `session.merge(diff)` exist with those names** (or equivalent) in icechunk 2.x. The docs and changelog around the "distributed sessions" feature are the source of truth.
- **Merge scales to ~10k diffs.** Each diff is small, but the coordinator-side merge needs to deduplicate and write manifest shards. Try with one year (365 diffs) first and measure.
- **Manifest split sizes by measuring.** After year 1 is committed, `du -sh repo/manifests/` and divide by 4 to get per-shard size for one year. If it's > 500 MB, tighten the split (e.g. 8,760 = half-year instead of 17,520).
- **Time-coord chunk size.** 17,520 per chunk means a single coord chunk read pulls one year. If most workflows want short time slices, drop to 1,440 (30 days) — coord storage cost stays trivial either way.

## Implementation sequence

- [ ] **Bring-up:** single-Lambda dry run. One Lambda writes one day's 48 refs into a fresh repo with splitting configured. Open with `xr.open_zarr` and verify.
- [ ] **Year-scale:** run all of 1998 (~365 Lambdas), measure per-shard manifest size, validate read latency on a random slice.
- [ ] **Concurrency stress:** run 5,000 Lambdas concurrently (5 years) and confirm the merge step holds up.
- [ ] **Full build:** all 473,328 files.
- [ ] **Validation:** scan for fill-value-heavy slices indicating failed writes; spot-check 100 random chunks against original HDF5 byte ranges.
- [ ] **Read-performance benchmark:** time-series at a point, global mean at a single timestep, regional subset over 1 year. Compare vs. opening individual HDF5 files.
- [ ] **(Future)** Batch rechunk virtual → native Icechunk for read-heavy use cases. Use the virtual store as the source.

## References

- VirtualiZarr: https://github.com/zarr-developers/VirtualiZarr
- Icechunk: https://icechunk.io
- Lithops: https://lithops-cloud.github.io
- Original notebook (1 + 2 file proof): `test-imerghh-virtualization.ipynb`
- Prior design doc (icechunk 1.x): `icechunk-stores.md`
