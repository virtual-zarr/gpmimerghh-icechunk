from notebooks import helpers
from datetime import datetime, timedelta

def write_day(day_idx: int) -> bytes:
    """Run on AWS Lambda. Writes 48 virtual refs into the icechunk store."""
    repo = helpers.open_or_create_repo()
    session = repo.writable_session("main")

    base_time = datetime(1998, 1, 1) + timedelta(days=day_idx)
    for k in range(48):
        t = base_time + timedelta(minutes=30 * k)
        time_idx = day_idx * 48 + k

        vds = helpers.open_vds(helpers.url_for(t))
        vds = vds.drop_vars(['lon', 'lat'])
        vds.vz.to_icechunk(
            session.store,
            region={"time": slice(time_idx, time_idx + 1)},
        )
    session.commit(f"Wrote day {base_time}")

write_day(0)
