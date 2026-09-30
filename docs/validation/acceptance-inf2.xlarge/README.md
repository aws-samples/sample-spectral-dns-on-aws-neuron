# Acceptance run

One fresh inf2.xlarge launched with `scripts/launch_ec2.sh` from this tree; the 64^3 row ran from user data
(`scripts/bootstrap.sh`), then `make check GRID=128 RANKS=2` and `make check GRID=256 RANKS=2` in one shell.
Times are UTC on 2026-09-21: instance launch 11:04:18, bootstrap start 11:04:52, 64^3 check passed 11:06:01,
128^3 and 256^3 two-rank checks passed 11:10:33. Launch to the three passing rows: 6 minutes 15 seconds.
