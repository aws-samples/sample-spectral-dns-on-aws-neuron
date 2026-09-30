# Validation runs

Retained output of `scripts/bootstrap.sh` on fresh instances, one directory per instance type, grid and rank count:
the bootstrap log, the `CHECK` lines and the run CSV. These are the artefacts behind the validation table in the
README; regenerate one with `./scripts/bootstrap.sh` (or `make check GRID=<N> RANKS=<R>`) on a Neuron instance.
