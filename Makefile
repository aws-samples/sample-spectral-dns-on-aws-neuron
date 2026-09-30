# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# neuron-spectral-dns: build, run and check the pseudo-spectral Taylor-Green solver on AWS Neuron.
#   source scripts/env.sh          (once per shell: Neuron venv, compiler target, runtime library path)
#   make check GRID=64 RANKS=1     (export the NEFFs, run to T_END, check against the shipped reference)
#   make check-repeat GRID=256 RANKS=8   (check, then run a second time and require step,t,E,Omega bitwise identical)
# Variables: GRID (64), RANKS (1), T_END (10). Artefacts land in build/n<GRID>_r<RANKS>/.
GRID  ?= 64
RANKS ?= 1
T_END ?= 10
BUILD  = build/n$(GRID)_r$(RANKS)
CSV    = $(BUILD)/tgv_$(GRID)_$(RANKS).csv
CSVB   = $(BUILD)/tgv_$(GRID)_$(RANKS)b.csv

.PHONY: check check-repeat run driver clean

check: $(CSV)
	python3 tools/check.py $(CSV) --grid $(GRID) --ranks $(RANKS) | tee $(BUILD)/check.log

check-repeat: check
	scripts/run.sh $(GRID) $(RANKS) $(T_END) b
	python3 tools/check.py $(CSV) --grid $(GRID) --ranks $(RANKS) --repeat $(CSVB) | tee $(BUILD)/check-repeat.log

run: $(CSV)

$(CSV): driver
	scripts/run.sh $(GRID) $(RANKS) $(T_END)

driver:
	$(MAKE) -C driver

clean:
	rm -rf build
	$(MAKE) -C driver clean
