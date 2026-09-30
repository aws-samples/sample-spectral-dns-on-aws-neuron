# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# libnrt drivers. The Neuron runtime headers and library ship with the Neuron SDK under /opt/aws/neuron.
NRT_INC ?= /opt/aws/neuron/include
NRT_LIB ?= /opt/aws/neuron/lib
CFLAGS  ?= -O2 -Wall -Wformat -Wformat-security -fstack-protector-strong -D_FORTIFY_SOURCE=2 -I$(NRT_INC)
LDFLAGS ?= -L$(NRT_LIB) -Wl,-rpath,$(NRT_LIB) -lnrt -lm -pthread

all: tgv_run tgv_fused

tgv_run: tgv_run.c
	gcc $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f tgv_run tgv_fused

tgv_fused: tgv_fused.c
	gcc $(CFLAGS) $< -o $@ $(LDFLAGS)
