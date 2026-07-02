# Silicon Entropy Engine — build the C engine stages and the kernel microbenchmarks.
#
# Default target = Zen 2 (native performance, the project's reference hardware).
# For a portable build (e.g. CI on a non-Zen runner) override ARCH:
#     make ARCH=-march=x86-64-v3
#
# The engine parity gates (bin/e1_engine --all, etc.) additionally need the release
# asset (checkpoints + exports + tokenizer + validation slice); see docs/REPRODUCE.md.
# `make test` runs only the weight-free kernel self-tests.

CC      ?= clang
ARCH    ?= -march=znver2
CFLAGS   = -O3 -mavx2 $(ARCH)
LDLIBS   = -lm
BIN      = bin

ENGINES    = e1 e2 e3 e35 e4
MICROBENCH = lutbench cachesweep

.PHONY: all engines microbench test charts clean

all: engines microbench

engines:    $(addprefix $(BIN)/,$(addsuffix _engine,$(ENGINES)))
microbench: $(addprefix $(BIN)/,$(MICROBENCH))

$(BIN)/%_engine: benchmarks/phase60/%_engine.c | $(BIN)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BIN)/lutbench: benchmarks/phase57/phase57_lutbench.c | $(BIN)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BIN)/cachesweep: benchmarks/phase57/phase57_cachesweep.c | $(BIN)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BIN):
	mkdir -p $(BIN)

# Weight-free self-test: the LUT kernels self-check bit-exactness vs a scalar reference.
test: microbench
	$(BIN)/lutbench

charts:
	python scripts/make_readme_charts.py

clean:
	rm -rf $(BIN)
