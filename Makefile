# Silicon Entropy Engine — build + test entry points.
#
#   make engines    build the five C engines (bin/)
#   make selftest   build + run the synthetic kernel self-tests (NO weights needed — this is what CI runs)
#   make gates      run the full parity-gate ladders (needs the exported weights + ids under results/;
#                   see REPRODUCE.md for how to obtain/regenerate them)
#   make charts     regenerate the README charts (python + matplotlib)
#   make clean
#
# Toolchain per project law (docs/ENGINE_PLAN.md): clang -O3, AVX2/FMA, znver2 tuning, NO -ffast-math.
# The binaries run on any AVX2+FMA x86-64 (znver2 is tuning, not a hard ISA wall).

CC      ?= clang
CFLAGS  ?= -O3 -mavx2 -mfma -march=znver2
LDLIBS   = -lm
BIN      = bin
SRC      = benchmarks/phase60
ARCH     = archive/benchmarks/phase60_stage_engines

ifeq ($(OS),Windows_NT)
EXE = .exe
else
EXE =
endif

# engine = the consolidated core (P4.3, parity-accepted 7/7 bit-identical — results/phase60/p43/).
# The five stage engines are archival parity oracles: kept buildable via `make stage-engines`.
STAGE_ENGINES = $(BIN)/e1_engine$(EXE) $(BIN)/e2_engine$(EXE) $(BIN)/e3_engine$(EXE) \
                $(BIN)/e35_engine$(EXE) $(BIN)/e4_engine$(EXE)
ENGINES = $(BIN)/engine$(EXE) $(STAGE_ENGINES)

.PHONY: all engines stage-engines selftest gates charts clean

all: engines

engines: $(ENGINES)

stage-engines: $(STAGE_ENGINES)

$(BIN)/engine$(EXE): $(SRC)/engine.c | $(BIN)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BIN)/%$(EXE): $(ARCH)/%.c | $(BIN)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BIN):
	mkdir -p $(BIN)

# Synthetic kernel self-tests: bit-exactness of every SIMD kernel vs a scalar integer reference,
# on random ternary weights + int8 activations (e2: dense LUT; e3: full/row-scalar/tile-skip;
# e35: poly-exp vs libm on the documented domain; e4: windowed rows + per-expert blocks).
# No weights, no dataset — runnable on any AVX2 CI runner. e1 is fp32-only (no custom kernel).
selftest: engines
	$(BIN)/engine$(EXE) --kselftest
	$(BIN)/e2_engine$(EXE) --kselftest
	$(BIN)/e3_engine$(EXE) --kselftest
	$(BIN)/e35_engine$(EXE) --kselftest
	$(BIN)/e4_engine$(EXE) --kselftest

# Full parity-gate ladders (E1 G1-G5, E2-E4 rungs). Requires results/phase60/*.bin exports and
# results/phase55/{ids.u16,meta.bin} — see REPRODUCE.md. These enter CI when the weight asset
# is attached to a release the runner can fetch.
gates: engines
	$(BIN)/e1_engine$(EXE) --all
	$(BIN)/e2_engine$(EXE) --all
	$(BIN)/e3_engine$(EXE) --all
	$(BIN)/e35_engine$(EXE) --all
	$(BIN)/e4_engine$(EXE) --all

charts:
	python scripts/make_readme_charts.py

clean:
	rm -rf $(BIN)
