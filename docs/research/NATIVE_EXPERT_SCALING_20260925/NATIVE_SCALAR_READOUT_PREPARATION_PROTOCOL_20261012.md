# Original engine scalar-norm readout compile preparation

12 October2026. Compile-only prerequisite for the orthonormal decoding compiler;
not a model/precision/run qualification. No native input or DLL loading here.

Read original engine.c at pinned SHA5f948fc0dcd28b1647a2a2c73067bdc0a76a7d41ae7f34395ede8120aeaa83ce.
Extract hsum256,dotf,matvec,rmsnorm verbatim and record complete body SHA.
Generate named scalar_rmsnorm by only adding epsilon argument/replacing1e-5f
and returning its computed inverse RMS. Scalar accumulation/gamma multiplication
unchanged,original rmsnorm retained. Native readout uses original matvec then
adds bias*inverse_RMS,not an unnormalized constant after head reading.
No original engine file mutation;new code/DLL/log namespace.

Actual configured compiler/flags from qualified prior engine capture:
-O3,-mavx2,-mfma,-march=znver2,-shared,-lm. OMP_PFOR empty. Record compiler extent,
exact argv/exit/source/DLL/log;engine+compiler hashes before/after. Compile45s cap,
CREATE_NO_WINDOW for compiler. No quality,native precision,50 throughput,causal
state or useful-bank admission. Future source/gauge/F32/full-V audit required.
