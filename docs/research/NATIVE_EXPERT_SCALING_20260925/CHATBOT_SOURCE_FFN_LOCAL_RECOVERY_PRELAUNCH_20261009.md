# Local recovery prelaunch correction

Static review before any worker/model/optimizer observation found that the
first code freeze c0468bff0b5848cea3323a9172fd710df144c907 cycled offsets even
for the18-row short FITcase. This contradicted the frozen protocol's use of
all18 rows on every short visit. No worker was launched under that binding;
optimizer updates/modelcalls0. Retain the initial binding as UNEXECUTED.

Fix offset=0 for N<=batch64;larger cases retain deterministic cyclic offsets.
Protocol/steps/learningrate/gates remain unchanged. Recompile andfreeze before
launch,then bind to the repair1 binding filename. The source/calibration
implementations andallprevious results remain immutable.

The first launcher invocation for the corrected binding supplied an incorrect
command-line SHA911a1b02... rather than the actual7172fd54... . The launcher
rejected it at its initial hash assertion BEFORE worker/log/directory creation.
Exit1,modelcalls/optimizersteps0;no result namespace or running worker existed.
Retain this invocation error. The corrected launch reads SHA directly from the
actual binding file and uses the same unobserved namespace/code/criteria.
