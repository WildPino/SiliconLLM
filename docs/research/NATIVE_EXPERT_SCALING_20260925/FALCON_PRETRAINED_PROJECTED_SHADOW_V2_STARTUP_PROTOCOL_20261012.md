# Shadow v1 startup fault and v2 detached launcher

12 October2026. Original v1 source65abb2e, binding/invocationb88eb13 preserved.
First actual launcher CLOSED exit1, worker27784/create1791790993.8985312,
holder23536/create1791790993.7018092,4.719s. Failure before first full input hash
and model construction: `unexpected child`. GPU allocated/reserved0. No output
directory/model/controls/tasks/DEV/result. Original terminal,holder/log retained.
Its receipt did not preserve the child's identity; do not invent that identity.

Two actual independent no-model startup probes reproduce the launcher mechanism:
CREATE_NO_WINDOW0x08000000 creates a direct `conhost.exe 0x4` descendant before
imports, through CUDA startup and all20 polls. DETACHED_PROCESS8 has NO descendants
at all corresponding observations. Exact PID/create/commands in probe JSONs.
This supports a launcher fault correction, not retroactive identification of the
first lost child and not a neural/capacity result. No foreign process terminated.

Separate v2 source uses DETACHED_PROCESS8; retains the strict zero-worker-child
guard, all original numerical/quality/data/custody/resource criteria and1200s
budget. No general process whitelist added. New binding/output namespace and
source freeze; never mutate/restart the failed v1 artifacts.

V2 also saves seven RAW residual-coordinate trajectories for every DEV history:
before layer0 and after4/8/12/16/20/24, BEFORE final norm. Observation hooks clone
the existing tensors; no changed model mathematics, coefficients or decisions.
This adds~81.7MB within unchanged3GiB cap and avoids another future GPU replay
solely to locate composed projection drift. Model/controls/metric function ASTs
are exactly equal to v1. Extra source data are not read by the predictor.
All prior shadow protocol requirements remain, including completing scientific
failures and later full independent parameter/behavior/data audit.
