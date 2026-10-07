# Original-source startup repair2: identify/prevent unexpected subprocess

Original source-capture freezeabe08c5, executora7fbfa/session33215, final239877
exit1. [Launcher first fault](chatbot_original_capture_20261007.launcher_failure.json)
SHA3cadb66d194edb4a2862ce30251b7996b46c6f54e183c419c85a090e374899ad;
empty [worker log](chatbot_original_capture_20261007.worker.log) SHAe3b0c442...
Launcher11744/create1791382270.5884209,worker25916/create1791382288.0190718,
worker actual forced exit1/through-exit OS peak597168128B; family elapsed33.031s,
parent last snapshot41377792B. Unexpected descendant identity/resource lost;
complete family resource gate UNKNOWN/FAILED, not promoted from worker peak.

Runtime/input/view pre-seal passes. Capture directory exists and is EMPTY.
By code order each binary opens BEFORE its first source forward, therefore
no source forward or response/captured operand was entered/completed. Source
weight decoding/load stage before forced kill remains UNKNOWN, not asserted0.
Worker did not produce a failure receipt after parent kill; only parent trace
exists. Original no-descendant requirement remains unchanged. Current owned
source Python instances no longer live; don't kill unrelated console hosts.

Repair ONLY startup observability/containment: print flushed import/load/forward
phase markers, register Python audit hook rejecting subprocess.Popen BEFORE
creation while preserving argv/executable/short stack (never environment).
Parent preserves PID+creation/name/exe/argv for any remaining native-started
descendants before stopping. No blanket allowlist/ignored children and no
source math/dtype/cache/serialization/case/partition/threshold/budget change.
If a subprocess attempt is caught by an optional-library probe, retain it and
continue only if the original pure-eager source contract/no-child gates still
hold. A fatal probe/unknown child still stops; identify its actual source call
before a further numbered startup repair. No cause is inferred from empty log.

Numbered metadata binding reuses whole runtime/source archive SHA unchanged,
updates only worker/launcher code and appends prior binding/fault/this protocol.
Use a NEW capture/output namespace. Failed startup re-executes imports and
possibly source loading; no completed source forward/response/control replay.
Freeze before first attempt to acquire original48 conversations. All fixed
ALL48/32-fit/16-development and worker300s/family1200s/OS/GPU/size gates stay.
