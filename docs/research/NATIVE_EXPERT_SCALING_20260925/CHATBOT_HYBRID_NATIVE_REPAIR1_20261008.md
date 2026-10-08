# Native apparatus repair1, before first C model inference

Original freeze dfb4c0e3d3a59f497905ee00f2f7284b7f35f373 and native binding
4d063eec1833b1044cd8c37fcffbf7ca3e71ed14e5ecf378ab7f59ec0093492b.
Kernel header/body receipt written, then process monitor rejected `clang-21.exe`:
installed MinGW clang.exe is a16KB target wrapper invoking the98KB versioned
driver. Original worker/launcher failure/empty compile log/header retained.
No executable/native queries/logits/trace were produced. Authoritative process
inventory confirms no compiler/worker descendants remain live.

Add exact observed clang-21.exe name to whitelist and hash existing selected
versioned driver/linker/libclang/LLVM DLL files alongside wrapper. Preserve
first fault externally because launcher interruption could stop worker exception
serialization. No C source, format, weights, numerical gate, data or budget change.
Reuse completed425,210,736B packed model SHA191d1d20058946702330038336553cd6588efbcebdbac85f70410a03126c3571;
no export/quantization or model observation repeated. First C compile/inference
still pending. Complete current full goal remains unproved.
