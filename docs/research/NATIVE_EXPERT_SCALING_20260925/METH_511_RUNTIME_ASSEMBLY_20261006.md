# METH511 local isolated runtime assembly — metadata before scientific freeze

Purpose: preserve qualified Torch2.6.0+cu124/Transformers4.57.6 actual bytes,
while excluding unused optional sklearn/pandas/PyArrow integrations from
native/donor phases. The earlier Arrow import fault was real and its cause
is unidentified; excluding an unused import does not diagnose that cause.

ONE new venv without pip; explicit junctions to30 already-installed required
distributions and their metadata/native library directories. Python3.12.10,
`-I` isolated invocation; no root .venv .pth/site injections or user packages.
No pretrained/model/corpus/numerical/native/compiler queries in setup.
All installed original environments/files remain untouched.

ONE public wheel download DuckDB1.4.4, CPython3.12/WindowsAMD64, <=16MiB,
SHA d525de5f282b03aa8be6db86b1abffdceae5f1055113a03d5b50cd2fb8cf2ef8.
The [maintainer's PyPI release](https://pypi.org/project/duckdb/1.4.4/)
publishes that wheel/checksum. Extract locally after byte verification; no
dependency install, auto extension download, package upgrade or Torch download.
[Parquet docs](https://duckdb.org/docs/current/data/parquet/overview.html)
describe read_parquet and physical file_row_number; cohort uses exact row IDs,
not an unordered result position. Decoder/code/metadata/reader controls and all
scientific phases will be frozen before importing/querying this reader.

Setup bound240s, downloader120s, junction helper120s, local own new bytes<=80MiB
excluding read-only junction targets already charged as existing runtime assets.
Stop/preserve first fault; never overwrite terminal setup output or repeat a
completed numerical/model/native call. Source/setup receipts and actual minimal
runtime admission are required before ANY511 scientific cohort or model work.

This is apparatus preparation, not capacity transfer or goal completion.
