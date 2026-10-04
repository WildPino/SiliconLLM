# M428 result: tiny finite-difference checker precision diagnosed

Frozen ec83636; first/only run exit0, ALL8 diagnostic gates PASS.
Raw b1e46a804d0b6c3b5beab03457545f2243e171353ddb32a08aac38aad7f344fa.
No fitting, real-source forward, C change or retrospective427 promotion.

SAME tiny real A F64 FD error reproduces427 EXACT0.0005427976577140341.
Its reference gradient norm is6.00393762723403e-10; B norm8.122673407575743e-10
also yields F64 FD failure0.000261401037137864. Decimal80 FD at SAME1e-4 step
gives A error8.954586425209421e-15/B1.1262355154491045e-14; independent fixed-mask
imaginary derivative gives A9.903212259981902e-15/B1.568233101901024e-14.
Tiny F64 finite arithmetic, not a detected native/local-gradient defect,
explains the failed checker at this same anchor. No step or bound changed.

Both arms ALL7 fields: imaginary/reference max2.463922e-14, Decimal80-FD/
reference max2.3190224e-9, native against either max3.335328e-7, all within
UNCHANGED1e-5/1e-3 bounds. Decimal80 FD/imaginary max2.319023e-9. No ReLU
crossings, offsets fixed, exact source prefix/native endpoints. Full F64 FD
failures retained alongside Decimal80/imaginary/native/reference arrays.

1.078s/max425181184B/645446B hashed/54232B output, CPU0/Torch1/BLAS1.
Two archives: real29066B SHAa209fd913afbbdf3df137074eace1dad71ab72e896b9cde24474baff1b03ade1;
adapter25166B SHAced28a6f7b9eccd3e091f221cdcacdfe273421ac556342cb9e0e5fcf570ffbed.
No GPU/T4/network/new corpus; no real model job. Exact known daemons preserved.

Next NEW429: use THIS immutable tiny Decimal80 FD and imaginary control in the
427 pilot prerequisites, keeping actual forward/backward, real directional FD,
ALL9 capacity gates and ALL learners/data/budgets/classifier fixed. Sole change
is precision of tiny independent checker. Freeze before numeric outcomes;
427 remains FAIL, and complete429 capacity outcome must decide the recipe.
