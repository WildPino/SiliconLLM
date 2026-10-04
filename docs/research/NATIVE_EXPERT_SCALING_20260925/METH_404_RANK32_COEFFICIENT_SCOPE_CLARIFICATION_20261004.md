# M404 protocol arithmetic clarification

The frozen protocol states1,179,648 active coefficients for all12 banks.
Correct two-factor matrix count is12*(768*32 +32*768)=589,824 coefficients.
The controller/raw resource ledger already computes589,824 correctly.
F32 factors plus two means occupy12*4*(768*32+32*768+2*768)=2,433,024B,
as correctly stated in protocol/raw. NPZ archive bytes include format overhead.

No scientific outcome/controller/protocol/raw is rewritten. This correction
reduces a prospective arithmetic cost, not a measured engine cost. All12 local
input-eligibility gates still FAIL; no model/accepted-rate claim.
