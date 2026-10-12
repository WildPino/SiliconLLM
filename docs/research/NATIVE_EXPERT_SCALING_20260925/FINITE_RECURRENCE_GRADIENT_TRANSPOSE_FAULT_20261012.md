# First tangent-projection implementation fault retained

V1 capture21582 closed exit1 after all24 FIT backward records were saved,32.070s.
It stopped before direction/FD/candidate selection or any accepted update.
Original normal projection implemented GprojW=G_W-V Lambda rather than the
protocol's G_W-V Lambda^T. Every head tangent residual fails (range~.0153-.5885);
slope identity relative.118052. This is a real algebraic implementation fault,
not rounding or a bad residual denominator. Initial gauge/all output/GY/
per-case V/W gradients and first-fault/terminal remain immutable on E/in repo.

V2 corrects ONLY that transpose and adds first-fault custody inputs/new storage
namespace. Same source operands/protocol/gradient/ranks/selection/resource
bounds/tolerances. No threshold loosened. Full independent audit must qualify
every saved V1 backward as well as V2 complete backward/direction/candidates/
new actual scans. Reproducing the wrong projection cannot substitute for an
independent Sylvester solution. V1 has no nonexistent candidate result claim.
