# Audit FD flag serialization fault retained

Partial auditV2/session53010 closed1 after all24 full autograd/bit checks and
independent tangent qualification,while emitting the first FD control. NumPy
comparison produced numpy.bool_,which standard JSON cannot encode. That run has
no complete adjudication;unwritten FD values are not invented.

AuditV3 converts ONLY comparison flags to Python bool for JSON. Same math,
control scales/tolerances/caps/data;new source/result namespace. All earlier
faults/terminals remain immutable. Full audit still required before a protocol
scale change or parameter update.
