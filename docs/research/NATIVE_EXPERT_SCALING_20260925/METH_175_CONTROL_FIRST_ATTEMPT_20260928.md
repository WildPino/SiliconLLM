# METH-175 control: first attempt and runtime-guard correction

The first full E1,280 control process started on 2026-09-28 at 21:21:15 UTC and stopped after update 16, at 52.469 seconds elapsed. Exact eight-prompt initial BF16 parity passed and update 16 was finite. No completed bank or result was written. The preserved failure and progress files are local evidence from this attempt.

The stop was caused by an inherited METH-136 projected-time guard. When called for `control`, that guard added the estimated time for a *second* arm in the same process. METH-175 instead runs each arm in its own process, under its own frozen cap. The guard therefore projected 18,905.909 seconds against the control's 16,200-second cap despite the single-arm projection being roughly half that value. The observed 16 updates do not establish that the entire arm will fit the cap.

The correction adds `project_following_arm=True` as the METH-136 default, preserving earlier combined-process behavior. METH-175 passes `False` for its separate-process arm. Its elapsed-time, GPU and RSS budget checks remain active on every update; the control cap remains 4.5 hours. Restart the control from update 1, keeping the failed attempt's logs separate. This correction does not change the frozen draws, objective, routing or quality gates.
