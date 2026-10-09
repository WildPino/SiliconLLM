# Audited interrupted checkpoint: complete fixed-state prefix evaluation

9 October2026. Preregistered before any new checkpoint286 forward.
Question:does actual learning recover the balanced source-prefix distribution,
and does held-out recovery justify continuation of this conversion recipe?
Decision:continue from actual state only if recovery supports it; otherwise
prioritize decomposition/representation initialization under the engine target.

## Reused and new observations

Reuse exact audited checkpoint boundary286/Adamstep294,
SHA7b95e699a57c9bc0ed320bef016d95ba78ac1df2e79b9ea98a0c12a1ba78eec5,
254,932,736 F32 parameters/211 tensors. Audited complete initial160 values
refer to the prior step8 model. Evaluate every original160 FIT/DEV case once at
NEW fixed boundary286. Checkpoint was selected by the original time reserve,
not DEV. No completed update/initial observation/source generation/native run
repeated; no missing historical routing counts reconstructed.

No donor execution or optimizer restoration/update. Load the saved checkpoint
on CPU, construct the original target and strictly load all model fields to GPU,
drop checkpoint/model/moment CPU copies, eval/no-grad/all parameters frozen.
Original AQ63 and ternary no-grad formula; no dither, fine-tuning or margin loss.
TF32 off/F32, original naive SSD/no optional kernels,case-zero recurrent/SWA
state and continuous sequence within each case. Prefix input/positions must match
actual corpus IDs. V65537; all5746 full-vocabulary rows retained durably.

Every finite saved F32 result receives stable NumPy F64 logsumexp and exact BF16
teacher bit lift; F64 forward KL>=-1e-10; per-label GPU F32 KL agrees within
absolute1e-4. All greedy IDs checked across CPU/GPU. Aggregate equal-case and
label-weighted metrics per split/domain. This uses different arithmetic in the
same worker, not an independent process/library/CUDA determinism certificate.

Collect final inference routing support for all12 sites/72banks across domains.
Counts must equal actual input IDs*8 per site/domain. This new final support
does not fill the missing historical before/training support of the stopped run.

## Fixed diagnostic gates and scope

Inherit original quality thresholds without changing original512 completion:
all160/5746 final rows; FIT caseKL<=.50 audited initial;DEV<=.75 initial and
<=1.0 with label disagreement<=20%;everyDEV domain<=2.0/<=35%;atleast8 visited
banks/site over all final cases. All gates true -> INTERRUPTED_STATE_PREFIX_RECOVERY_CLOSE,
otherwise INTERRUPTED_STATE_PREFIX_RECOVERY_FAIL. Both are diagnostic states,
not BALANCED_RECOVERY_ELIGIBLE or chatbot/native admission. Original incomplete
512 run remains incomplete. No DEV-best checkpoint search.

128 FIT/32 DEV,eight related authored template domains,23 partial source replies.
No new/fresh/RESERVED source query or own-history generation. Prefix recovery
cannot establish broad chatbot quality,>=50 accepted batch1 rate,useful large n,
structured CPU routing mass/DRAM or family/scale transfer.

## Cost, identity and stops

ONE local GPU family<=900s/10GiB OS/6GiB GPU allocated/7GiB reserved/2GiB output;
worker guard reserves60s. Actual previous whole160 initial evaluation~433s plus
load/launcher/hash/packet reductions. New model shape unchanged; cap is a
feasibility test, not a claimed measured prediction. Expected raw rows exactly
5746*65537*4=1,506,302,408B plus JSON/support/log. Retain partial first fault,
all completed durable rows and actual terminal; never restart merely for expired
observation. No other timing/model job or T4. Through-exit OS/actual GPU peaks,
worker/family time,source/student/optimizer counts reported.

Bind actual successful state audit/result SHA+receipt,checkpoint/corpus/all160
source packets,original criteria,target/worker/launcher/Python/protocol,selected
runtime files and preserved foreign tracked bytes. Isolated Python3.12.10,
Torch2.6.0+cu124,Transformers5.13.1,NumPy2.4.6,tokenizers0.22.2,psutil7.2.2.
Freeze before binding/run. This does not certify every native DLL/library file.

After closure adopt saved metrics/state identity before deciding a continuation
or controlled sum/mixture experiment. The actual engine-shaped converter,
fresh same-artifact quality+speed and larger useful capacity remain the goal.
