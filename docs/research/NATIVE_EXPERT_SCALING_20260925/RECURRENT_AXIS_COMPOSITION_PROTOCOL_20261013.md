# Fixed channel/state composition, site23

13 October2026. Freeze before new observation. Reuse independently qualified
FIT adaptive384 recurrent-output maps, site23 (both local norm-mode gates PASS),
and fixed FIT dual96 state maps. No new fit/rank/coordinate/DEV choice.

Source state S_h is64x256. Channel maps Vc,Wc are64xr_h, state maps Vs,Ws
are256x96; Wc^T Vc=I, Ws^T Vs=I. Scalar head decay commutes with these
time-independent maps: T_h=Wc^T S_h Ws, T'=a_t T+delta*x'_t*B'_t^T,
x'=Wc^T x, B'=Ws^T B, C'=Vs^T C. Approximate recurrent readout is
Vc T C', with FULL D_h*x_h added afterwards. This is not equality to the
original readout: omitted channel/state subspaces and cross interaction remain.
Nonuniform packed widths total384;all48 heads retained. Source B/C/gate/postconv
x/delta supplied;no closure from compact causal hidden state demonstrated.

All48 exact histories,24 FIT/24 DEV. Actual F32 source SSD chunk128 kernels:
3072x96 state-only and384x96 joint,96 new scan/history records. Reuse actual
384x256 channel baseline outputs/latents. Fresh joint input control versus
baseline<=1e-6. Full joint scan versus state-only output projected with same
Wc<=1e-4, not comparison against the uncompressed source (which has real error).
Both output modes: fixed source denominator oracle and reconstructed combined
postgate RMS. Controlled F64 decoding/out contraction with BF16 normalized
input/output;old full F32 precision bridge not silently admitted.

Store projected B/C96,full state-only scan,joint scan/fresh projected inputs,
all four new BF16 outputs per case. Recompute all baseline outputs bit exact.
Retain raw recurrence-error component/cross/interaction energies:
e_joint=e_channel+e_state+e_interaction;identity relative<=1e-10. This definition
does not establish additivity of errors after gate/RMS/out projection.

Original local DEV budgets:equal-case mean<=.1,worstcase<=.2,
worstdomainmean<=.2,centered mean<=.1,both modes independently reported.
No selection among arms using DEV. If joint qualifies, native state/kernel cost
pilot becomes justified; otherwise isolate interaction/numerics before refit.
No site0/12/global/chatbot admission from a site23 pass. Full source generation,
convolution/gate/skip/norm/out costs still retained;21.33x recurrent state
coordinate reduction is algebraic,not measured whole-model speedup.

Budget producer900s/audit1200s,OS6GiB,GPUallocated4GiB/reserved5GiB,
new output2GiB on E. SixCPU threads/offline donor/no source or native forwards,
no T4/RESERVED/history extension. Full audit even if local quality fails:
independent NumPy maps,EVERY3072x96 and384x96 CPU state/timestep,
full latent/commutation/decoded BF16 vectors,all metrics/domains/energies/gates,
input/result/output hashes before/after. No automatic subagent;hours-long jobs
follow user announce/launch/pause instruction. No clean timing admission.
