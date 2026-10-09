# Broader-data repair1: preserve full dialogue identity and reuse pinned files

9 October2026. Preregistered before repair execution. Original attempt remains
FAILED:[first faults](CHATBOT_BROAD_DATA_FIRST_FAILURE_20261009.md).
No model/source/training observations occurred; both full downloads survive.

## Changed variable and unchanged destination

First-user-only identity merged histories sharing a generic greeting; the
everyday test component had only5 such groups. New group identity is SHA256 of
the canonical JSON list of ALL ordered user-turn contents,each normalized by
NFKC/casefold/collapsed whitespace. This preserves information in the subsequent
questions. It does not assert semantic independence,donor-pretraining absence,
or different first greetings across splits. Report first-user AND full-sequence
distinct group counts by source/split so the change is visible.

Read the COMPLETE existing test shard first and exclude all its normalized
user-sequence groups from selected train. All selected groups and canonical
last-user prefixes remain globally unique. Preserve old279 individual user-query
keys,13-source quotas32 FIT/8 DEV/8 RESERVED=624,deterministic rank selection,
no model/answer/length filtering or truncation. Reserve104 cases unqueried.
Prior assistant history/external final replies keep their public-data provenance,
not donor label status. Canonical source tokenizer/template adoption remains.

The new grouping is a different data variable and requires reading retained
rows under that identity. Do not rerun the old first-user selection or repeat
downloads. Original source API/card/LFS identity and two actual download extents
are inputs. Also bind original failed launcher/reader/progress/byte code archives.
Original missing reader PID/memory/exit receipt remains missing permanently.

## Apparatus and budget

Allow Python and ONLY the pinned System32 conhost executable as worker children.
The launcher tracks observed own descendant PID/create-time identities and stops
them on failure,preventing an orphan reader from continuing after supervision
ends. This changes error cleanup; no unrelated process may be terminated.

ONE new family<=600s/6GiB OS/768MiB NEW output,worker30s reserve/reader240s4GiB.
Actual network bytes0. Existing328,261,485B are not copied into the new namespace.
Hold actual new reader/worker handles through exit,record memory/commands/cost,
bind/check all inputs before/after. Arrow-only child/no Torch/Transformers/Rust;
parent Rust/template adoption/no Arrow. Same selection/transport gates under
the new declared identity;full goal remains ACTIVE/INCOMPLETE.

Do not mark previous failed gates true. BROAD_DATA_ADOPTION_PASS applies only
to this repair's own complete receipts and the new corpus. Data coverage ready
is not preserved chatbot quality or accepted50. The next step is a costed
donor-label/whole recovery pilot respecting all actual context-length tiers.
