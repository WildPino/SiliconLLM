# METH-213: fresh saved-core quality cohort frozen

After the METH-212 loader pass and the committed METH-213/214 protocol,
deterministic model-free selection excludes 3152 prior source IDs and
the prior audit/training/teacher fragment corpus. It selects eight code
files, eight PG19 shard-14 books and eight technical documents at the
pinned historical Git commit. All lexical, span, source-disjoint and
fragment-overlap gates pass. Historical repository bytes are used even
where the current working tree has unrelated edits.

[Manifest](meth213_stored_core_fresh_manifest.json) SHA256:
`bcd8295be1b2d62dceaf103f985182bad11ac50dd739a63cfb92d11400b5af10`.
[Answerability](meth213_answerability.json) SHA256:
`462eb44c18c83a2df3b27780b7c82e49ecdd2cce8d581a1d1843acef20010409`.
All 24 anchors are specific details present in the visible excerpts and
are frozen before model inference. The first anchor-binding invocation
rejected a four-character anchor under its >=6-character assertion;
the surrounding visible phrase was substituted before any inference.
No source, prompt, candidate or threshold changed.

Selection: 206.531 seconds CPU, 2.640 GB end RSS. This is provenance
and answerability evidence only. Next run the frozen three-arm
prediction gate, stopping generation on any failure. No fresh quality,
native rate or larger-n specialist claim follows from selection.
