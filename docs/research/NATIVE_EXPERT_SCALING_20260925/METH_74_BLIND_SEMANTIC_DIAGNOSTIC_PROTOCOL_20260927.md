# METH-74: blinded semantic diagnosis of the METH-72 responses

METH-72's automatic decision remains `stop_instruct_parent_promotion`
because its pair-utility gate failed. Its 24 donor/student response pairs
were generated on frozen, source-disjoint excerpts before that decision.
Review those existing pairs to learn whether the E1280 student's answers
regress semantically; this diagnostic cannot override METH-72.

Build an A/B file with only the 384-character excerpt and two 128-token
continuations for each source. Assign donor to A when the first 64-bit
prefix of SHA-256(`meth74-blind-74074|source_id`) is even; otherwise
assign it to B. Keep the mapping in the script and omit it from the
review file. Do not inspect donor/student-labeled continuation rows
before freezing the verdict.

For each side, count distinct unsupported factual claims about the
excerpt; flag severe unsupported named-entity, numerical, comparative
or causal claims among them. Record whether the response omits a
requested specific detail. Cite the shortest excerpt span that
contradicts or limits each counted claim, and keep ambiguous readings
separate. Commit the A/B verdict before unblinding and calculate
donor/student totals afterward. Report raw totals and whether E1280
is no worse than donor in all three finding categories. A single
reviewer on 24 excerpts is a limited diagnostic, not a population
estimate or a promotion gate.
