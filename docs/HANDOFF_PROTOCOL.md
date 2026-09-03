# Handoff protocol for spawned figures

**Why this exists.** On 2026-09-03 a session rate-limit killed two working agents mid-task. Everything
they held in context died with them; the S1 BPB probe they had launched died too, because it was a
child of the agent's shell. Restarting cost a full re-derivation of state from logs and JSON.

**What cannot be done.** There is no programmatic read of remaining session quota. Verified
2026-09-03: the `claude` CLI exposes no `usage`/`quota`/`limit` subcommand; `/usage` is client-side
only and is not expanded under `-p`; the `quotaLimits` payload (with `resetsAt` and `five_hour`)
appears **only** inside a 429 rejection, never on a successful response and never in the transcript on
disk. So a "write your handoff at 80% of quota" trigger is not buildable. Do not promise one.

**What replaces it.** A threshold you cannot read is a bad trigger anyway. Write the handoff
*continuously*, so a kill at any instant leaves a current one.

---

## The three rules

### 1. Long runs are detached from the session, never children of it

A run that outlives a single agent turn must survive the agent. Launch it so the harness cannot reap
it:

```powershell
$p = Start-Process -FilePath "python" -ArgumentList "<script>","<args>" `
     -WorkingDirectory $dir -RedirectStandardOutput "$dir\results\run.log" `
     -RedirectStandardError "$dir\results\run.err" -WindowStyle Hidden -PassThru
$p.Id | Out-File -Encoding utf8 "$dir\results\run.pid"
```

Not `nohup ... &` inside the Bash tool — that dies with the agent.

Any run longer than a few minutes must also **checkpoint and resume**. The S1 probe is the reference
implementation: it writes its JSON after every completed `p`, and on restart reloads it, logs
`RESUME: n p-values already on disk`, and recomputes only what is missing. It also recomputes the
baseline every time and compares it to the stored one — an identical baseline across two separate
processes is a free determinism check on the whole forward path.

### 2. Every spawned figure writes its handoff as it goes

The spawn prompt must contain this instruction, verbatim:

> Write your handoff to `docs/handoffs/<ROLE>_<TASK>.md` **before your first substantive action**, and
> update it after every milestone — a completed measurement, a written section, a decision taken.
> Assume you will be killed without warning and that your successor will be spawned **fresh, with no
> conversation context**, and will read only this file. It must be sufficient on its own.

Not at a threshold. Not at the end. As it goes.

### 3. The successor is spawned fresh, and reads only the handoff

Never re-spawn a figure by replaying context at it. Spawn a clean agent whose entire briefing is:
its role, and the path to its handoff. This is the point of the protocol — a fresh agent with a good
handoff outperforms a resumed one carrying a bloated context, and costs a fraction of the tokens.

---

## The handoff template

```markdown
# HANDOFF — <ROLE> — <TASK>
updated: <ISO timestamp>   status: IN PROGRESS | BLOCKED | DONE

## 1. The task, in one paragraph
What I was asked to do and what "done" means. Written so someone who has read nothing else
understands the goal.

## 2. Pre-registration / constraints I am bound by
Brief path + commit. Anything pre-registered that I may not change. Decisions already taken
above me that I must not relitigate.

## 3. What is DONE — with the artefact that proves it
| # | what | artefact on disk | key number |
Every claim names the file it comes from. No claim without a path.

## 4. What is RUNNING right now
| what | PID | log path | resumable? | expected finish |
If nothing is running, say "nothing running".

## 5. What is NEXT, in order
1. ... (the exact next command, copy-pasteable)
2. ...

## 6. Traps I have already hit
The failures a fresh agent would otherwise repeat. This section is the whole reason the file
is worth more than the transcript.

## 7. Everything needed to restart cold
Paths, env vars, thread counts, seeds, the exact command lines. Assume no memory of anything.
```

---

## Roles

| figure | does | must never |
|---|---|---|
| **Costruttore** | builds the instrument, runs it, reports what it measured | know the expected conclusion — see the no-anchoring rule |
| **Controllore** | audits the Costruttore's output against its artefacts, issues BLOCK/FLAG/PASS | write the thing it audits |
| **Media Manager** | lays out and pushes | decide content; it publishes what it is given |

The no-anchoring rule stands and is not weakened by this protocol: **a handoff written for the
Costruttore must not contain the expected result.** It carries the task, the constraints, the state
and the traps — not the answer.
