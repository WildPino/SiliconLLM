# R4.3: metadata-only UTC event-window correction

The retained R4.2 main Windows query declares17:28..17:49UTC but returned actual
Event1000 XML timestamps15:47/15:48UTC, PID16992/arrow.dll. Those events are outside
the scientific interval and do not belong to main PID31088. Thus the original
Get-WinEvent FilterHashtable UTC DateTime path does NOT establish the intended
time-filter contract on this host. Preserve that first query, including both
actual unrelated event records; do not silently replace it or claim its zero
matches prove the correct interval was searched. No diagnosis of those other
process failures is established here.

After the sole independent R4.2 audit finishes, freeze this new metadata helper,
then query Application Event1000 using explicit TimeCreated SystemTime UTC XPath.
Read times from actual XML, require every returned event lies in the exact
requested UTC interval, and use the two retained actual events as a positive
control of the query. Match actual process PID, own interval and event process
creation time when available (2ms tolerance for retained Unix float precision).

Requery actual successful R3 preparation, original main root/all578 original
command roots, R4.2 quality and R4.2 audit. Recovery's inherited old command list
must not become its actual process scope. Retain all events and unmatched records.
The original prepare/main/quality/audit queries remain immutable. New UTC receipt
and metadata-only admission supersede their zero-match time-window claims.

No C, model, donor capture, export, cohort, metric, bootstrap or independent
numerical audit replay. RAW quality14/15 and economics4/5 remain negative. Known
PyArrow import faulthandler diagnostics remain explicitly retained; this query
does not turn a first-chance diagnostic into a zero-exception claim. Stop and
record unavailable/invalid query or actual matching event before admission.
