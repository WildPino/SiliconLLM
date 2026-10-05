# METHOD historical UTF8 character repair

5 October2026. During480documentation closeout, five historical METHOD lines
contained recursively Windows1252-decoded UTF8 characters. They were already
present at80e6588; an implicit-code-page editor pass further amplified them
before being reverted outside intended480updates. Final repair uses explicit
UTF8 and strictly reversible inverse-encoding steps on ONLY those five lines.
Each complete ASCII subsequence, including every numeric value and word, is
unchanged; all other lines unchanged. Complete old text remains in Git.

METHOD reduced from13709844B to114383B; numerical science/native/source/
RAW/RET/old decisions are unchanged. [Full repair record](meth480_method_utf8_repair.json).
Future text edits must explicitly read/write UTF8; no default Windows codepage.
