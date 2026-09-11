## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Comunicazioni con l'utente

Quando mi serve un'azione dall'utente (macchina scarica per un timing, un run sul T4, una
credenziale, una decisione), la scrivo in `docs/COMMUNICATION.md` oltre che nella risposta:
l'utente controlla quel file ogni tanto e nella chat le indicazioni si perdono. Una voce per
richiesta, con stato APERTO/FATTO/INFO, data, il comando esatto se ce n'e uno, e cosa si sblocca
quando e fatta. Le voci chiuse si spostano nello storico in fondo, non si cancellano.
