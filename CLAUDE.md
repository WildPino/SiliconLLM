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
l'utente controlla quel file ogni tanto e nella chat le indicazioni si perdono.

**`COMMUNICATION.md` CONTIENE SOLO COSE DA FARE, ED E CORTO.** Regola data dall'utente il
2026-09-12, dopo che il file era arrivato a 1.407 righe e non si capiva piu cosa fosse gia
fatto:

- **In cima una tabella `DA FARE ORA`**, una riga per voce aperta: cosa deve fare, quanto costa,
  cosa sblocca. Si deve capire tutto da li, senza scorrere.
- **Una voce aperta sta in ~20-25 righe.** Il ragionamento lungo non va qui: va nella sonda,
  nell'`INDEX.md` o nel ledger.
- **NIENTE voci INFO lunghe** e **niente catena "Ultimo aggiornamento" annidata** (era diventata
  una riga sola da 6.002 caratteri: e il motivo per cui il file era illeggibile). Una data secca.
- **Le voci chiuse non si cancellano ma non restano per esteso**: una riga nello storico in fondo
  (data, voce, esito) e il testo completo in `docs/COMMUNICATION_ARCHIVE.md`, che l'utente non
  deve mai aprire. Anche l'archivio e gitignorato.
- Ogni voce aperta ha: stato, data, il comando esatto se ce n'e uno, e cosa si sblocca.
