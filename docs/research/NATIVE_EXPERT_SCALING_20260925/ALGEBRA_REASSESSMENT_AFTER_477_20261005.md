# Dopo477: capacità, geometria attiva e informazione del routing

5 ottobre2026. Goal ACTIVE/INCOMPLETE. Riesame e proposta del prossimo lavoro;
nessun nuovo478 è congelato o eseguito. I risultati citati sono già completati.

## Vista complessiva e quattro problemi distinti

| Problema | Evidenza disponibile | Incognita che può bloccare il goal |
| --- | --- | --- |
| Capacità utile distinta |369: corrispondenza funzioni/router causalmente utile;370/371: reali64/128 da256;372:64peggiora,128esiti misti |Incrementi realmente utili oltre256; cardinalità e RAM non bastano |
| Rappresentazione e costo attivo |453: locale full-widthWI-I4/WO-I8 informazione conservata;454/456: costo delle ricette fallisce;472..475: altre geometrie respinte |Una trasformazione conveniente su tutti i bank/core, con costo completo |
| Informazione predittiva |476/477-R1: contesti esatti e impatto finito verificato;2.83%RMS naturale con190/3065argmax cambiati |Nuova ricetta ottimizzata sul contributo osservabile; composizione e nuovi stati |
| Ricerca e massa con n crescente |373:64->256 costoFULL+2.91%,router~3.83x;393:tail ampia;394/395ricette respinte |Gerarchia/LUT CPU che preservi vincitore E ampiezza con costo reale contenuto |

Le due conversioni originali Switch qualificate a7.415B/14.664B sono la base
reale per indagini sul transfer. La loro qualità e i rate warm/S29 specifici
non passano a un artefatto modificato. Altre famiglie/scala100B restano aperte;
397QwenNext ha qualificato header, non valori/predizioni. Le LUT sintetiche e
i formati Giga/Ling/Granite respinti restano chiusi nel proprio perimetro.
Nessuno dei quattro problemi si risolve contando nuovi record sperimentali.

## 1. Che cosa significa separare capacità e lavoro attivo

Per un bank omogeneo, in un modello idealizzato esplicito:

    P_tot = P_core + n P_expert,
    B_tot = B_core + n B_expert + B_router(n) + B_index(n),
    T_token = T_core + T_search(n,x) + T_tables(x)
              + sum_(e consultati) T_e(x) + T_dispatch + T_other.

RAM pone un limite a B_tot; numero di esperti consultati, loro costo e ricerca
ponono altri limiti. A parità di tutte le altre dimensioni, moltiplicare n per10
moltiplica la sola quota degli esperti per10. Una sorgente100B non implica
automaticamente10n rispetto10B: core, larghezze, topk e formati possono cambiare.
L'obiettivo è costruire una geometria che renda quel controllo praticabile,
poi dimostrarne utilità, non presumere il risultato dalla conta dei parametri.

Sul runtime373 il router piatto legge logicamente18432*n B/decoder posizione;
è4*768*n su sei bank. Sono4.718592MB a256 e47.185920MB a2560 a struttura
invariata. Questa è contabilità di indirizzi, non banda DRAM misurata o una
previsione dei tempi. A50token/s il budget medio è20ms: ogni stadio va pagato.
Le misure458 sul costo completo mostrano che la sola eliminazione ipotetica
degli esperti lascia comunque core/dense/head/router; il loro contributo non
può essere nascosto dalla compressione del bank.

## 2. Informazione della funzione: il quoziente dipende dal contesto

Su prefisso originale fissato, p_route=a:

    r=pre+a f, r'=pre+a f',
    l=Head_A16(FinalNorm_F32(r)/sqrt(D)), delta=l'-l,
    KL=log(sum_i p_i exp(delta_i))-sum_i p_i delta_i,
    deltaNLL_y=LSE(l')-LSE(l)-delta_y.

Le formule KL/NLL sono identità in aritmetica reale per i logit finiti misurati;
477-R1 ha verificato il vero operatore quantizzato, non solo il Jacobiano.
Softmax elimina delta+c1. La norma liscia elimina la direzione radiale solo
conepsilon0, e la direzione radiale dipende da r. In generale non esiste un
unico sottospazio globale dell'errore della funzione che sia invisibile per
tutti i pre/a/r. Il readout e le distribuzioni cambiano con input/prefisso.

La metrica locale infinitesima è il pullback

    G_(x,pre)=a^2 J_norm^T H^T [diag(p)-p p^T] H J_norm,

con H comprendente il fattore1/sqrt(D) del readout liscio. Il quantizzatore A16
ha discontinuità e non viene descritto dal Jacobiano. G è una motivazione per
un futuro obiettivo informativo, non un certificato o una nuova ricetta testata.
Un fit futuro deve usare contesti di sviluppo, pesi per libro/query espliciti,
regolarizzazione e budget della rappresentazione; validare poi il vero head e
l'intero modello su dati nuovi. Nessun fit sui logit validation477 e nessuna
eccezione per gli ID peggiori osservati ora.

Il risultato477-R1 orienta questa scelta: RMS energetico naturale2.83%, ma
KL media.03947 e6.20%argmax cambiati. Gli altri ID fuori25/37 hanno86.64%RMS,
KL.04318 e tutti190cambi naturali. Energia e informazione sono quantità diverse.
Il Fisher quadratico non sostituisce le code finite. La vecchia472 rimane FAIL.

## 3. Il routing deve conservare la somma, oltre al massimo

Score s_e=w_e^T x, Z=sum_e exp(s_e), vincitore e*, ampiezza a=exp(s_e*)/Z.
Se il vincitore è preservato e il nuovo denominatore è Zhat:

    a_hat/a = Z/Zhat.

Omettere massa tail Z_tail inflaziona l'ampiezza.393 lo misura già: a256,
shortlist oracle minimo m mediana140/p95 191 per errore<=1%. Non rifare
top64-only o le griglie low-rank/I8-refinement394/395 con un'altra soglia.
Nuova variabile proposta: sommare GRUPPI certificati senza leggere ogni score,
confrontando costo di centroidi/limiti/refinement con la scansione piatta.

### Bound di gruppo: deduzione reale, ancora da qualificare in F32

Per un gruppo g di m righe, centro medio c e residui v_e=w_e-c con somma0:

    u_e=v_e^T x, rho=R_g ||x||_2, R_g=max_e ||v_e||_2,
    c^T x-rho <= s_e <= c^T x+rho.

Jensen fornisce sum exp(u_e)>=m. La corda della funzione convessa exp su
[-rho,rho] fornisce

    exp(u) <= cosh(rho)+(u/rho)sinh(rho),
    m exp(c^T x) <= Z_g <= m exp(c^T x) cosh(rho).

Per rho0 la formula si estende per continuità con cosh0=1. Il bound sfrutta
somma residui0 ed è più stretto del solo m exp(c^T x+rho); per rho piccolo,
logcosh(rho)=rho^2/2+O(rho^4). Non richiede sparsità della massa tail.
Se rho grande, logcosh(rho)~rho-log2 resta largo: geometria delle righe e
dimensione effettiva possono rendere inutile la gerarchia. Questo è il rischio
da misurare, non una promessa che la massa si possa consultare in O(logn).

Per centroidi arrotondati la media dei residui non è necessariamente0. Occorre
il termine medio residuo, oppure intervalli conservativi per la sua deviazione;
applicare direttamente cosh con somma0 sarebbe scorretto. Prima di qualunque
scouting numerico occorre anche qualificare l'errore fra w^T x reale e il
preciso dot/router F32 sorgente e l'aritmetica degli intervalli. Nessun claim
di certificazione floating-point segue dalla sola deduzione qui.

Con score del candidato vincitore calcolato esattamente, un frontier di gruppi
non espansi dà intervalli L<=Z<=U. Si conserva il vincitore se il suo score
supera tutti gli upper score non visitati, con tie-break originale esplicito.
La sua probabilità sta in[exp(s*)/U,exp(s*)/L]. Il rapporto U/L controlla
la larghezza relativa; usare U come denominatore dà errore relativo al massimo
1-L/U<=epsilon se U/L<=1/(1-epsilon). Un gate più stretto U/L<=1.01 limita
l'errore a meno dell'1%, prima dell'inviluppo F32. Gli exp si calcolano con
shift comune; i bound si sommano mantenendo la stessa scala.

## 4. Prossimo passo scelto e criterio di arresto

Proposta NEW478: fattibilità di UNA gerarchia sui router reali128/256, mantenendo
visibile che sono checkpoint indipendenti, non una curva causale n. Riutilizzare
ALL full-score/input393 teacher/natural già qualificati e le righe originali;
nessun nuovo capture/modello/rifit delle funzioni o benchmark delle vecchie LUT.

Prima del freeze definitivo:

Ispezione statica successiva: il router393 usa prodotti/riduzione F64 di input
e pesi F32, poi cast F32. La softmax include sottrazione F32, exp libreria,
cast F32 e somma F64 sequenziale. [Nota di contratto nativo](METH_478_STATIC_ROUTER_CONTRACT_NOTE_20261005.md)
separa i bound deducibili dalle ipotesi/runtime ancora da ammettere. Non
chiamare certificato floating-point il solo bound geometrico in aritmetica reale.

1. Ammettere file/schema e rivedere il dot F32 sorgente; completare i bound per
   centro arrotondato, score/exp/somme e tie-break. Nessun risultato geometrico
   prima del congelamento di sorgente/protocollo/input/runtime.
2. Congelare UNA costruzione binary balanced, direzione principale dei soli
   pesi router e mediana con tie-breakID, leaf singleton. Nessuna query/label
   usata per scegliere partizione, rango, precisione o distanza.
3. Congelare UNA traversata: espandere per upper score finché il vincitore è
   certificato; poi espandere il gruppo con maggiore gap di massa finché
   U/L<=1.01 oppure fallback piatto. Pagare ogni centro/dot originale/norma/
   indice/exp/limite e fallback. Centri precomputati per tutti i nodi non
   diventano costo gratuito. Conservare le query anche quando il bound fallisce.
4. Qualificare controlli piccoli, tutti i bank/mode e tutti gli ID/vincitori/
   probability intervals contro393. Gate di lavoro PROPOSTO: media coefficienti
   letti<=60%flat, ogni bank/mode<=80%, p95<=100%, dopo tutti gli overhead
   vettoriali. È uno screening logico, non un test di velocità o DRAM.
5. Budget da concretizzare staticamente: CPU0/BLAS1, <=180s/2GiB/new96MiB;
   include byte admission completa. Se non praticabile, rivedere il piano
   PRIMA di osservare, senza campionare bank/query per farlo passare.

Bound/primal qualificati ma lavoroFAIL: chiudere questa gerarchia/norm-ball,
nessuno sweep di cluster/beam/soglie. Passaggio successivo motivato è una
struttura appresa del gating e della massa, con diverso modello/transfer e
proprio costo di training. LavoroPASS: ONE C cost inquiry con costruzione tabelle,
dispatch e fallback, poi nuovo artefatto/fresh quality/SAMErate. Neanche un PASS
a256 dimostra10x o100B; solo abilita la successiva scala reale.

In parallelo logico rimane il problema della rappresentazione informativa delle
funzioni. Il suo prossimo fit richiede una geometria conveniente già ammessa,
contesti di sviluppo completi e budget, non altra PCA euclidea sugli stessi
criteri falliti. Non avviare training costoso o importare automaticamente il
vecchio next del branch donor-adaptation. La scelta qui ritorna al vincolo n
e sfrutta le osservazioni esistenti prima di spendere su un nuovo artefatto.

## Invariante e checkpoint di rivalutazione

Rivalutare dopo ciascuna evidenza decisiva: quale requisito congiunto è stato
reso reale? quali costi sono solo contati? quale variabile nuova giustifica
il prossimo test? Fermare una ricetta fallita senza confutare l'intera famiglia
matematica. Goal resta capacità pretrained trasferita, qualità e>=50 sullo
stesso artefatto, routing/massa/LUT/DRAM, utile n crescente e più famiglie/scale.
Nessun esito di questo documento completa quel goal.
