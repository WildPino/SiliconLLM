#!/usr/bin/env python3
"""Freeze all excerpt-only findings without consulting the saved arm mapping."""
import hashlib
import json
from pathlib import Path
import meth267_complete_core_blind_semantic as B

PANEL_SHA='11cdc89ecef16e11bf3408f952053cacf8b19036fedc9501c309a2df6222631c'
FINDINGS={}
AMBIGUOUS={}


def finding(i,s,claim,evidence,severe=False):
    FINDINGS.setdefault((i,s),[]).append({'claim':claim,'evidence':evidence,'severe':severe})


def ambiguous(i,s,claim,evidence):
    AMBIGUOUS.setdefault((i,s),[]).append({'claim':claim,'evidence':evidence})


for s in 'ABC':
    finding(0,s,'Performance is measured in rows per second (RPS).','The excerpt gives GB/s bandwidth rows and tok/s; it never gives rows per second.')
finding(0,'A','The precision is TERMINARY.','The visible precision is TERNARY.')
finding(0,'A','7.4x to50 is a specific token count.','7.4x is the remaining performance factor to50tok/s, not a token count.')
finding(0,'A','37/42GB/s is a token size.','GB/s denotes bandwidth; the cited ledger rows are withdrawn.')
finding(0,'B','The performance has the lowest RPS and TPS values.','No lowest-values comparison or RPS measurement is visible.')
finding(0,'B','Qwen2.5-1.5B is a file read from T2_TERNARIZATION_RULE.md.','Qwen2.5-1.5B names the donor in a line of that document, not a file being read.')
ambiguous(0,'C','The performance has37/42GB/s rows.','Those numbers are mentioned, but the response does not clarify their withdrawn status.')

for s in 'AC':
    finding(1,s,'The check tests whether a specific byte is present in g_b.','The branch tests the truth value of g_b["fires"], not byte presence.',True)
finding(1,'C','When the byte is present, the script writes OUT and exits with an error.','Writing OUT and SystemExit occur inside if not g_b["fires"].',True)
finding(2,'A','It calculates the percentage of tokens within tolerance.','near compares normalized scalar error; it neither counts tokens nor calculates a percentage of them.',True)
finding(2,'A','The tolerance is used to determine the E48 sweep6 value.','E48_B_MS_PER_POS is a fixed literal; no such determination appears.')
finding(2,'B','E48_B_MS_PER_POS measures tokens per second.','Its name/comment identify milliseconds per position/whole token, not tokens per second.',True)
finding(2,'B','E48 sweep6 is in the B.5 phase of E46.','The E46 comment belongs to an earlier line; the E48 constant has its own sweep6 comment.')
finding(2,'B','TARGET_TOK_S is a target token count.','The constant is a token-per-second rate.')
finding(2,'B','near compares the difference directly with the tolerance.','The function divides abs(got-want) by abs(want) before comparing.',True)
finding(2,'C','The E48/B.5 numbers are token-frequency and token-count metrics.','They are a milliseconds-per-position constant and a fixed relative tolerance.',True)
finding(2,'C','The target is a token count.','TARGET_TOK_S is a rate,50.0tokens/s.')
finding(2,'C','A function calculates token frequency.','near computes a relative comparison; the other visible function only begins a four-way-table docstring.',True)
finding(3,'A','818389... is the Source Run.','That clipped hash ends the preceding tuple; SOURCE_RUN is the separate path prefix.')
finding(3,'B','This is a JSON object.','The visible tuples, path division and SOURCE_RUN identifier are Python expressions, not JSON.')
finding(3,'C','Source Run is SOURCE_RUN/pinned_reference/prefill8/q-0.full.f32le.','That complete path is the ref_q file value, not the SOURCE_RUN prefix itself.')

for s in 'AB':
    finding(4,s,'One JSON object includes the configuration and expected/actual-or-wrong lists.','The excerpt shows a file-record dictionary, a loaded JSON result, and separately built expected/wrong Python objects.')
finding(4,'A','Model parameters are stored in h1_result.','Only a JSON result is loaded; model parameter storage is not established.')
finding(4,'B','The model parameters are expected to be hard-coded into its configuration.','expected contains audit expectations, not a statement about model parameter storage.')
finding(4,'C','The record path is a model file.','The visible generic path/size/digest record does not identify a model file.')
finding(4,'C','A sample JSON object supplies expected and wrong values.','expected/wrong are separate Python variables/comprehension, not a sample JSON object.')
ambiguous(4,'C','A SHA256 hash and a digest are listed.','sha256 has the value digest; the wording may imply two objects but may also name one hash twice.')
finding(5,'A','A function is imported from a library named fc.','run_strat01_layer1_ffn_output_component_cross_input is imported as the module alias fc.')
finding(5,'B','The imported components come from engines named by the run modules.','Those names are Python modules; ENGINE is obtained separately from fc.ENGINE.')
finding(5,'B','The first imported component is used to run the rung2c component.','Only imports and constant assignments are visible; no such call is shown.')
finding(5,'C','A function is imported from a module.','The run_strat01_layer1_ffn_output_component_cross_input module itself is imported as fc.')
finding(5,'C','That function runs a component in a neural-network model.','The excerpt shows no function call or model execution.')
finding(5,'C','The script is located in the phase60 .h file.','That .h path is assigned to HEADER, not the location of the Python script.',True)
finding(6,'A','The most frequent token is selected among agreed-upon tokens.','FLOOR_ARM uses each arm\'s own most-frequent emitted token, without that restriction.')
finding(6,'C','The score averages the most frequent emitted tokens across all agreements.','It is a constant predictor for each arm, scored against the common reference, not an average across agreements.',True)

for s in 'AB':
    finding(8,s,'He files something called a return of the patent application.','The name after files what is called is clipped; return describes what he received earlier.',True)
finding(8,'C','The author is an exact duplicate of another man\'s patent.','The duplicate is a patent/document, not a human author.',True)
finding(8,'C','The citation is against the other man\'s patent.','The citation is against his returned application and cites the existing patent.',True)
finding(8,'C','He then files a patent application of his own.','The new filing\'s name/type is not visible; his initial application already existed.')
for s in 'ABC':
    finding(9,s,'The woman giving the kiss is identified as Pattie.','Pattie is named in the following comparison, but the speaker giving the kiss is not identified in the clipped excerpt.')
for s in 'AB':
    finding(9,s,'She threatens to cheat a man named Ikey.','Ikey occurs in a clipped phrase; the addressee of cheat you too is not identified as Ikey.')
finding(9,'A','The coincidence is between Pattie and the author\'s cousin Ikey.','It is between cousin Pattie and the narrator.',True)
finding(9,'A','The author gives Pattie the kiss.','She gives the narrator a kiss, reversing the action.',True)
finding(9,'C','The man she threatens to cheat is her fiance.','No fiance relation is stated.')
finding(9,'C','She is kissed by the man.','She gives the kiss to the narrator.',True)
finding(9,'C','That man is the person she is running away with.','The visible text does not identify an elopement partner for the kiss giver.')

for s in 'ABC':
    finding(10,s,'A female hunter loves Orion.','The strong hunter is Orion, the male object of the goddess\'s feelings.',True)
    finding(10,s,'She loves him deeply.','The passage qualifies this as as near to loving as she ever came, rather than asserting deep love.')
finding(10,'A','She never considered becoming Orion\'s wife.','The stated refusal is Orion never considering becoming a goddess\'s husband.',True)
finding(10,'A','She falls in love with the mortal princess.','Orion falls in love with that princess.',True)
finding(10,'A','Oenopion is a daughter of Chios.','Oenopion is King of Chios, an island, not its daughter.',True)
finding(10,'A','The king is asked for a marriage.','The object after asked the king fo is clipped and not specified.')
finding(10,'C','She never considers him a goddess.','The passage says Orion never considered being husband of a goddess.',True)
finding(10,'C','Orion asks the king for the princess and then falls in love.','The request is clipped; falling in love is stated before that request.')

for s in 'ABC':
    finding(11,s,'A man is currently conversing with a young girl.','The narrator\'s sex is not shown, and Miss Raby was a girl seven years earlier, not identified as one now.')
finding(11,'A','The boy\'s name is Dren.','dren is a clipped word ending, not a visible proper name.',True)
finding(11,'A','The narrator discusses his own writing about that boy.','The excerpt refers to a separate he always writing, without assigning the writing to the narrator.')
finding(11,'A','The man is unsure what he is talking about in this confusing conversation.','He reports having heard the old affairs confusedly; that is not confusion about his own speech.')
finding(11,'C','The man discusses his own writing about the death.','The writer and topic of writing are not established as narrator/death.')
finding(11,'C','He is surprised to hear the death from Miss Raby.','No surprise is expressed.')
finding(11,'C','The death is very recent and occurred seven years ago.','Seven years dates the affairs; the death\'s date is not established and recent is unsupported.',True)
for s in 'AB':
    finding(12,s,'The worn-out event takes place on April16(28).','Worn out precedes the new dated entry; the visible April16 entry begins with sleeping well.')
finding(12,'A','The messenger informs the narrator whether the narrator wanted anything.','The messenger comes to ask whether anything is wanted.')
finding(12,'A','The unspecified source shows American visitors are from a different country or region.','Neither visitors nor their origin is established by the redacted source.')
finding(12,'B','The April16 experience is in Paris.','Paris is the earlier contrasting impression, not the current location.',True)
finding(12,'C','The messenger is described as remarkable.','The remark describes Americans generally, not the messenger specifically.')

for s in 'AC':
    finding(13,s,'The woman is busy with daily tasks.','She busied herself with none of the preceding things.',True)
    finding(13,s,'She is eating literal food.','Digesting solid food is a metaphor for absorbing the truth being taught.')
    finding(13,s,'She is preparing church services.','The church/choir list does not show service preparation.')
    finding(13,s,'She is aware of or concerned about consequences of her actions.','No such consequence statement is visible.')
finding(13,'A','She declares church,choir,Sunday-school,prayer-meeting and stove all wrong.','The declaration is clipped before its predicate; wrong is not stated.',True)
finding(13,'B','Being busier makes her realize identity and dignity must be maintained.','Neither that causal relation nor identity/dignity appears; the visible lesson concerns no right to retreat into a shell.',True)
finding(13,'C','She is aware of her right to build a shell and enter it.','She learns she has no more right to do so.',True)
for s in 'AB':
    finding(14,s,'The waiter asks the guest whether he wants wine.','Abe volunteers to take wine; the waiter asks what kind and number.')
finding(14,'C','The Indian trader is British.','British is not specified.')
finding(14,'C','The waiter addresses the trader about copying Abe\'s wine.','Abe is the guest asking for the trader\'s wine, not the reverse.',True)
finding(14,'C','Abe is already drinking that wine.','He has just decided to order some; drinking is not shown.')
finding(14,'C','The trader points at the bottle.','Abe points at the trader\'s bottle.',True)
finding(14,'C','He wants the same amount of wine as Abe.','The question concerns wine kind/number, not matching an amount.',True)
for s in 'AC':
    finding(15,s,'The Duke is ten years old.','The narrator says I was then ten, not that the Duke was ten.',True)
    finding(15,s,'The young Duke is identified as Duke of Holstein.','Holstein identifies the Prince Guardian/Administrator; this Duke\'s title is not stated.')
    finding(15,s,'The Duke is present with Augustus and Anne at this scene.','Their presence is stated; the narrator hears a report about the Duke, without his presence being stated.')
finding(15,'B','The Duke is restless at the table.','Restive is visible, but its clipped continuation does not attach it to table; table attaches to intoxication.')

for s in 'AC':
    ambiguous(16,s,'The hash is of a binary32 addition operation.','The hash belongs to the resulting l_out tensor; operation may be shorthand for its result.')
for s in 'AB':
    finding(17,s,'A shipper pays for an arm.','Shipped describes an existing packed rule; E62 pays an owed experiment, not a commercial shipper.',True)
finding(17,'B','The missing3B cell is not included in the calculation.','E62 explicitly carries a third arm to pay that owed item; exclusion from a calculation is not stated.',True)
finding(17,'C','E62 pays for the pipeline.','The owed item is the3B experimental cell, not pipeline funding.')
finding(18,'B','Levels with L<=6 are excluded.','Only those levels enter the fit; L8 is excluded and L5 kept.',True)
for s in 'ABC':
    ambiguous(19,s,'The study measures a task estimator.','The excerpt says estimand; the response is vague and may be using estimator as loose task terminology.')
finding(20,'A','Structural reasons explain the BPB/ranking discrepancy.','Structural reasons explain why the old agreement number is not rerun, not the cause of the discrepancy.')
finding(20,'B','The rank partner is not verified at source.','The excerpt explicitly says verified at source, not assumed.',True)
finding(20,'B','The number of calls to e14_gn3_greedy.py is not rerun.','The number is agreement45.6%(73/160), not a call count.')
finding(20,'C','The number of times the rank metric was used is not rerun.','The unrepeated number is the measured agreement, not a usage count.')
for s in 'BC':
    finding(22,s,'The test found that the method cannot detect damage.','The excerpt prescribes a diagnostic control, not an observed failure.',True)
finding(22,'C','The method itself must be clearly worse than other methods.','The high-sparsity random/null armC must be worse than armsA/B, not the method itself.',True)
finding(23,'A','An oscillating job is invisible to the gate.','Mid-measurement job starts are explicitly invisible; the sentence about oscillating jobs is clipped before its finding.')
finding(23,'B','Clean guarantees that the process did not exceed1GB.','Clean means no process exceeded1GB at startup, not a check only on the engine process.')
ambiguous(23,'C','The binary starts with a clean state.','The state certified is specifically other processes\' memory at the starting instant, not every possible meaning of clean.')

DETAILS=[
    'Names Qwen2.5-1.5B or TERNARY as the visible measurement context.',
    'Names g_b and the conditional/error operation; B correctly names VOID and OUT.',
    'Names tolerance0.25 or the visible E48 sweep6/B.5 setting.',
    'Provides the ref_q key and its visible path or8*6144 size.',
    'Mentions the visible path/size/hash record or expected configuration keys.',
    'Names the imported run_strat01_layer1_ffn_output_component_cross_input module or the visible HEADER path.',
    'Identifies the most-frequent emitted token as the constant predictor.',
    'Describes is_unpinned string/marker check or reason_of whitespace removal.',
    'Mentions return of the application and the previously issued/published patent.',
    'Mentions preparation for elopement; A/B also mention her kiss.',
    'Mentions the mortal princess/Oenopion or the goddess not wanting Orion to leave.',
    'Names Miss Raby or the little boy\'s death, both visible.',
    'Mentions slept very well or the messenger enquiry.',
    'A preserves the no-right-to-build-a-shell detail; B/C have no concrete supported proposition beyond an unspecified woman.',
    'Mentions the bottle just called for by the Indian trader, despite actor errors.',
    'Mentions the Duke\'s drinking or the attendants preventing table intoxication.',
    'Correctly states internal repeat/fail-closed and pre-freeze check limitation.',
    'Mentions third arm/shipped packed rule or the3B/R3 owed cell.',
    'A/C give L<=6,L8 excluded,L5 kept; B correctly names6threads and L5 kept despite inverted inequality.',
    'Preserves the specific separation from the MTP pilot.',
    'Preserves the SCORE/RANK pairing rule or E14 source verification.',
    'Gives the supported -0.000016893BPB byte-conversion delta.',
    'A names the critical top-k by|h| mask; B/C convert prescribed controls into a supposed method failure and provide no correctly attributed specific detail.',
    'Preserves startup-only checking,before mode dispatch or mid-run blindness.'
]
MISSING={(13,'B'),(13,'C'),(22,'B'),(22,'C')}


def main():
    panel_path=B.DOC/'meth267_anonymous_response_panel.json';assert B.digest(panel_path)==PANEL_SHA
    panel=B.read(panel_path);assert len(panel['rows'])==24
    rows=[]
    for i,item in enumerate(panel['rows']):
        row={'source_id':item['source_id']}
        for side in 'ABC':
            row[side]={'unsupported':FINDINGS.get((i,side),[]),'ambiguous':AMBIGUOUS.get((i,side),[]),
                'missing_detail':(i,side) in MISSING,'detail_evidence':DETAILS[i]}
        rows.append(row)
    assert all(0<=i<24 and side in 'ABC' for i,side in (*FINDINGS,*AMBIGUOUS,*MISSING))
    out=B.DOC/'meth267_anonymous_verdict.json'
    B.write(out,{'experiment':'METH-267-committed-anonymous-excerpt-verdict','panel_sha256':PANEL_SHA,
        'reviewer':'single agent,arm identities withheld,map and labeled texts unopened',
        'rubric':'Frozen267/123 excerpt-only atomic unsupported claims; severe central entity/event/result inversion; any concrete supported detail qualifies; ambiguity separate.',
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows})
    print(json.dumps({'responses':72,'verdict_sha256':B.digest(out),'arm_mapping_consulted':False}))


if __name__=='__main__':main()
