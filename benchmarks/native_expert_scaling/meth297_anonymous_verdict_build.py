#!/usr/bin/env python3
"""All source-excerpt findings entered while only anonymous A/B/C are visible."""
import json
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
PANEL=DOC/'meth297_native_semantic_panel.json'
PANEL_SHA='9ae7f30e2348bbeb48080b2e3006a38068e216598ecd2900144f672285cc28d1'

def finding(detail,claims=(),ambiguous=(),missing=False):
    return {'unsupported':[{'claim':c,'evidence':e,'severe':s} for c,e,s in claims],
            'missing_detail':missing,'detail_evidence':detail,
            'ambiguous':[{'claim':c,'evidence':e} for c,e in ambiguous]}

FINDINGS={
0:{
 'A':finding('The response explicitly names the visible ref_ffn_out key;49152 and the first hash are also present, though misinterpreted as shape.',[
  ('t is a TensorFlow tensor/operation.','Visible values are file-path/hash/count tuples; no TensorFlow object or operation is shown.',True),
  ('ref_ffn_out is a TensorFlow tensor/operation.','The visible ref_ffn_out tuple names a .f32le path,hash and49152; no TensorFlow tensor is shown.',True),
  ('The first tensor has shape(hash,49152).','A hexadecimal hash and count accompany a path; no tensor shape is declared.',True),
  ('The second tensor has a different shape.','Both visible complete entries carry49152; no differing shapes are provided.',False)]),
 'B':finding('The two prefill8 paths ffn_shexp-1.full.f32le and ffn_out-1.full.f32le are quoted correctly.',[
  ('t is a TensorFlow Prefill model.','The excerpt contains a file metadata tuple, not a TensorFlow model.',True),
  ('ref_ffn_out is a TensorFlow Prefill model.','The excerpt contains a file metadata tuple, not a TensorFlow model.',True),
  ('c_moe_out is a TensorFlow Prefill model.','The c_moe_out value begins with a C_ROOT file path; no model type is established.',True),
  ('The C file path is C_ROOT/ffn_moe_out-1.f32.','The visible basename is prefill8_ffn_moe_out-1.f32.',False),
  ('The models belong to a larger set of Prefill models.','No model collection or further models are described in the excerpt.',False)]),
 'C':finding('The response correctly identifies the visible prefill8 directory and file paths with metadata, despite its subsequent format/name inventions.',[
  ('The first file is a JSON object with the stated properties.','The excerpt records paths/hash/count tuples, not file contents or a JSON property schema.',True),
  ('A file path is ref_ffn_shexp-1.full.f32le.','The visible path is prefill8/ffn_shexp-1.full.f32le,without ref_ in the basename.',False),
  ('A file path is ref_ffn_out-1.full.f32le.','The visible path is prefill8/ffn_out-1.full.f32le,without ref_ in the basename.',False),
  ('A file path is c_moe_out-1.f32.','c_moe_out is a key; the visible basename is prefill8_ffn_moe_out-1.f32.',False),
  ('The second file is a JSON object with the same properties.','No second file contents or matching JSON schema are shown.',True)])},
1:{
 'A':finding('It correctly states pruning zeros the frac smallest-|w| entries.',[
  ('prune_unstructured copies the weight matrix to a new matrix.','The preceding separate restoration loop copies saved weights into existing model weights; the prune function only retrieves W in the visible body.',True)]),
 'B':finding('It accurately lists model,layers,organ,frac,snap and states independent smallest-|w| pruning.',[
  ('prune_unstructured copies weights from saved state to current state.','The saved-state restoration precedes the def line and is outside prune_unstructured.',True)]),
 'C':finding('It accurately lists model,layers,organ,frac,snap and states independent smallest-|w| pruning.',[
  ('prune_unstructured copies weights from saved state to current state.','The saved-state restoration precedes the def line and is outside prune_unstructured.',True)])},
2:{
 'A':finding('ExtractionError and its invalid-data message are explicitly present.',[
  ('The JSON is decoded to a Python dictionary.','json.loads is shown but the input JSON contents/type are not; it need not yield a dictionary.',False)],
  [('Decoded data does not match expected format.','This can mean invalid JSON/UTF8,which the excerpt does check; no separate postparse schema test is shown.')]),
 'B':finding('It correctly says invalid JSON raises ExtractionError with an error message.',[
  ('The JSON is decoded into a Python dictionary.','No JSON payload/type is shown; json.loads alone does not establish a dictionary.',False)]),
 'C':finding('It correctly states the complete capped-index range check and invalid-JSON ExtractionError.')},
3:{
 'A':finding('Repository,revision,dtype and device/evaluation-mode settings are explicit from_pretrained/to/eval arguments.',[
  ('run_one returns an AutoModelForCausalLM instance.','The excerpt assigns the instance to m; no return from run_one is visible.',True)]),
 'B':finding('Its listed run_one parameter names match the visible signature.',[
  ('run_one returns the name attribute of AutoModelForCausalLM from a backend.','The earlier return getattr(_BACKEND_ENUM,name) is in a preceding helper; run_one instead loads a model into m.',True)]),
 'C':finding('Its listed run_one parameter names match the visible signature.',[
  ('run_one returns the name attribute of AutoModelForCausalLM from a backend.','The earlier return getattr(_BACKEND_ENUM,name) is in a preceding helper; run_one instead loads a model into m.',True)])}
}
FINDINGS.update({
4:{
 'A':finding('It correctly identifies re.compile regular-expression matching/extraction alongside flags and window sizes.',ambiguous=[('Configuration file for a model.','The visible material is benchmark/configuration parsing code; configuration file is loose framing rather than a demonstrated external file format.')]),
 'B':finding('The BENCH regex visibly captures token count,seconds and tok/s,as the response states.',ambiguous=[('Configuration file for a model.','The visible material is benchmark/configuration parsing code; configuration file is loose framing.')]),
 'C':finding('It correctly identifies re.compile regular-expression matching/extraction alongside flags and window sizes.',ambiguous=[('Configuration file for a model.','The visible material is benchmark/configuration parsing code; configuration file is loose framing rather than a demonstrated external file format.')])},
5:{
 'A':finding('The available_ram None/below-MIN_FREE_RAM_BYTES check supports the stated minimum-RAM requirement.'),
 'B':finding('It correctly states shard-manifest/tokenizer processing and the minimum-RAM check.',[
  ('Expected payload capacity is100MB.','Only expected_payload_cap is shown; no numeric capacity appears.',False),
  ('Expected asset SHA256 is0x1234567890ABCDEF.','Only expected_tokenizer_asset_sha256 is shown; no hash value appears.',False),
  ('The tokenizer loader is a custom loader.','tokenizer_loader is passed but its implementation or origin is not shown.',False)]),
 'C':finding('The response accurately notes manifest/tokenizer processing and the minimum-RAM check.')},
6:{
 'A':finding('The --reps3,--threads6 and --tokens0 defaults are correctly stated.',[
  ('The Python script is named r_engine_e25.exe.','A .exe path fragment is visible; it does not identify the enclosing Python script.',True),
  ('--smoke defaults toTrue.','action=store_true is shown,not a defaultTrue; the flag enables the mode when provided.',False)]),
 'B':finding('The listed reps/threads/tokens/smoke options and result-output path are visible.',[
  ('The Python script is named r_engine_e25.exe.','The excerpt shows an executable-path fragment,not the Python script identity.',True),
  ('reps defaults to0.','--reps explicitly defaults to3.',False),
  ('threads defaults to0.','--threads explicitly defaults to6.',False)]),
 'C':finding('The --out os.path.join(HERE,results,e25_rank_cost.json) default is correctly quoted.',[
  ('The Python script is named r_engine_e25.exe.','The .exe path fragment does not identify a Python script.',True),
  ('The parser takes up to3 arguments.','The response itself lists four options and --out; the excerpt also shows more than3.',False)])},
7:{
 'A':finding('The W2-W1 negative/large-difference detail is present,although its interpretation is wrong.',[
  ('A cell ran with the described self-contradictory series.','The excerpt identifies a criterion registered BEFORE ANY CELL RAN; no such measured series is supplied.',True),
  ('The series is self-contradictory because negative W2-W1 violates monotonicity.','A monotone decline has a negative difference; the conjunction in the registered rule is malformed,not the monotone series.',True)]),
 'B':finding('It accurately quotes the registered conjunction and explains the negative W2-W1 under decline.',[
  ('A cell ran with the described monotone decline in W2.','The excerpt concerns a rule registered before any cell,not a reported observed W2 series.',True),
  ('The cell was registered on the first run.','The brief registered TIME before cells ran; planted controlB-1 caught the malformed rule on the first run.',True),
  ('The problem is the system monotonicity detection.','The excerpt calls the registered conjunction MALFORMED,not a detection failure.',True)]),
 'C':finding('It retains the concrete negative/large decline detail from the excerpt.',[
  ('A cell ran with the described series.','The excerpt explicitly places criterion registration before any cell ran and provides no observed such series.',True),
  ('The time series itself is self-contradictory.','The registered conjunction is self-contradictory; a genuine monotone decline is not.',True)],
  [('Cannot satisfy the monotone condition.','This can refer to the malformed combined rule,which indeed cannot be satisfied; it is not necessarily a separate claim that negative decline is nonmonotone.')])}
})

FINDINGS.update({
8:{
 'A':finding('Sir Percy smoking fiercely is a directly supported concrete detail.',[
  ('The girl is being pursued by Sir Percy.','The excerpt shows fascination and his return to chambers,not a pursuit action.',False)],
  [('The girl has the fine flower of civilisation.','The parenthetical identifies Eleanor as the likely referent; attachment of the metaphor is imprecise rather than an unambiguous separate inversion.')]),
 'B':finding('Sir Percy smoking and returning to his chambers,and Lucy commanding his attention,are explicitly present.',[
  ('The compared girl is Eleanor Chantrey.','The girl is compared WITH Eleanor; Eleanor is the counterpart,not that girl.',True),
  ('The fine flower of civilisation refers to Sir Percy.','The excerpt explicitly names Eleanor Chantrey as that fine flower.',True),
  ('Lucy Armytage is the same person as the previously identified Eleanor.','Eleanor is the comparison counterpart; no equivalence of these named women is shown.',True)]),
 'C':finding('It retains Eleanor Chantrey as the comparison counterpart.',[
  ('The comparison concerns the girls intelligence.','No intelligence criterion is stated.',False),
  ('The comparison concerns standing up to the influence of a powerful woman.','No such power/influence/resistance is stated.',False)],
  [('The girl has the fine flower of civilisation.','The parenthetical names Eleanor; metaphor attachment is ambiguous.')])},
9:{
 'A':finding('Hathersage,Derbyshire and1617 are visible;Ilios/Grammar of the Lotus/Migrations and Symbols are named sources.',[
  ('The Scandinavian symbols are found on that church bell.','X identifies Scandinavian symbols and Y separately identifies the church-bell legend.',True),
  ('The name of the church bell is supplied.','Only location and year are supplied,not a bell name.',False),
  ('Count Goblet dAlviella is a work with an author.','Count Goblet dAlviella is the named contributor preceding The Migrations and Symbols,not a work title.',True)]),
 'B':finding('The church-bell location/year and named Ilios/Grammar of the Lotus/Migrations and Symbols citations are present.',[
  ('The Scandinavian symbols are found in the Hathersage church bell.','X and Y are distinct caption entries,not one linked artifact.',True),
  ('The church bells name is given.','The excerpt gives location/year,not a bell name.',False),
  ('H.Colley March is a work with an author.','H.Colley March is a credited contributor,not a work title.',True)]),
 'C':finding('Ancient Scandinavian symbols and a church-bell legend in Hathersage/Derbyshire are correctly mentioned.',[
  ('The excerpt reports a significant archaeological find.','It lists symbol-caption sources; no discovery event is reported.',True),
  ('The archaeological find dates to the16th century.','No such find/date is shown;1617 is the church-bell legend date and is not16th century.',True),
  ('The excerpt gives a detailed manuscript description.','No manuscript appears.',True),
  ('A manuscript was found in the18th century.','No manuscript discovery or18th-century date appears.',True)])},
10:{
 'A':finding('The group is pleased to be at work again and out on the plains;heavy work is anticipated.',[
  ('They have been through a winter of hard work.','The source says winter and hard times,not winter labor.',False),
  ('Men do not get enough rest.','The excerpt ends at men dont get;the missing object/rest is not provided.',False)]),
 'B':finding('Some have had tough times and they are pleased to work again on the plains.',[
  ('People are currently working hard and occasionally experiencing setbacks.','Current work hardship/setbacks are not described;winter hardship is past and heavy work is forecast.',False),
  ('The speaker has been through a winter of hard work.','Winter/hard times do not establish winter labor.',False)]),
 'C':finding('Being pleased to be back at work on the plains and the forecast that this will not last are present.',[
  ('They have been through a winter of hard work.','The source says winter/hard times,not winter labor.',False)])},
11:{
 'A':finding('It retains the specific lost-names/two-extant-totem-kins motif,though its relationship is confused.',[
  ('The classes later develop into separate groups.','Classes are called a relatively late development;no subsequent transformation into other groups is shown.',False),
  ('The totem kins of phratriac names may have had two extant totem kins.','It is the PHRATRIES that may have been named after two extant totem kins;the response changes the relationship.',True)]),
 'B':finding('It correctly retains uncertainty about vanished kins,possible names from two extant kins and lost phratry names.',[
  ('These groups are not necessarily related to each other.','The excerpt does not discuss their relatedness.',False)]),
 'C':finding('It retains class grouping and the fact that phratry names are unknown.',[
  ('Classes later develop into separate groups.','The source calls classes a relatively late development,not a further group transformation.',False),
  ('The totem kins have not vanished.','The source says we CANNOT SAY they vanished,not that nonvanishing is established.',True)])},
12:{
 'A':finding('Curly Locks visits her uncle after her parents take her,and cousin Harry is explicitly named.',[
  ('They often went fishing.','The excerpt is clipped at went f;fishing is not supplied.',False),
  ('They played in the country.','No playing activity is supplied.',False)]),
 'B':finding('Curly Locks dreams of the country and her parents take her to her uncle weeks later.',ambiguous=[
  ('What she liked best about her uncles visit.','Earlier wording correctly identifies HER visit to the uncle;the possessive is loose rather than an unequivocal opposite travel event.')]),
 'C':finding('Curly Locks dreaming of the country,parental transport and cousin Harry are directly supported.',[
  ('The author had the dream.','Curly Locks,not the narrator/author,is explicitly the dreamer.',True),
  ('Harry often accompanied her to the country.','Harry is met during her stay;the subsequent often went f activity is clipped and is not a country-arrival statement.',False)])},
13:{
 'A':finding('A hunting scene involving the fox and huntsman is supported.',[
  ('The fox is caught in a daisy chain.','The fox is killed in the open and then lifted;no daisy chain exists.',True),
  ('The huntsman catches him using a dugg and a dugg.','Dogs bay around the lifted fox,but no such capture method is given.',False)]),
 'B':finding('The fox/huntsman hunting scene is supported.',[
  ('The fox is caught in a daisy chain.','The source has the fox killed in the open;no chain exists.',True),
  ('The huntsman catches him with a dugg and a goblet.','No goblet or such capture method is given.',False)]),
 'C':finding('No concrete supported hunting detail:the output only substitutes a fox/rabbit chase and dodging for the depicted kill/huntsman/dogs.',[
  ('The fox is chasing a rabbit.','The fox is the killed quarry of the hunt;no rabbit appears.',True),
  ('The fox dodges repeatedly.','No fox-dodging action is given.',False)],missing=True)},
14:{
 'A':finding('Flags waved to supposed victorious comrades and advancing-troop fire leaving dead/dying are explicit.',[
  ('Sigels soldiers fire upon Lyons troops.','The advancing troops fire UPON Sigels band;Lyon is only in the clipped adjoining-hills expectation.',True)]),
 'B':finding('The concrete point-blank fire and dead/dying-ground details are present despite inverted agency.',[
  ('Sigels men attack Lyon and use the destructive fire.','Sigels band is the recipient of the advancing-troop fire,not its stated attacker.',True),
  ('The fire signals their victory.','No Sigel victory is reported;his band is hit by the destructive fire.',True)]),
 'C':finding('Flags and sudden advancing-troop point-blank fire are visible.',[
  ('Sigels men fire upon Lyons troops.','The advancing troops fire UPON Sigels band;no Sigel-on-Lyon fire is stated.',True),
  ('The flag recipients are their victorious comrades.','The source qualifies this as what they SUPPOSED,followed by destructive fire upon them.',True)])},
15:{
 'A':finding('Not telling Mrs.Westenra and the cliff/sleep-walking incident are explicit source details.',[
  ('The narrator is the person made ill after the cliff night.','The narrator describes how ill SHE was,another woman.',True),
  ('The narrator forgot her own sleep-walking adventure.','The narrator almost forgot the other womans subsequent illness,not her own adventure.',True),
  ('The man is her husband.','No husband relationship is stated.',True),
  ('She told him about it as an established fact.','She MUST HAVE told him is the narrators inference,not reported direct knowledge.',False)]),
 'B':finding('The distinctive sleep-walking/cliff incident remains recognizable despite its wrong participants/direction.',[
  ('The narrator is the woman made ill.','The source distinguishes I from the woman referred to as she.',True),
  ('She forgot her own adventure.','The narrator nearly forgot the other womans illness.',True),
  ('The man is her husband.','No spouse relationship is provided.',True),
  ('The husband told her the incident.','The narrator infers SHE told HIM,then HE asks the narrator to explain.',True),
  ('She is eager to tell him to understand his perspective.','HE wants her account so HE may understand;her stated hope concerns nondisclosure to Mrs.Westenra.',True)]),
 'C':finding('It retains the sleep-walking cliff episode and Mrs.Westenra nondisclosure concern.',[
  ('The main narrator forgot her own illness.','I almost forgot how ill SHE was;the illness belongs to the other woman.',True),
  ('Her husband visited her.','No visit event is described.',True),
  ('The man is her husband.','No spouse relationship is stated.',True),
  ('She revealed that she had told him her adventure.','The narrator only infers that the OTHER woman told him;no such revelation is shown.',True),
  ('She hopes she did not say anything to Mrs.Westenra.','The hope is whether nondisclosure was RIGHT,not whether nondisclosure occurred.',False)])}
})

FINDINGS.update({
16:{
 'A':finding('It correctly retains not clean-box/interleaved and no passed>=50tok/s/engine.c gate.',[
  ('The described process is checkpointing.','The source discusses a paired run and temporary checkout diagnostics,not saving/restoring model checkpoints.',True)]),
 'B':finding('Not clean-box/interleaved,no passed>=50tok/s/engine.c gate and audit details/log reference are visible.',[
  ('The described process is checkpointing.','The excerpt discusses paired/state/checkout diagnostics,not checkpointing.',True)]),
 'C':finding('It retains the concrete lack of clean-box/interleaved evidence and the audit-log context.',[
  ('The process is checkpointing.','The source concerns a paired run and temporary code checkout,not model checkpointing.',True),
  ('The checkpointing process fails required quality standards.','Not a quality proof and no passed speed/engine gate do not report an observed quality failure.',True),
  ('MTP4 is a setting.','The clipped text ramo MTP4 starts the alternative output-ID value;no MTP4 setting is stated.',False)])},
17:{
 'A':finding('ffn_norm-0,Rung-2B cross-input comparison and identity-audit closure without a run are explicitly present.',[
  ('SwiGLU means Synthetic Gradient Language Unit.','The excerpt does not supply that expansion.',True),
  ('The issue lies in SwiGLU/Q6/layer1 propagation.','Current reference/reference SwiGLU passes;the text says not to rewrite these and locates first failure at kqv_out-1.',True),
  ('kqv_out-1 is reference/reference current SwiGLU.','The text separates that propagated failure from the passing current SwiGLU.',True)]),
 'B':finding('It correctly states reference/reference current SwiGLU passes and identity-audit closure without a run.',[
  ('The failing stage is SwiGLU/down and a SwiGLU/Q6/layer1 architecture problem.','These local gates pass;first propagated failure is kqv_out-1 and the instruction is do not rewrite SwiGLU/Q6/layer1.',True)]),
 'C':finding('It correctly states reference/reference current SwiGLU passes and identity-audit closure without a run.',[
  ('The failing stage is SwiGLU/down and a SwiGLU/Q6/layer1 architecture problem.','These local gates pass;first propagated failure is kqv_out-1 and the instruction is do not rewrite SwiGLU/Q6/layer1.',True)])},
18:{
 'A':finding('It retains the concrete statement that quality-improving rungs are slower.',[
  ('Conversion speed is actually50tok/s.','50tok/s is the target budget;the fastest all-ternary rung is explicitly short of it.',True),
  ('That50tok/s model can activate7.07G ternary weights per token.','The budget permits only.98-1.06G;7.07G is the donors incompatible active weight count.',True),
  ('The7B donor runs at6.9times the speed.','The fastest rung is6.9times SHORT of the target,not faster.',True),
  ('The target is98-106% accuracy rather than100%.','The source gives.98-1.06G active weights and9.8-10.6% model fraction,not any accuracy target.',True)]),
 'B':finding('It correctly retains.98-1.06G permissible active weights,donor7.07G,and all-ternary6.9x short with slower quality rungs.',[
  ('Conversion speed is actually50tok/s.','50tok/s is the speed target/budget,not achieved conversion performance.',True),
  ('The target is merely improving output quality rather than reaching a specific level.','The excerpt analyzes the unreachable50tok/s target;it does not replace it by generic quality improvement.',True)]),
 'C':finding('It retains.98-1.06G permissible active weights and the6.9x shortfall.',[
  ('Conversion speed is actually50tok/s.','50tok/s is a target budget,not observed performance.',True),
  ('The7B donor itself is the fastest ladder point.','The ALL-TERNARY conversion rung is described as the fastest point,not the original donor.',True),
  ('The target is generic quality improvement rather than a specific level.','The text analyzes inability to reach the stated50tok/s target,not an alternative generic-quality goal.',True)])},
19:{
 'A':finding('The earlier pre-reference/pre-F16 state and Q4 lowering making block0 nearly exact are explicit.',[
  ('The newly pinned GGML_CPU_GENERIC F1 coordinate stops at kqv_out-1.','The second coordinate is clipped at F1;the kqv stop belongs to the EARLIER start-state experiment.',True)]),
 'B':finding('The earlier pre-reference/pre-F16 state and Q4 lowering making block0 nearly exact are explicit.',[
  ('The newly pinned GGML_CPU_GENERIC F1 coordinate stops at kqv_out-1.','The second coordinate is clipped at F1;the kqv stop belongs to the EARLIER start-state experiment.',True)]),
 'C':finding('It correctly retains reference-generic Q4 lowering making block0 terminal output nearly exact.',[
  ('The NEW design uses the pre-reference-generic/pre-F16-reduction state and stops at kqv_out-1.','The excerpt assigns those properties to the EARLIER run,before the changed causal coordinates.',True),
  ('The pinned GGML_CPU_GENERIC F1 coordinate stops at kqv_out-1.','No such stopping outcome is stated for that clipped second coordinate.',True)])}
})

FINDINGS.update({
20:{
 'A':finding('The descriptor-only to real-payload scalar C row-matvec change is concretely retained.',[
  ('The accepted weights changed.','Donor,artifact,quantization and model graph remain fixed;ENGINE FIDELITY is the changed coordinate.',True),
  ('The hypothesis remains unchanged.','No earlier hypothesis or unchanged-hypothesis assertion is supplied.',False),
  ('The hypothesis concerns fidelity of model predictions.','The stated hypothesis starts with a preregistered real tensor;this stage tests numerical payload/operator consumption,not model predictions.',True)]),
 'B':finding('It correctly retains engine-fidelity change to real payloads via scalar C row-matvec and fixed quality evidence.',[
  ('The hypothesis itself was modified from descriptor inspection to numerical consumption.','That is explicitly the changed COORDINATE;no old/new hypothesis comparison is shown.',False)]),
 'C':finding('The scalar C row-matvec/real-payload versus descriptor-only coordinate change is correct.',[
  ('The hypothesis remains unchanged.','The excerpt states a hypothesis but supplies no unchanged-hypothesis comparison.',False)])},
21:{
 'A':finding('The named POST_F16_BLOCK0_SWIGLU_SSE2_EXACT_REPAIR on immutable gate/up operands is explicitly established.',[
  ('These checkpoints/frozen schedules manage model training.','They concern engine production-integration/numerical checkpoints,not a model-training process.',True)]),
 'B':finding('The closed SSE2 result and byte-exact repair on immutable reference gate/up operands are concrete supported details.',[
  ('The path is a frozen schedule.','The path is tested under TWO frozen schedules;it is not itself a schedule.',False),
  ('The path passes block0 and all32 complete layer1 checkpoints.','The source asks whether it DOES pass;it does not report that outcome.',True)]),
 'C':finding('The closed SSE2 result and byte-exact repair on immutable reference gate/up operands are concrete supported details.',[
  ('The path is a frozen schedule.','The path is tested under TWO frozen schedules;it is not itself a schedule.',False),
  ('The path passes block0 and all32 complete layer1 checkpoints.','This is the open question,not a reported pass.',True)])},
22:{
 'A':finding('g1p(mass) being declined in E23 and disclosed is explicitly present.',[
  ('The declined g1p(mass) variant is the zero-charge,no-per-token-score object registered here.','The0M/no-score/preregistration statement concerns K148-STATIC,not the separate declined g1p(mass) entry.',True)]),
 'B':finding('The g1p(mass) declined-in-E23/disclosed detail is present.',[
  ('g1p(mass) is charged0M rather than11M and needs no per-token score.','Those properties are explicitly assigned to K148-STATIC.',True)]),
 'C':finding('The g1p(mass) declined-in-E23/disclosed detail is present.',[
  ('g1p(mass) is charged0M rather than11M and needs no per-token score.','Those properties are explicitly assigned to K148-STATIC.',True)])},
23:{
 'A':finding('It correctly retains export-only authorization without forward/calibration/heldout/T4.',[
  ('The donor is named Fissato.','fissato is the Italian adjective fixed,not a donor identity.',True),
  ('The format is exported to BF16 rather than W4-v2 with BF16 scales.','W4-v2 a scala BF16 specifies the quantized W4-v2 representation with BF16 scale,not conversion to a full BF16 output format.',True),
  ('The result is not a complete export.','The source explicitly heads the result export completo;its distinction is export versus loading/quality,not incomplete versus complete export.',True)]),
 'B':finding('It accurately states only W4-v2 export/no forward/calibration/heldout/T4 and format/export rather than loading/model-quality evidence.',ambiguous=[
  ('BF16 scalability.','The source says BF16 scale precision;scalability is loose wording without an explicit model-size scaling claim.')]),
 'C':finding('It accurately states only W4-v2 export/no forward/calibration/heldout/T4 and format/export rather than loading/model-quality evidence.',ambiguous=[
  ('BF16 scalability.','The source says BF16 scale precision;scalability is loose wording without an explicit model-size scaling claim.')])}
})

def main():
    assert hashlib.sha256(PANEL.read_bytes()).hexdigest()==PANEL_SHA
    panel=json.loads(PANEL.read_text(encoding='utf-8'))
    assert set(FINDINGS)==set(range(24)), 'All24 sources/all72 responses must be reviewed before final construction.'
    out=DOC/'meth297_native_anonymous_verdict.json';assert not out.exists()
    rows=[{'source_id':r['source_id'],**FINDINGS[i]} for i,r in enumerate(panel['rows'])]
    assert all(set(r)=={'source_id','A','B','C'} for r in rows)
    out.write_text(json.dumps({'panel_sha256':PANEL_SHA,'rubric':'Unchanged excerpt-only atomic unsupported/severe/missing-detail; ambiguity descriptive; arm map unseen.','rows':rows},indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({'reviewed_sources':len(rows),'reviewed_responses':72,'verdict_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
