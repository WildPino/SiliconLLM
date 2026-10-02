#!/usr/bin/env python3
"""Source-only literal anchors and supported claims; no model outputs accessed."""
import json
from pathlib import Path
import meth293_native_source_answerability as A

# Fixed source order from292. Claims are restricted to the visible384-character excerpts.
FINDINGS=(
 ('prefill8/ffn_out-1.full.f32le','The code records reference and C FFN dump-file paths with hashes and associated metadata.','The reference FFN output entry includes the value49152.','Visible dictionary entries name dump files and hash strings.'),
 ('smallest-|w| entries','The pruning function zeros a fraction of the smallest-magnitude weights separately in each layer/organ matrix.','The preceding restoration code copies saved weights back and clears its saved dictionary.','The docstring explicitly states the pruning rule and surrounding code shows restoration.'),
 ('invalid fresh source index','The extractor requires the range response to contain the complete capped index and decodes that index as UTF8 JSON.','Decode or JSON parsing failures raise ExtractionError.','Visible length check and exception clauses support the summary.'),
 ('AutoModelForCausalLM.from_pretrained','The helper resolves a backend enum and loads a model using the supplied repository, revision, dtype and attention implementation.','The model is moved to the requested device and put in evaluation mode.','The displayed function call and to(device).eval() are explicit.'),
 ('DRIFT_WINDOWS = [40, 1280]','The code defines drift windows and benchmark/NLL/config parsing patterns.','It lists E49 AVX4 reference medians121.65 and74.11 for windows40 and1280.','Constants and regex names are visible without inferring a new benchmark result.'),
 ('MIN_FREE_RAM_BYTES','The acquisition code validates a shard manifest and tokenizer identity, then checks available RAM against a minimum.','The RAM check treats an unavailable reading as insufficient.','The visible function calls and None-or-below-minimum condition support this.'),
 ('0 = per-shape default','The command-line parser accepts repetition, thread, token-count, smoke and output options.','Defaults include three repetitions, six threads and token count zero for per-shape defaults.','Argument declarations explicitly give the defaults and zero-token meaning.'),
 ('MALFORMED under the E4 precedent','The comment reports that a planted control caught a contradictory timing criterion before any cell ran.','It classifies the registration as malformed rather than a failed measurement.','The visible comment explains the monotone-decline versus W2-minus-W1 contradiction.'),
 ('Lucy\nArmytage','Sir Percy cannot escape the fascination Lucy Armytage has held over him since first sight.','He smokes fiercely while returning to his chambers.','The narrative explicitly gives his reaction and return; no identities outside the excerpt are assumed.'),
 ('Hathersage, Derbyshire, 1617','The passage lists labelled symbols and bibliography references.','One entry identifies a1617 church-bell legend at Hathersage in Derbyshire.','This is a list of references, not an unsupported narrative about their origins.'),
 ('going through the winter','The speaker connects current restraint and pleasure in working again with the difficult winter.','The speaker predicts the present situation will not last as work becomes heavy.','Visible dialogue supplies both the winter explanation and prediction without naming a setting.'),
 ('their\nnames are lost','The argument warns against concluding that totem kins vanished when the names of phratries are unknown.','Arunta subdivision names listed include Panunga, Bukhara, Purula and Kumara.','The passage distinguishes missing names from disappearance of the groups.'),
 ('Curly Locks dreamed','Curly Locks dreams of visiting the country, then her parents take her to her uncle after a few weeks.','Her cousin Harry is near her own age.','The family visit and cousin are directly stated; later activities are cut off.'),
 ('When they kilt him in the open','The dialect verse recounts the death of a hunted fox and the hunters celebrating around it.','The huntsman holds the fox up while the dogs bay.','The visible hunting scene supports a narrow summary without identifying a poem or author.'),
 ('point-blank destructive fire','Sigels small band mistakes approaching troops for victorious comrades, then receives destructive close-range fire.','They had waved their flags and expected friendly troops.','The contrast between mistaken comrades and the sudden fire is explicit.'),
 ('Mrs. Westenra','The narrator recalls a womans illness after a sleepwalking episode on the cliff and is asked to explain the event.','The narrator hopes it was right to keep the matter from MrsWestenra.','Visible narration supplies the illness, cliff event and withheld disclosure; other identities are not inferred.'),
 ('nessun gate','The status excerpt says batch shape and position/KV state need controlled checks and the paired run does not establish quality or a50-token/s engine gate.','The first25 output IDs match before the baseline chooses4734 at index25.','These are claims of the excerpt; the truncated MTP token value is not reconstructed.'),
 ('kqv_out-1','The audit locates a propagation failure at kqv_out-1 despite current SwiGLU/down gates passing.','It calls for checking ffn_norm-0 identity and says that audit closes the apparent opening without a run.','The excerpt states the failure boundary and cautions against rewriting already qualified parts.'),
 ('0.98\u20131.06 G','The text argues that a50-token/s active-weight budget is far smaller than the7B donors active weight count.','Even the all-ternary rung is described as6.9times short.','The numbers support the stated cost argument without treating it as a new present-day result.'),
 ('two\ncausal coordinates changed','The protocol distinguishes a new layer1 cross-input check from an earlier check stopped at kqv_out-1.','Reference-generic Q4 lowering had made block0 terminal output nearly exact.','The excerpt states changed coordinates and the earlier failure; it does not report a new pass.'),
 ('reusable scalar C row-matvec','The brief proposes moving engine fidelity from descriptors to numerical use of actual payload bytes by a reusable scalar C row-matvec.','Donor, artifact, quantization, graph and quality evidence are held fixed.','Changed-coordinate and hypothesis paragraphs are proposals, not executed result claims.'),
 ('all 32 complete layer-1','The protocol asks whether the ordinary GGUF path passes block0 and all32 complete layer1 checkpoints in both schedules.','It cites a closed SSE2 repair byte-exact on immutable reference gate/up operands.','The opening is a question; the cited narrow repair is distinguished from production integration.'),
 ('K148-STATIC','The brief charges K148-STATIC zero per-token scoring cost and says it is cheaper if it ties the ridge.','A base control must meet160/160 on both metrics or the run is void.','The cost argument is conditional and the displayed gate explicitly defines the void condition.'),
 ('solo l\u0027export del formato W4-v2','The report describes only a W4-v2 export at the fixed donors BF16 scale.','It explicitly excludes forward, calibration, held-out and T4 and does not establish loading or model quality.','The excerpt sharply limits the reported outcome to export/format.'),
)

def main():
    assert A.digest(A.MANIFEST)==A.MANIFEST_SHA
    manifest=json.loads(A.MANIFEST.read_text(encoding='utf-8'));assert len(manifest['items'])==len(FINDINGS)==24
    rows=[]
    for item,(anchor,summary,detail,basis) in zip(manifest['items'],FINDINGS):
        assert anchor in item['excerpt'] and 4<=len(anchor)<=128
        assert 12<=len(summary)<=240 and 8<=len(detail)<=240 and len(basis)>=12
        rows.append({'source_id':item['source_id'],'category':item['category'],'answerable':True,
                     'anchor':anchor,'supported_summary':summary,'supported_detail':detail,'basis':basis})
    out=A.DOC/'meth293_native_source_answerability_annotations.json';assert not out.exists()
    out.write_text(json.dumps({'manifest_sha256':A.MANIFEST_SHA,'model_outputs_consulted':False,'rows':rows,
        'script_sha256':A.digest(Path(__file__))},indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({'annotation_sha256':A.digest(out),'rows':len(rows)}))

if __name__=='__main__':main()
