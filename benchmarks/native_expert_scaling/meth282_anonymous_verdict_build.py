#!/usr/bin/env python3
"""Fixed excerpt-only findings; no arm map or labeled generations are read."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
PANEL=DOC/'meth282_anonymous_response_panel.json'
PANEL_SHA='fa9c56f1e47db1d6644c4b3f9a485872160f9027595ce83b45f371ecd840612b'


def u(claim,evidence,severe=False):
    return {'claim':claim,'evidence':evidence,'severe':severe}


def a(claim,evidence):
    return {'claim':claim,'evidence':evidence}


def f(detail,unsupported=(),ambiguous=(),missing=False):
    return {'unsupported':list(unsupported),'ambiguous':list(ambiguous),
            'missing_detail':missing,'detail_evidence':detail}


DATA=[
    [f('Model Path argument/default MODEL and output-directory switch are visible.',[
        u('The recovery flag recovers existing models.','--recover-existing is clipped at its action;what existing object it recovers is not stated.')]),
     f('The named model/output/apparatus/recovery switches are in the excerpt.',[
        u('It parses arguments to construct a parser object and uses that to create an argument parser.','ArgumentParser is constructed before add_argument;no parsing/creation-from-parsed-values is shown.',True)]),
     f('The named command-line switches are visible.',[
        u('It then parses arguments to perform a specific task.','The fragment only constructs/adds parser arguments;neither parsing nor a task execution is visible.')])],
    [f('The selected E67/E68 held-out token hash mismatch is explicitly named.',[
        u('The raised exception is ValueError.','The visible statement raises RuntimeError.',True),
        u('The error is raised by _FitCapture.','The raise precedes _load_labels and later _FitCapture construction.',True),
        u('labels is a list of tensors.','The return type of _load_labels is not specified.'),
        u('labels_meta is a list of tensors.','The metadata return type is not specified.'),
        u('The captures dictionary contains tensors for each layer.','Its values are constructed _FitCapture objects,not shown as tensors.'),
        u('The hash error is raised during inference.','The raise occurs before the later inference_mode block.',True)]),
     f('RuntimeError on the selected E67/E68 token hash mismatch is visible.',[
        u('Labels are stored in a dictionary.','_load_labels returns labels but its storage type is not shown.'),
        u('Captures are loaded from the model.','The dictionary constructs _FitCapture objects;no capture-loading operation is visible.'),
        u('It loads labels and captures for a specific sequence.','Label loading/capture construction precede the sequence loop and take layers,not a sequence argument.')]),
     f('The inference_mode block and sequence loop are visible.',[
        u('Labels are stored in a dictionary.','Their type is not shown by the _load_labels assignment.'),
        u('Captures are loaded from the model.','The dictionary constructs _FitCapture objects rather than loading captures.'),
        u('It captures model output for each sequence.','The loop body is clipped at captures.valu;no output capture execution is visible.')])],
    [f('Odd T and handling a leftover plane are concrete visible conditions.',[
        u('The multiple of32 is a byte count.','The opening fragment gives no unit/subject for the multiple of32.'),
        u('Only the high nibble of the leftover plane is loaded.','LOW nibble ONLY;high nibble NEVER LOADED is the explicit opposite.',True)]),
     f('LOW nibble and never loading HIGH into a shuffle are explicit.',[
        u('The multiple of32 is a byte count.','The clipped opening does not identify its unit.')],[
        a('Tp-1 is called the previous byte-plane.','The excerpt calls tb=Tp-1 the final plane;previous relative to what is unclear,so no definite inversion is counted.')]),
     f('The final-plane LOW nibble and [0,T/2) pair loop are visible.',[
        u('The multiple of32 is a byte count.','The opening fragment does not identify its unit.')],[
        a('When the code is odd.','The source says T is odd;code could loosely mean its length/count or an individual value. The omitted referent is ambiguous.')])],
    [f('Greater score wins and equal scores prefer the smaller ID;both are explicit.'),
     f('The greater-score comparison is visible despite the wrong tie rule.',[
        u('better returns a choice.','It returns an int comparison predicate,not a Choice object.',True),
        u('Equal scores prefer a higher ID.','The predicate is a.id < b.id.',True),
        u('The new choice is always the best one.','insert can insert below better existing entries or not insert;it does not guarantee every new choice is best.',True)]),
     f('The greater-score comparison is visible despite the wrong tie rule.',[
        u('better returns a choice.','It returns an int predicate,not a Choice object.',True),
        u('Equal scores prefer a higher ID.','The explicit tie comparison selects the lower ID.',True),
        u('The new choice is always best.','insert finds a position only where c is better than an existing entry;higher entries may remain.',True)])],
    [f('Sorting r[layer] values for the depth section is explicitly shown.',[
        u('Every processed dictionary has both delta and layer.','delta is accessed through rr[l];layer is accessed in separate depth_sweep rows. A shared record schema is not shown.'),
        u('The first loop iterates dictionaries and appends a value per dictionary.','The visible loop iterates levels and indexes rr by l;no dictionary-iteration list is given.'),
        u('Its initial and depth tables print layer and delta fields.','The initial row prints o/ax and level deltas;the depth header shows layer/attention columns,not delta.',True)]),
     f('Formatting delta to four decimal places is explicit.',[
        u('lls is a list of dictionaries traversed by the loop.','lls starts empty;the visible loop traverses levels and writes cells.'),
        u('The final section is triggered by a dictionary containing a layer key.','The explicit condition is d.get(depth_sweep),not presence of layer.',True),
        u('It sorts dictionary keys and prints those keys and values.','It sorts a set of r[layer] values,not dictionary keys.',True)]),
     f('Appending formatted delta to cells and joining cells with a separator are visible.',[
        u('The loop iterates the dictionaries of lls.','The visible loop iterates levels and writes cells;lls is empty at the clipped start.'),
        u('depth_sweep is a dictionary.','Its container type is not stated;the expression iterates rows having layer fields.'),
        u('It prints the sorted layers and their corresponding attention values.','Only sorting and a clipped header print are visible;no corresponding value-row print is in the excerpt.')])],
    [f('path.resolve(strict=True) is a concrete visible detail,without its claimed error cause.',[
        u('Resolving the path with strict=True causes the propagation twin mismatch error.','RunnerError for the mismatch precedes path.resolve;no such causal link is shown.',True)]),
     f('The error text explicitly includes the name interpolation and propagation twin mismatch.',[
        u('name identifies the runner that encountered the error.','The referent of name is not specified;it is also an evidence dictionary key.')]),
     f('The error text explicitly includes name and the propagation twin mismatch.',[
        u('name identifies the runner that encountered the error.','The excerpt does not define name as a runner identity.')])],
    [f('The header,probe.c,oracle.cpp,build.py and test.py filenames are visible.'),
     f('The literal oracle.cpp and other listed filenames are visible.',[
        u('strat01_q6k_q8k_avx2_oracle.cpp is a C source file.','The .cpp entry is C++ rather than the claimed C-source type.')]),
     f('The named header,C probe,C++ oracle and Python build/test files are visible.')],
    [f('contig reads the same volume in one unbroken run as the planted control.',[
        u('This means the instrument is measuring volume rather than granularity.','That diagnosis is conditional on missing the dense ceiling;no measured failure is reported.',True)],[
        a('Data is not believed unless present in the dense ceiling.','This may paraphrase the performance trust condition;present is unclear rather than a definite new residency claim.')]),
     f('A shuffled reading order is explicitly mentioned for naive gather.',[
        u('The shuffled read is used as the planted control.','The planted control is contig with no scatter,not the shuffled gather.',True),
        u('The instrument measures volume and its result is not believed.','That is a conditional failure diagnosis,not an observed unconditional result.',True)]),
     f('contig reads the same volume in a single unbroken run as a control.',[
        u('The trust condition is that contig returns the same volume.','Same volume is already held fixed;the condition is reaching the dense performance ceiling.',True)])],
]


DATA.extend([
    [f('The reported0.145/0.152 fatal doses occur explicitly in the excerpt.',[
        u('No dose caused death in the third experiment.','No third experiment or no-death result is described;the visible passage concerns doses that proved fatal.',True)],[
        a('The0.12 dose was not considered normal.','The text calls the subject abnormal;which could loosely refer to that exceptional case or incorrectly to the dose itself.')]),
     f('No.38 died at0.12 and is exceptional because the subject was not normal.'),
     f('No.38 died at0.12 and is exceptional because the subject was not normal.')],
    [f('People coming to windows at the sound of wheels is explicit.',[
        u('Amy follows a person rather than the hack.','The hack passes her and wheels are the local vehicle context;the response changes the followed entity to a person.',True),
        u('That followed person is coming home.','The returning person named by the source is Peggy Raymond,not the followed hack/person.',True),
        u('The return is from a meeting.','No meeting or previous activity is mentioned.')],[
        a('Taffy is also present.','Taffy is mentioned but the clause is clipped;present could mean named in the passage or physically present at the scene.')]),
     f('Amy is panting and red in the face while pursuing the hack.',[
        u('The hack is coming home.','Peggy Raymond is the one expected home;no homecoming is assigned to the hack.',True),
        u('The hack comes from work.','No work or preceding destination is stated.')],[
        a('The hack is described using who.','This may be personification or a mistaken person interpretation;the pronoun alone is not counted as a separate definite entity assertion.')]),
     f('Amy follows after being passed;this pursuit event is visible.',[
        u('Amy follows a person instead of the hack.','The passing hack and sound of wheels supply the vehicle context;the response substitutes a person.',True),
        u('The followed person is coming home.','Peggy Raymond,not the passing followed entity,is the stated homecoming.',True),
        u('The return is from a meeting.','No meeting is present in the excerpt.')])],
    [f('Literal entries Dollars and Cents,John C.Gabel and others are actually listed.',[
        u('The list is a collection of books.','The visible fragment lists names/pseudonyms;it supplies no book-collection classification.',True),
        u('There are10 books.','No count10 or book count is given.'),
        u('Every book has a different author.','No per-book authors or author-uniqueness relation is shown.'),
        u('Every book has a different title.','No per-book titles or title-uniqueness relation is shown.')]),
     f('Literal entries Dollars and Cents,Percy L.McDermott and others occur in the list.',[
        u('The list is a collection of books.','Names/pseudonyms are visible;no book collection is identified.',True),
        u('There are10 books.','No such count is given.'),
        u('Every book has a different author.','No per-book author or uniqueness relation is shown.'),
        u('Every book has a different title.','No per-book title or uniqueness relation is shown.')]),
     f('No visible name or other concrete supported list detail is supplied.',[
        u('The excerpt is a book collection.','It only supplies names/pseudonyms,without that classification.',True),
        u('It contains1000+books.','There is no quantity1000+or book count.'),
        u('Each book has a unique title.','No book-title uniqueness relation is stated.'),
        u('Each book has a unique author.','No book-author uniqueness relation is stated.'),
        u('The books are organized by publisher.','No publishers or publisher organization appears.'),
        u('The books are organized by publication date.','No publication dates or such organization appears.')],missing=True)],
    [f('Keeping whites out of Indian Country is the explicit frontier-maintenance condition.',[
        u('Oregon and Santa Fe were established in the described event.','The text places them beyond the frontier and discusses trails,not their establishment.')]),
     f('Oregon and Santa Fe are explicitly associated with the trail context.',[
        u('The United States changes its frontier-maintenance approach.','The passage describes the constraint becoming unworkable,not an enacted policy shift.'),
        u('Interest in the Rocky Mountains declines.','The passage states much interest beyond the Rockies;no decline is described.',True),
        u('The frontier effort controls expansion of Native American populations.','The stated effort is keeping whites out of Indian Country,not restraining Native American expansion.',True)]),
     f('Oregon/Santa Fe trails and the Indian frontier are visibly named.',[
        u('The United States changes its frontier-maintenance approach.','No actual change of policy is reported;the text explains a constraint.'),
        u('Interest in the Rocky Mountains declines.','Much interest beyond the Rockies is described,not declining interest.',True),
        u('The effort is to control the Indian population.','The explicit control target is whites entering Indian Country.',True)])],
    [f('A hillside house and turning in darkness are concrete visible scene details.',[
        u('The pursuer/searching man and man turning to shout are conflated.','The old fellow hears my footsteps:the listener who turns and the first-person follower are distinct.',True),
        u('The quoted shout is altered and completed with I see a house.','The source has von sneak Yankee and ends the speech at I s;the added completed house utterance is not visible.')],[
        a('A man determines where he himself is located.','The repeated he referents are unclear;the source has a follower seeking another man,so self-location is not asserted as a separate definite error.')]),
     f('A hillside house,heard footsteps and turning in darkness are visible.',[
        u('The old man hears footsteps of the man about to enter the house.','The old fellow is the one about to enter;he hears the narrator follower footsteps.',True),
        u('The literal shout is completed with I see a man.','The visible quoted speech is different and clipped at I s;this completion is not supplied.')]),
     f('The hillside house and old fellow turning in darkness are explicit.',[
        u('The old man hears the man he is looking for.','The narrator follows/seeks the old fellow;the old fellow hears the narrator,not a stated target of his own search.',True),
        u('The shout includes I see a house built on the hillside.','That house is narrative context,not the stated quoted speech;the source quotation is clipped.')])],
    [f('The Comic English Grammar and its8-shilling price are visible.',[
        u('The second title is Merrily England in T.','The visible title fragment is MERRIE ENGLAND IN T,not Merrily.'),
        u('Both illustrated books cost8shillings.','The second two-volume listing costs21s,not8s.',True)]),
     f('The Comic English Grammar and illustrated book listings are explicit.',[
        u('The second title is Merrill England in T.','The printed title fragment is MERRIE ENGLAND IN T,not Merrill.'),
        u('Both illustrated books cost8shillings.','The second two-volume entry is priced21s.',True)]),
     f('The Comic English Grammar and its8-shilling price are visible.',[
        u('The second title is Merrily England in T.','The source title fragment says MERRIE,not Merrily.'),
        u('Both books cost8shillings.','The second two-volume entry is21s,not8s.',True)])],
    [f('Christian kings sought suppression while old-religion adherents clung to the practice.'),
     f('Christian kings sought suppression while old-religion adherents clung to the practice.'),
     f('Christian kings sought suppression while old-religion adherents clung to the practice.')],
    [f('Nobility of some and strength of others are explicit observed qualities.',[
        u('The hue is vivid.','The source adjective is livid,not vivid.'),
        u('The disease itself has that uniform hue.','The clipped opening does not identify the hue referent as the disease.'),
        u('The disease is less common than other diseases.','It was not so with many men contrasts the appearance effect,not disease prevalence.',True),
        u('The disease is idealized by its appearance.','Ill-health idealizes the men facial features,not the disease itself.',True)]),
     f('Nobility and strength emerging from facial appearance are visible traits.',[
        u('The hue is vivid.','The excerpt says livid.'),
        u('The disease has the uniform hue.','The clipped antecedent is not identified as the disease.'),
        u('The disease is less common than others.','The comparison concerns its appearance effect in men,not disease prevalence.',True),
        u('The disease is idealized by its features.','The faces of men are idealized by ill-health;the response reverses that object.',True)]),
     f('The described man is explicitly short,gingery and active.',[
        u('The hue is vivid.','The visible adjective is livid.'),
        u('The disease has a uniform hue.','The hue referent is absent from the clipped opening.'),
        u('The disease is less common than others.','No prevalence comparison is made;not so contrasts the effect on appearance.',True),
        u('The disease is particularly noticeable in men.','No relative noticeability/prevalence claim about men versus other people is stated.')])],
])


DATA.extend([
    [f('Loss of88tokens and BPB more than doubling from1/3 to1/8 are explicit.'),
     f('The88token loss and specific r512/D1536 attention-rank comparison are explicit.'),
     f('The88token loss,BPB doubling and r512/D1536 specificity are visible.')],
    [f('KernelWorkerStatus.COMPLETE and selfcheck_pass=true are literal source fields.',[
        u('Kernel.__init__ is called with no delay and successfully initializes the kernel.','The clipped source says _ready cleared;neither __init__ nor a Kernel class constructor call is stated.',True),
        u('The kernel itself downloads and verifies its artifacts.','The report says downloaded/verified from JSON,without assigning that operation to the kernel.'),
        u('dtype_achieved is the device count.','dtype_achieved is torch.float16;device count is gpu.n_devices_achieved.',True)],[
        a('selfcheck_pass suggests all necessary checks passed.','All necessary checks could mean that selfcheck or every wider acceptance check;the scope is unspecified and phrased as a suggestion.')]),
     f('COMPLETE status and selfcheck_pass=true are explicitly reported.',[
        u('Kernel.__init__ is called without delay and initialization succeeds.','The visible fragment describes _ready clearing,not a constructor call.',True),
        u('The kernel downloads/verifies the JSON artifacts.','The reporting actor is unspecified;this is not assigned to the kernel itself.'),
        u('dtype_achieved is the device count.','That field is torch.float16,not the reported GPU count.',True)],[
        a('The selfcheck flag suggests all checks passed.','The tentative all-checks scope may mean the selfcheck or wider acceptance;only one flag is visible.')]),
     f('KernelWorkerStatus.COMPLETE and selfcheck_pass=true are visible.',[
        u('Kernel.__init__ is called with no delay and initialization succeeds.','The source describes _ready clearing;it does not identify such a constructor.',True),
        u('The kernel itself downloads/verifies artifacts.','The actor downloading/verifying the artifacts is not named.'),
        u('dtype_achieved is the device type.','torch.float16 identifies arithmetic dtype,not device hardware type.',True)],[
        a('The selfcheck suggests all necessary checks passed.','The scope of necessary checks is not specified beyond the literal flag.')])],
    [f('Qwen-7B green/yellow region and Qwen-14B upward column are visible.',[
        u('These patterns are in ls command output.','The initial ls. is clipped text;the source describes Qwen panels,not a shell-command listing.',True)]),
     f('The two named Qwen panels and their light regions/column occur in the excerpt.',[
        u('The column extends upward in the first block size.','The source says upward in n,not an ordinal first block size.',True)]),
     f('The named Qwen panels,green/yellow regions and column extending upward in n are visible.')],
    [f('The accepted artifact was not opened and a canonical model-free result exists.',[
        u('The diagnostic/projection/constructed-input/RMSNorm/up/downstream processes are not performed.','Only no donor execution and no accepted-artifact opening are stated;the clipped operator list gives no such collective execution status.')]),
     f('The APPARATUS_READY_NO_DONOR_EXECUTION status and canonical result link are visible.',[
        u('The linked APPARATUS_RESULT.md is a protocol file that the protocol probes.','It is labeled the apparatus result for a protocol,not a protocol file/probing target.',True)],[
        a('The protocol is not executed by a donor.','This may loosely mean no donor-model execution;the phrase reverses the apparent executor/object roles but its intended status is unclear.')]),
     f('The accepted artifact was not opened and a model-free result is available.',[
        u('Diagnostic/projection/constructed-input/RMSNorm/up/downstream are unavailable.','No operator availability statement occurs in the clipped list.')])],
    [f('tf>=48 passes at115 with delta87;second500 remove26.6percent of the surviving residual.'),
     f('The115/delta87 gate and26.6percent second-block residual removal are explicit.'),
     f('The instrument firing on the known-positive is a concrete stated observation.',[
        u('97.175percent of the residual surviving the first500 is removed.','97.175 is a prior damage-removed aggregate;the second500 remove26.6percent of the surviving residual.',True)])],
    [f('GO requires all sealed gates and hardware/protocol reproducibility;NO-GO requires localized mechanism and closest result.'),
     f('The literal GO/NO-GO acceptance requirements are faithfully reproduced as source data.'),
     f('The literal GO/NO-GO acceptance requirements and amendment heading are reproduced as source data.')],
    [f('Biased sigmoid/softmax substitution/omitted normalization are named negative-control mutations.',[
        u('The fifth mutation transposes or swaps selected weights.','The fifth item is clipped at transpose or s;its completed action/object is not visible.')],[
        a('It emphasizes the importance of using the wrong routing variants.','For model-free tests this can mean intentionally testing negative mutations;the response omits explicit rejection context without definitely recommending production use.')]),
     f('Omitting selected-weight normalization is an explicit listed mutation.',[
        u('The mutation replaces softmax with softmax.','The source replaces sigmoid plus top-k normalization with softmax,not an identity change.',True),
        u('The final mutation transposes or swaps selected expert IDs.','The clipped fifth item does not supply this completed action/object.')],[
        a('It emphasizes importance of biased sigmoid and omitted normalization.','The model-free-test context may denote negative controls;correct production use is not explicitly asserted.')]),
     f('The first four wrong-implementation mutations and their required failure are accurately reproduced.',[
        u('The fifth item swaps positions of selected IDs and weights.','The item is clipped at transpose or s and does not specify those objects.')],[
        a('These changes aim to make the model more robust and less error-prone.','This could mean implementation checks catch bugs or could imply changed learned robustness;the response states a suggested aim,not an achieved model result.')])],
    [f('The100percent-versus3percent mismatch and active_params*bytes_per_weight expression are visible.',[
        u('The100percent model was trained on the3percent architecture.','The source reports running the model,not training it.',True),
        u('The model density is constrained by the architecture density.','The author explicitly says the constraint was the model density,not the architecture.',True),
        u('The third listed lever is model density.','Only activation fraction and bits per weight are visible;the third lever is clipped.')],[
        a('Speed is a product with number of bits per weight.','It also reproduces the bytes_per_weight formula;bits are a stated precision lever,so literal units versus proportional dependency are unclear.')]),
     f('The model-density constraint and not attacking it are explicit.',[
        u('The100percent model was trained on the3percent architecture.','The source says ran,not trained.',True),
        u('Model density is the third enumerated multiplicative lever.','The visible enumeration is clipped after its first two levers.')]),
     f('The100percent/3percent mismatch and activation/precision influences are visible.',[
        u('The100percent model was trained on the3percent architecture.','The stated action is running the model,not training.',True)],[
        a('Model density was constrained by model density.','This tautology may mean density was the constraint,as in the source;no definite additional causal mechanism is counted.')])],
])


def main():
    assert hashlib.sha256(PANEL.read_bytes()).hexdigest()==PANEL_SHA
    panel=json.loads(PANEL.read_text(encoding='utf-8'))
    assert len(DATA)==len(panel['rows'])==24
    rows=[{'source_id':visible['source_id'],**dict(zip('ABC',findings))}
          for visible,findings in zip(panel['rows'],DATA)]
    assert all(len(findings)==3 for findings in DATA)
    out=DOC/'meth282_anonymous_verdict.json';assert not out.exists()
    out.write_text(json.dumps({'panel_sha256':PANEL_SHA,'arm_map_consulted':False,
        'labeled_generations_consulted':False,'rubric':'Unchanged123/267 excerpt-only atomic claims;severe central inversion;ANY concrete supported detail;ambiguity descriptive.',
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows},indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({'sources':len(rows),'responses':len(rows)*3,'verdict_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
