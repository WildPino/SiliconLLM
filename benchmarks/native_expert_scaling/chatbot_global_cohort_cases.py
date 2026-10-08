"""Fixed self-authored transfer prompts; no teacher answers or routing outcomes.

160 FIT +40 development; separate64 endpoint prompts are NEVER collected here.
The families are limited and partly templated, not a broad knowledge benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CATEGORIES=('arithmetic','code','instructions','factual_text','translation','multilingual','extraction','history')


def conversation(category,slot):
    n=11+3*slot;k=slot%5;mode='generate'
    if category=='arithmetic':
        text=(f'Compute {n}+{n+8}. Give the integer and a short explanation.',
            f'What is {n} times {k+3}? Show one calculation.',
            f'Solve 2*x + {n} = {3*n}. Explain the steps briefly.',
            f'A rectangle has sides {n} and {n+2} cm. Give its perimeter.',
            f'A jar has {n} red beads and {n+5} blue beads. How many beads are there?')[k]
    elif category=='code':
        text=(f'Write a Python function called count_{n} that counts positive numbers in a list.',
            f'In Python, explain why [{n}]*3 creates three list items. Show the result.',
            f'Write Python code to keep dictionary values greater than {n}.',
            f'Debug this Python comparison: value is {n}. We need numerical equality. Explain the fix.',
            f'Write a Python function that adds {n} to every number in a list.')[k]
    elif category=='instructions':
        text=(f'Give exactly three short suggestions for organizing {n} paper notes. Use bullets.',
            f'Write a two-sentence invitation to a study session lasting {n} minutes. Invent no location.',
            f'Produce a numbered checklist with four steps for reviewing document {n}.',
            f'Rewrite politely: Send me file {n} immediately. Use one sentence.',
            f'Summarize in one sentence: Team {n} drafted a plan, checked it, and saved a backup.')[k]
    elif category=='factual_text':
        text=(f'Read: Box {n} is green. Box {n+1} is orange. Which box is orange?',
            f'Read: In workshop {n}, the inspection happened after painting. What happened first?',
            f'Read: Lina placed {n} books on a shelf and {n+2} in a drawer. How many are in the drawer?',
            f'Read: Route {n} visits a library before a park, then a cafe. What immediately follows the library?',
            f'Read: Ticket {n} belongs to Omar. Ticket {n+1} belongs to Eva. Who owns ticket {n}?')[k]
    elif category=='translation':
        text=(f'Translate into English: Ho messo {n} matite nella scatola blu.',
            f'Translate into Italian: The meeting lasts {n} minutes and starts after lunch.',
            f'Translate into English: La caja contiene {n} monedas y una llave.',
            f'Translate into English: Nous avons prepare {n} copies du document.',
            f'Translate into English: A sala tem {n} cadeiras e duas janelas.')[k]
    elif category=='multilingual':
        text=(f'Rispondi in italiano: quanto fa {n} piu {n+4}? Spiega brevemente.',
            f'Responde en espanol: una caja tiene {n} lapices y recibe cuatro mas. Cuantos tiene ahora?',
            f'Reponds en francais: donne deux conseils pour organiser {n} feuilles.',
            f'Antworte auf Deutsch: Eine Kiste enthalt {n} Bucher. Drei kommen hinzu. Wie viele sind es?',
            f'Responda em portugues: explique em uma frase como guardar {n} notas.')[k]
    elif category=='extraction':
        text=(f'Output JSON only: item id {n}, name pencil, enabled true.',
            f'Extract as JSON: Order {n} contains two pens and three notebooks.',
            f'Output a JSON array of these integers in increasing order: {n+2}, {n}, {n+1}.',
            f'Convert to JSON only: label batch{n}; count {n+3}; complete false.',
            f'Read code R{n}, status ready. Output an object with keys code and status.')[k]
    else:
        assert category=='history'
        if k==4:
            mode='continue';return mode,[dict(role='user',content=f'Explain a careful way to sort {n} notes.'),
                dict(role='assistant',content='First, choose a clear sorting rule. Then')]
        histories=(
            (f'Remember: locker {n} contains a blue coat.', 'I will use that information.', f'What color is the coat in locker {n}?'),
            (f'Our draft is version {n}. The review is tomorrow.', 'The draft version and review date are clear.', 'Which version will we review?'),
            (f'I have {n} stamps. I give three to a friend.', 'You have given away three stamps.', 'How many stamps do I have left?'),
            (f'In plan {n}, testing follows editing.', 'Editing comes before testing.', 'Which activity should start first?'))
        first,reply,last=histories[k]
        return mode,[dict(role='user',content=first),dict(role='assistant',content=reply),dict(role='user',content=last)]
    return mode,[dict(role='user',content=text)]


def endpoint_conversation(category,slot):
    """Different concrete task texts; same limited domains, not domain holdout."""
    a=101+7*slot
    if category=='arithmetic':text=f'A shop sold {a} cards in the morning and {a-9} later. Find the total, then subtract 13 returns.'
    elif category=='code':text=f'Write a Python function that returns the indices of all list values equal to {a}. Include a tiny example.'
    elif category=='instructions':text=f'Give exactly two numbered sentences describing how to prepare folder V{a} for review.'
    elif category=='factual_text':text=f'Story: Neri carries badge {a}; Sol carries badge {a+1}. They swap badges. Which badge does Neri carry now?'
    elif category=='translation':text=f'Translate to English: Prima di iniziare, abbiamo controllato tutte le {a} etichette.'
    elif category=='multilingual':text=f'Spiega in italiano: {a} fogli vengono divisi in due gruppi. Il primo ne ha {a-17}. Quanti ne ha il secondo?'
    elif category=='extraction':text=f'Output JSON only with id and owner: Parcel X{a} is owned by Dana.'
    else:
        assert category=='history'
        return 'generate',[dict(role='system',content='Answer using only the facts in this conversation.'),
            dict(role='user',content=f'In sequence Z{a}, archiving comes after checking and before delivery.'),
            dict(role='assistant',content='I understand the sequence.'),dict(role='user',content='What directly precedes delivery?')]
    return 'generate',[dict(role='user',content=text)]


def manifest(endpoint=False):
    rows=[]
    for category in CATEGORIES:
        for slot in range(8 if endpoint else 25):
            mode,messages=endpoint_conversation(category,slot) if endpoint else conversation(category,slot)
            rows.append(dict(id=f'global_v1_{"endpoint" if endpoint else "transfer"}_{category}_{slot:02d}',
                category=category,split='endpoint' if endpoint else ('fit' if slot<20 else 'development'),mode=mode,messages=messages))
    return dict(schema='QWEN_GLOBAL_COHORT_CASES_V1',scope='Self-authored finite partly templated domains; endpoint unqueried' if endpoint else 'Transfer FIT/development, never fresh endpoint quality',
        max_prompt_tokens=256 if endpoint else 128,max_context_tokens=512 if endpoint else 256,
        max_new_tokens=64 if endpoint else 16,EOS=[151645,151643],cases=rows)


def main(args):
    from chatbot_capture_cases import manifest as original_manifest
    transfer=manifest();endpoint=manifest(True);old=original_manifest()
    def key(case):return json.dumps([case['mode'],case['messages']],ensure_ascii=False,sort_keys=True,separators=(',',':'))
    sets=[{key(c) for c in m['cases']} for m in (transfer,endpoint,old)]
    assert len(transfer['cases'])==len(sets[0])==200 and len(endpoint['cases'])==len(sets[1])==64
    assert not any(a&b for i,a in enumerate(sets) for b in sets[i+1:])
    assert sum(c['split']=='fit' for c in transfer['cases'])==160
    assert sum(c['split']=='development' for c in transfer['cases'])==40
    for path,value in ((args.transfer,transfer),(args.endpoint,endpoint)):
        payload=(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf8')
        with path.open('xb') as stream:stream.write(payload)
        print(json.dumps(dict(path=str(path),cases=len(value['cases']),sha256=hashlib.sha256(payload).hexdigest())))


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
    p=argparse.ArgumentParser();p.add_argument('--transfer',type=Path,required=True);p.add_argument('--endpoint',type=Path,required=True);main(p.parse_args())
