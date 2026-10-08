"""Deterministic balanced Falcon calibration prompts; no model calls."""
import argparse
import json
from pathlib import Path
import random
import sys

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
from chatbot_falcon_usability import write

DOMAINS = ('arithmetic', 'code', 'reading', 'rewrite', 'history', 'instruction', 'explanation', 'planning')
NAMES = ('Lena', 'Omar', 'Nia', 'Felix', 'Tess', 'Ravi', 'Cora', 'Ivan')
OBJECTS = ('lantern', 'camera', 'violin', 'backpack', 'compass', 'helmet', 'notebook', 'radio',
           'umbrella', 'guitar', 'telescope', 'suitcase', 'binoculars', 'keyboard', 'flashlight',
           'calculator', 'thermometer', 'microphone', 'tripod', 'pencil', 'bottle', 'clock',
           'charger', 'scanner', 'tablet', 'brush', 'ruler', 'basket')


def case(domain, i):
    surname = ('Arden', 'Blake', 'Crane', 'Dale')[i // 8]
    name, obj = NAMES[i % 8] + ' ' + surname, OBJECTS[(i * 3 + 1) % 28]
    other1, other2 = [NAMES[(i+j) % 8] + ' ' + surname for j in (1, 2)]
    n, m, variant = i + 7, 2 * i + 5, i % 4
    hour = i % 10 + 7
    history = []
    if domain == 'arithmetic':
        prompts = [f'{name} has {n} green counters and {m} yellow counters. {i+2} green counters are lost. How many counters remain? Answer with one integer.',
                   f'There are {n} boxes with 4 pencils each. {i+3} pencils are given away. How many pencils remain? Answer with one integer.',
                   f'A ticket costs {n} dollars. {name} buys 3 tickets and pays with {3*n+12} dollars. How many dollars of change are due? Answer with one integer.',
                   f'The four numbers are {n}, {n+2}, {n+4}, and {n+6}. What is their arithmetic mean? Answer with one integer.']
        prompt = prompts[variant]
    elif domain == 'code':
        prompts = [f'What is the exact Python value of sorted(set([{n}, {m}, {n}, {n-2}]))? Return only the list.',
                   f'What is the exact Python value of sum(x for x in [{n}, {m}, 4, 8] if x % 2 == 0)? Return only the integer.',
                   f'What is the exact Python value of {{"left": {n}, "right": {m}}}.get("center", 9)? Return only the integer.',
                   f'What is the exact Python value of list(range({n}, {n+7}))[1:6:2]? Return only the list.']
        prompt = prompts[variant]
    elif domain == 'reading':
        filler = ' '.join(f'The inventory sheet lists shelf {j} with a blue tag; this does not describe the requested item.' for j in range(1, 1 + (i % 3) * 3))
        facts = [f'{name} put the {obj} in drawer {n}. Later, Omar moved it to drawer {m}. Where is the {obj} now? Answer with the drawer number only.',
                 f'The {obj} belongs to {name}. It was lent to Ravi for a day. Who owns the {obj}? Answer with the name only.',
                 f'{name} arrived after {other1} but before {other2}. Which of these three arrived first? Answer with the name only.',
                 f'The store originally opens at {hour}:00. The notice says opening is delayed by two hours. At what hour does it open today? Return the hour number only.']
        prompt = f'Read the supplied notes and answer using only them. {filler} {facts[variant]}'
    elif domain == 'rewrite':
        requests = [f'Give me the {obj} right now.', f'You must check the {n} entries immediately.',
                    f'Send {name} the report before noon.', f'Stop leaving your {obj} on my desk.']
        prompt = f'Rewrite the following request politely in one sentence, preserving its meaning and details: {requests[variant]}'
    elif domain == 'history':
        history = [dict(role='user', content=f'My name is {name}. My first storage code was {n}. My travel item is a {obj}.'),
                   dict(role='assistant', content=f'Your name is {name}, your first storage code was {n}, and your travel item is a {obj}.'),
                   dict(role='user', content=f'Change my current storage code to {m}. Keep the original code as a historical fact.'),
                   dict(role='assistant', content=f'Your current storage code is {m}; the original code was {n}.')]
        prompt = ["What is my current storage code? Return the integer only.",
                  "What was my original storage code? Return the integer only.",
                  "What item did I say I travel with? Return the item only.",
                  "What name did I give you? Return the name only."][variant]
    elif domain == 'instruction':
        prompts = [f'Return exactly this word in uppercase, with no other text: {obj}',
                   f'Return only a JSON object with keys "name" and "count", using values "{name}" and {n}.',
                   f'Return these integers in ascending order, separated by semicolons and with no spaces: {m}, {n-1}, {n}.',
                   f'From the text "owner={name}; item={obj}; count={n}", return only the value of item.']
        prompt = prompts[variant]
    elif domain == 'explanation':
        facts = [('a plant bends toward a window', 'Light reaches the window side more strongly. The shaded side of the stem grows faster, causing a bend.'),
                 ('a wet cloth dries faster in moving air', 'Evaporation adds water vapor near the cloth. Moving air removes that vapor, allowing further evaporation.'),
                 ('a bicycle slows when its brakes are pressed', 'Brake pads rub the wheel. Friction converts part of the bicycle motion energy into heat.'),
                 ('a sealed empty plastic bottle shrinks when cooled', 'Cooling lowers the pressure of the gas inside. Outside air then exerts greater pressure on the bottle walls.')]
        event, information = facts[variant]
        prompt = f'{name} asks why {event}. Use these supplied facts: {information} Explain the cause in two short sentences and address {name} by name.'
    else:
        prompts = [f'{name} must charge a battery before testing a {obj}, and test it before packing it. List the three actions in order in one short sentence.',
                   f'{name} has {n} minutes. Task A needs {n-2} minutes and task B needs 4 minutes. Can both fit in that time without overlap? Answer yes or no, then briefly explain.',
                   f'The train leaves at {hour}:30 and the bus arrives at {hour}:45. Can {name} catch that train after taking that bus? Answer yes or no, then briefly explain.',
                   f'{name} needs to buy paper, print a document, and deliver it. Printing requires paper and delivery requires the printed document. Give the order as three numbered steps.']
        prompt = prompts[variant]
    return dict(domain=domain, prompt=prompt, **({'history': history} if history else {}))


def main(a):
    old = []
    for name in ('chatbot_hybrid_pilot_cases_v1.json', 'chatbot_falcon_usability_cases_v1.json'):
        old += json.loads((B / name).read_bytes())['cases']
    used = {json.dumps([c.get('history', []), c['prompt']], sort_keys=True) for c in old}
    active, reserved = [], []
    for i in range(28):
        split = 'FIT' if i < 16 else 'DEV' if i < 20 else 'RESERVED'
        stripe = [dict(id=f'{split.lower()}_{domain}_{i:02d}', split=split, **case(domain, i)) for domain in DOMAINS]
        random.Random(20261009 + i).shuffle(stripe)
        (active if i < 20 else reserved).extend(stripe)
    all_cases = active + reserved
    keys = [json.dumps([c.get('history', []), c['prompt']], sort_keys=True) for c in all_cases]
    assert len(set(keys)) == 224 and not (set(keys) & used)
    write(a.out, dict(schema='HYBRID_BALANCED_CALIBRATION_V1', max_new_tokens=96, context_limit=512,
                      cases=active, reserved_cases=reserved, domains=list(DOMAINS),
                      scope='128 FIT/32 DEV, 8 domains, deterministic authored template families. 64 RESERVED unqueried, not a complete final quality suite. No source/capture calls; limited-domain calibration, not broad preservation evidence.'))
    print(json.dumps(dict(cases=len(active), FIT=128, DEV=32, RESERVED=len(reserved), domains=8)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
