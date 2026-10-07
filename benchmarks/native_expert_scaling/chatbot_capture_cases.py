"""Prewritten, self-authored original-donor calibration conversations.

These are calibration/development only, NEVER fresh final chatbot quality.
Two fit conversations and one reserved development conversation per category.
No observed IDs, donor answers or source activation data define this manifest.
"""

CASES = [
 ('arithmetic','Compute 17*23. Show one short calculation and the final integer.','A box contains 18 red and 27 blue balls. What fraction is red? Simplify it.','Calculate (84-19)/5 and explain the order of operations.'),
 ('algebra','Solve 3*x + 7 = 28. State each transformation.','If f(x)=2*x*x-3*x+1, compute f(4) and f(-1).','Solve 5*(x-2)=2*x+8 and check the result.'),
 ('geometry','A rectangle measures 7 cm by 12 cm. Give area and perimeter with units.','Explain why two nonparallel straight lines in a plane have at most one intersection.','A right triangle has legs 9 and 12. Find its hypotenuse.'),
 ('logic','All cedar trees in a garden are tall. Some tall plants are roses. Does it follow that some cedars are roses? Explain briefly.','There are three switches and only switch B is on. Write the truth values for A AND B, B OR C, and NOT B.','If rain implies a wet road, does a wet road imply rain? Give one counterexample.'),
 ('python','Write a Python function that counts each character in a string using a dictionary. Include an example.','Write a Python function to remove duplicates from a list while preserving first occurrence order.','Explain why appending to a list while iterating over that same list can be problematic. Give a safe alternative.'),
 ('code_debug','In Python, why does a mutable default argument retain changes between calls? Show a corrected function.','A function sorts its input list in place but the caller needs to preserve the original order. Explain the fix with a short example.','Explain the difference between == and is in Python, with an example where the distinction matters.'),
 ('data','Turn the following into a compact JSON object: name Ada, age 37, active true. Output JSON only.','Given rows (apple,3), (pear,2), (apple,4), aggregate quantities by fruit and output a JSON object.','Output a JSON array containing exactly three objects with keys id and enabled. Use ids 1,2,3 and enabled values true,false,true.'),
 ('instructions','Name four common kitchen utensils as a numbered list. Use exactly four lines.','Explain evaporation in exactly two short sentences suitable for a child.','Give three ways to organize a desk. Each bullet must contain at most six words.'),
 ('italian','Spiega in italiano la differenza tra area e perimetro usando un esempio semplice.','Scrivi una breve email in italiano per spostare una riunione da lunedi a mercoledi. Non inventare nomi o orari.','Riassumi in italiano: Un gruppo ha progettato un piccolo ponte. Prima ha verificato il terreno, poi ha scelto i materiali e infine ha controllato il progetto.'),
 ('translation','Translate into English: Il treno parte alle sette e arriva prima di mezzogiorno.','Translate into Italian: Please keep the window closed until the paint is dry.','Translate into English: Abbiamo salvato una copia del documento prima di modificarlo.'),
 ('explanation','Explain why dividing a task into smaller parts can make errors easier to locate.','Explain the difference between a measurement and a prediction using weather as an example.','Explain why keeping all units consistent matters when calculating a physical quantity.'),
 ('planning','Plan a one-hour study session with three activities and a short break. The durations must sum to one hour.','Create a checklist for preparing a small presentation: outline, evidence, slides and rehearsal.','Plan how to move files into a new folder while retaining a recoverable original copy.'),
 ('rewrite','Rewrite more concisely: Due to the fact that the room was very small in size, we made a decision to hold the meeting in another location.','Rewrite politely: Your report is late. Send it now.','Rewrite in plain language: The implementation of the proposed procedure necessitates the utilization of additional resources.'),
 ('uncertainty','A scale displays 12.4 kg with a resolution of 0.1 kg. Explain what the displayed precision does and does not tell us.','A survey of 20 people finds that 12 prefer tea. Explain why that does not establish the preference of an entire city.','A program was fast in one run and slow in another. List three plausible causes without claiming which one actually occurred.'),
 ('history','I put the blue notebook on the shelf and the green notebook in a drawer. Where is the green notebook? Answer in one sentence.','Our schedule is: draft on Tuesday, review on Thursday, publish on Friday. Which step immediately precedes publication?','Mira has a red cup. Leo has a yellow cup. They exchange cups. What color is Mira holding afterward?'),
 ('continuation','Continue this explanation: To compare two fractions with different denominators, we first','Continue this short story: The gardener found a small key beside the old gate. Instead of trying every door, she','Continue the following Python function body after the colon. Include only the indented body:\ndef sum_even(values):')
]


def manifest():
    rows=[]
    for category,first,second,development in CASES:
        for slot,text in enumerate((first,second,development)):
            mode='generate'
            messages=[dict(role='user',content=text)]
            if category=='continuation' and slot<2:
                prefix,continuation=text.split(': ',1)
                messages=[dict(role='user',content=prefix+'.'),dict(role='assistant',content=continuation)]
                mode='continue'
            rows.append(dict(id=f'{category}_{slot}',category=category,
                split='development' if slot==2 else 'fit',mode=mode,messages=messages))
    assert len(rows)==48 and len({r['id'] for r in rows})==48
    return dict(schema='QWEN_ORIGINAL_CHAT_CALIBRATION_V1',scope='self-authored calibration/development, NOT final fresh quality',
        max_new_tokens=96,max_prompt_tokens=512,EOS=[151645,151643],cases=rows)
