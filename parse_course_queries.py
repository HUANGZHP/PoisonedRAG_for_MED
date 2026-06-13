"""
Parse 病程单选题-DeepSeek.docx and generate:
1. adv_json file (usable as --adv_json_path)
2. ids file (usable as --target_ids_path)
3. queries file (question text lookup)
"""

import json
import re
from docx import Document

doc = Document('病程单选题-DeepSeek.docx')
paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

# We'll parse line by line
questions = []
current_q = None
question_counter = 0
case_counter = 0
options_pattern = re.compile(r'^([A-D])[.．、]\s*(.*)')
answer_pattern = re.compile(r'→\s*\*\*正确答案[：:]\s*([A-D])\*\*')
note_pattern = re.compile(r'^【.*】$')

def save_current():
    global current_q, question_counter
    if current_q and 'options' in current_q and len(current_q['options']) == 4:
        # Determine correct option
        corr_opt = current_q['correct_option']
        # Pick first wrong option as target
        wrong_opt = None
        target_label = None
        for opt_key in sorted(current_q['options'].keys()):
            if opt_key != corr_opt:
                target_label = opt_key
                wrong_opt = current_q['options'][opt_key]
                break
        current_q['correct answer'] = current_q['options'][corr_opt]
        current_q['incorrect answer'] = wrong_opt
        current_q['target_label'] = target_label
        current_q['adv_texts'] = []  # no adv texts needed
        
        qid = f"course:{question_counter:04d}"
        current_q['id'] = qid
        
        questions.append(current_q)
        question_counter += 1

for i, line in enumerate(paragraphs):
    # Case header
    if line.startswith('## 病例'):
        save_current()
        current_q = None
        case_counter += 1
        continue
    
    # Separator
    if line.startswith('---'):
        save_current()
        current_q = None
        continue
    
    # Question line (numbered)
    q_match = re.match(r'\*\*(\d+)\.\s+(.*?)\*\*', line)
    if q_match:
        save_current()
        qnum, qtext = q_match.group(1), q_match.group(2)
        current_q = {
            'question_num': int(qnum),
            'question': qtext,
            'options': {},
        }
        continue
    
    # Option line
    if current_q is not None:
        opt_match = options_pattern.match(line)
        if opt_match:
            opt_key, opt_text = opt_match.group(1), opt_match.group(2)
            current_q['options'][opt_key] = opt_text.strip()
            continue
    
    # Answer line
    ans_match = answer_pattern.match(line)
    if ans_match and current_q is not None:
        current_q['correct_option'] = ans_match.group(1)
        continue

# Save last question
save_current()

print(f"Parsed {len(questions)} questions from {case_counter} cases")

# Build the dict format expected by the project
adv_dict = {}
for q in questions:
    qid = q['id']
    adv_dict[qid] = {
        'id': qid,
        'question': q['question'],
        'options': q['options'],
        'correct answer': q['correct answer'],
        'correct_option': q['correct_option'],
        'incorrect answer': q['incorrect answer'],
        'target_label': q['target_label'],
        'adv_texts': q['adv_texts'],
    }

# Write adv_json
output_path = 'results/adv_targeted_results/course_colorectal_75.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(adv_dict, f, ensure_ascii=False, indent=2)
print(f"Written: {output_path}")

# Write ids file
ids_path = 'results/adv_targeted_results/course_colorectal_75.ids'
with open(ids_path, 'w', encoding='utf-8') as f:
    for qid in adv_dict:
        f.write(qid + '\n')
print(f"Written: {ids_path}")

# Write queries file
queries_path = 'results/adv_targeted_results/course_colorectal_75.queries.json'
queries_dict = {qid: adv_dict[qid]['question'] for qid in adv_dict}
with open(queries_path, 'w', encoding='utf-8') as f:
    json.dump(queries_dict, f, ensure_ascii=False, indent=2)
print(f"Written: {queries_path}")

# Also print a summary
for i, q in enumerate(questions):
    opt_str = ', '.join([f"{k}. {v[:30]}..." for k, v in q['options'].items()])
    print(f"{q['id']}: {q['question'][:60]}... | Correct: {q['correct_option']} | Target: {q['target_label']}")
