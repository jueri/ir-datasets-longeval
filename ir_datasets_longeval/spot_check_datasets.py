#!/usr/bin/env python3
from pathlib import Path
import os
import json
from tqdm import tqdm

QUERY_IDS = [
    '2f85a909-c4bf-49b6-b7a6-819f7d45f44c',
    'e6195d15-a4b6-4ce7-9b4f-cf39acdd37e3',
    '1901ce75-b35a-4dde-b331-e5e5366dd188',
]

INPUT_DIR = Path('/home/maik/.ir_datasets/longeval-sci/longeval_sci_training_2025')

OUTPUT_DIR = Path('spot-check')
DOCS_DIR = OUTPUT_DIR / 'documents'
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def starts_with_query(line):
    for qid in QUERY_IDS:
        if line and line.startswith(qid):
            return True
    return False

docs = set()

with open(INPUT_DIR / 'qrels.txt', 'r') as i, open(OUTPUT_DIR / 'qrels.txt', 'w') as o:
    for l in i:
        if starts_with_query(l):
            docs.add(l.split()[2])
            o.write(l)

for m in ['bm25', 'default']:
    with open(f'data/run-chatnoir-{m}.txt', 'r') as f:
        for l in f:
            docs.add(l.split()[2])

with open(INPUT_DIR / 'queries.txt', 'r') as i, open(OUTPUT_DIR / 'queries.txt', 'w') as o:
    for l in i:
        if starts_with_query(l):
            o.write(l)

docs_path = INPUT_DIR / "documents"
jsonl_doc_files = [i for i in os.listdir(docs_path) if 'json' in i]

for jsonl_doc_file in tqdm(jsonl_doc_files):
    lines = []
    with open(docs_path / jsonl_doc_file, 'r') as f:
        for l in f:
            l_parsed = json.loads(l)
            if str(l_parsed["id"]) in docs:
                lines.append(l)

    if len(lines) > 0:
        with open(DOCS_DIR / jsonl_doc_file, 'w') as outp:
            for l in lines:
                outp.write(l)


