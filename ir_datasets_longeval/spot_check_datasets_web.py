#!/usr/bin/env python3
import gzip
import os
from pathlib import Path

from ir_datasets.util import ZipExtractCache, home_path
from tqdm import tqdm

QUERY_IDS = [
    "10892",  # recette gateau framboises
    "4772",  # loi de modernisation de la fonction publique
    "16711",  # voiture telecommandee
]

INPUT_DIR = home_path() / "longeval-web/release_2025_p1/release_2025_p1"

OUTPUT_DIR = Path("spot-check-web")


SUB_COLLECTIONS_TRAIN = [
    "2022-06",
    "2022-07",
    "2022-08",
    "2022-09",
    "2022-10",
    "2022-11",
    "2022-12",
    "2023-01",
    "2023-02",
]


def starts_with_query(query_id):
    for qid in QUERY_IDS:
        if query_id and query_id == qid:
            return True
    return False


sub_collection = SUB_COLLECTIONS_TRAIN[0]


docs = set()
qrels_path = (
    INPUT_DIR
    / f"French/LongEval Train Collection/qrels/{sub_collection}_fr/qrels_processed.txt"
)
qrels_out = OUTPUT_DIR / f"French/LongEval Train Collection/qrels/{sub_collection}_fr"
qrels_out.mkdir(parents=True, exist_ok=True)
with open(qrels_path, "r") as i, open(qrels_out / "qrels_processed.txt", "w") as o:
    for l in i:
        qid = l.split(" ")[0]
        if starts_with_query(qid):
            docs.add(l.split(" ")[2])
            o.write(l)


# for m in ["bm25", "default"]:
for m in ["BM25"]:
    with gzip.open(f"data/longeval-web-fr-{sub_collection}-{m}.gz", "rt") as f:
        for l in tqdm(f):
            qid = l.split(" ")[0]
            if starts_with_query(qid):
                docs.add(l.split()[2].strip("doc"))

queries_path = INPUT_DIR / "French/queries.txt"
with (
    open(queries_path, "r") as i,
    open(OUTPUT_DIR / "French/queries.txt", "w") as o,
):
    for l in i:
        if starts_with_query(l.split("\t")[0]):
            o.write(l)


docs_path = OUTPUT_DIR / f"French/LongEval Train Collection/Trec/{sub_collection}_fr"
docs_path.mkdir(parents=True, exist_ok=True)


docs_path_in = INPUT_DIR / f"French/LongEval Train Collection/Trec/{sub_collection}_fr"
doc_files = [i for i in os.listdir(docs_path_in) if "trec" in i]

found = set()

for doc_file in tqdm(doc_files):
    found_docs = []
    doc_lines = []
    add_doc = False

    with open(docs_path_in / doc_file, "r") as f:
        for l in f:
            if l.startswith("<DOC>"):
                doc_lines = [l]
                add_doc = False  # reset per doc
            elif l.startswith("<DOCNO>"):
                docid = l.split("<DOCNO>")[1].split("</DOCNO>")[0].strip("doc")
                if docid in docs and docid not in found:
                    add_doc = True
                    found.add(docid)
                    print(f"Adding {docid} from {doc_file}")
                doc_lines.append(l)
            elif l.startswith("</DOC>"):
                doc_lines.append(l)
                if add_doc:
                    found_docs.append(doc_lines)
            else:
                doc_lines.append(l)

    if found_docs:
        with open(docs_path / doc_file, "w") as outp:
            for doc in found_docs:
                for line in doc:
                    outp.write(line)