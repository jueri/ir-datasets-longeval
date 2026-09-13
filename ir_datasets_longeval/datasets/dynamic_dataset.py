from __future__ import annotations

import contextlib
import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path
from pkgutil import get_data
from typing import Dict, Iterator, NamedTuple, Optional

import ir_datasets
from ir_datasets import registry
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import (
    BaseDocs,
    BaseQrels,
    BaseQueries,
    TrecQrel,
    TrecQrels,
)
from ir_datasets.indices import PickleLz4FullStore
from ir_datasets.util import ZipExtractCache, home_path

from ir_datasets_longeval.formats import MetaDataset
from ir_datasets_longeval.util import DownloadConfig

logger = ir_datasets.log.easy()

NAME = "dynamic-dataset"
QREL_DEFS = {
    2: "highly relevant",
    1: "relevant",
    0: "not relevant",
}
class ExtractedPath:
    def __init__(self, path):
        self._path = path

    def path(self, force=True):
        if force and not self._path.exists():
            raise FileNotFoundError(self._path)
        return self._path

    @contextlib.contextmanager
    def stream(self):
        with gzip.open(self._path, "rb") as f:
            yield f
            
def iter_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
                
class DynamicDatasetDocument(NamedTuple):
    doc_id: str
    base_doc_id: str
    url: str
    title: str
    text: str
    scrape_status: str
    fetched_at: datetime
    publish_date: Optional[str]
    author: Optional[str]
    language: Optional[str]
    
    def default_text(self) -> str:
        return self.title + "\n" + self.text


class DynamicDatasetQuery(NamedTuple):
    automatic: bool
    generated_at: datetime
    llm_model: str
    prompt: str
    query: str
    query_id: str
    topic_id: str
    valid_from: datetime
    valid_to: datetime
    serp_file: Path
    topic: DynamicDatasetTopic
    
    def default_text(self):
        return self.query
class DynamicDatasetTopic(NamedTuple):
    automatic: bool
    debate_date: datetime
    debate_id: str
    description: str
    llm_model: str
    narrative: str
    prompt: str
    title: str
    topic_id: str
    valid_from: datetime
    valid_to: datetime
    generated_at: datetime | None
    
    def default_text(self):
        return self.title
    
class DynamicDatasetDocuments(BaseDocs):
    def __init__(self, docs_dlcs):
        self._docs_dlcs = docs_dlcs

    @ir_datasets.util.use_docstore
    def docs_iter(self) -> Iterator[DynamicDatasetDocument]:
        yield from self._docs_iter_raw()
    
    def docs_count(self) -> int:
        return sum(1 for _ in self.docs_iter())
    
    def docs_store(self):
        snapshot_date = self._docs_dlcs.path().name.removesuffix(".jsonl.gz")
        return PickleLz4FullStore(
            path=self._docs_dlcs.path().parent / f"day-{snapshot_date}.pklz4",
            init_iter_fn=self._docs_iter_raw,
            data_cls=self.docs_cls(),
            lookup_field="doc_id",
            index_fields=["doc_id"],
        )
        
    def docs_cls(self):
        return DynamicDatasetDocument
    
    def _docs_iter_raw(self) -> Iterator[DynamicDatasetDocument]:
        for doc in iter_jsonl_gz(self._docs_dlcs.path()):
            yield DynamicDatasetDocument(
                doc_id=doc["doc_id"]+f".{doc['fetched_at'][:10]}",
                base_doc_id=doc["doc_id"],
                url=doc["url"],
                title=doc["title"],
                text=doc["text"],
                scrape_status=doc["scrape_status"],
                fetched_at=datetime.fromisoformat(doc["fetched_at"]),
                publish_date=doc.get("publish_date"),
                author=doc.get("author"),
                language=doc.get("language"),
            )

    def docs_cls(self):
        return DynamicDatasetDocument
    
class DynamicDatasetQueries(BaseQueries):
    def __init__(self, topics_dlcs, queries_dlcs):
        self._queries_dlcs = queries_dlcs
        self._topics_dlcs = topics_dlcs
    
    def queries_iter(self):
        topics = {}
        for topic in iter_jsonl_gz(self._topics_dlcs.path()):
            topic_id = topic["topic_id"]
            generated_at = topic.get("generated_at", None)
            if generated_at is not None:
                generated_at = datetime.fromisoformat(generated_at)
            
            topics[topic_id] = DynamicDatasetTopic(
                automatic=topic.get("automatic", False),
                debate_date=datetime.fromisoformat(topic["debate_date"]),
                debate_id=topic["debate_id"],
                description=topic["description"],
                generated_at=generated_at,
                llm_model=topic["llm_model"],
                narrative=topic["narrative"],
                prompt=topic.get("prompt", ""),
                title=topic["title"],
                topic_id=topic_id,
                valid_from=datetime.fromisoformat(topic["valid_from"]),
                valid_to=datetime.fromisoformat(topic["valid_to"]),
            )
        
        for query in iter_jsonl_gz(self._queries_dlcs.path()):
            topic_id = query["topic_id"]
            query_id = query["query_id"]
            serp_file = self._queries_dlcs.path().parent / "serps" / f"{topic_id}"/f"{query_id}.json.gz"
        
            query = DynamicDatasetQuery(
                    automatic=query["automatic"],
                    generated_at=datetime.fromisoformat(query["generated_at"]),
                    llm_model=query["llm_model"],
                    prompt=query["prompt"],
                    query=query["query"],
                    query_id=query["query_id"],
                    topic_id=query["topic_id"],
                    valid_from=datetime.fromisoformat(query["valid_from"]),
                    valid_to=datetime.fromisoformat(query["valid_to"]),
                    serp_file=serp_file,
                    topic = topics[query["topic_id"]]
            )
            yield query


    def queries_cls(self):
        return DynamicDatasetQuery
    
class DynamicDataset(Dataset):
    def __init__(self, date_path: Path, qrels_set: str, prior_datasets: tuple[Dataset, ...] = ()):
        self.timestamp = datetime.strptime(date_path.name, "%Y-%m-%d")
        self.prior_datasets = prior_datasets

        # Documents
        docs_path = date_path / "documents.jsonl.gz"
        docs = DynamicDatasetDocuments(ExtractedPath(docs_path))
        
        # Topics
        topic_path = date_path / "topics.jsonl.gz"
        queries_path = date_path / "queries.jsonl.gz"
        topics = DynamicDatasetQueries(ExtractedPath(topic_path), ExtractedPath(queries_path))
        
        # Qrels
        qrels = None
        qrels_path = date_path / f"qrels.{qrels_set}.trec.gz"
        if qrels_path.exists() and qrels_path.is_file():
            qrels = TrecQrels(ExtractedPath(qrels_path), QREL_DEFS)
        self.qrels = qrels
        
        super().__init__(qrels, topics, docs)
    
    def get_timestamp(self):
        return self.timestamp

    def get_snapshot(self):
        # snapshots are the timestamps in this view
        return self.timestamp.strftime("%Y-%m-%d")

    def get_datasets(self):
        return None

    def get_prior_datasets(self):
        if not self.prior_datasets:
            return []
        return list(self.prior_datasets)

class DynamicDatasetSnapshotQueries(BaseQueries):
    def __init__(self, base_path: Path, snapshot: int, queries_emerged: dict[str, datetime]):
        self.base_path = base_path
        self.snapshot = snapshot
        self.queries_emerged = queries_emerged
        
    def queries_iter(self):
        for query_id, emerged_at in self.queries_emerged.items():
            snapshot_date = emerged_at + timedelta(days=self.snapshot)
            date_path = self.base_path / snapshot_date.strftime("%Y-%m-%d")
            topics_path = date_path / "topics.jsonl.gz"
            queries_path = date_path / "queries.jsonl.gz"

            if not topics_path.is_file() or not queries_path.is_file():
                continue

            daily_queries = DynamicDatasetQueries(
                ExtractedPath(topics_path),
                ExtractedPath(queries_path),
            )

            for daily_query in daily_queries.queries_iter():
                if query_id == daily_query.query_id:
                    yield daily_query
    
    def queries_cls(self):
        return DynamicDatasetQuery

class DynamicDatasetSnapshotQrels(BaseQrels):
    def __init__(self, base_path: Path, qrels_set: str, snapshot: int, topic_emerged: dict[str, datetime]):
        self.base_path = base_path
        self.qrels_set = qrels_set
        self.snapshot = snapshot
        self.topic_emerged = topic_emerged
        
    def qrels_iter(self):
        for topic_id, emerged_at in self.topic_emerged.items():
            snapshot_date = emerged_at + timedelta(days=self.snapshot)
            date_path = self.base_path / snapshot_date.strftime("%Y-%m-%d")
            qrels_path = date_path / f"qrels.{self.qrels_set.lower()}.trec.gz"

            if not qrels_path.is_file():
                continue

            qrels = TrecQrels(ExtractedPath(qrels_path), QREL_DEFS)

            for qrel in qrels.qrels_iter():
                if qrel.query_id == topic_id:
                    yield qrel
    
    def qrels_cls(self):
        return TrecQrel
    
            
class DynamicDatasetAllDocs(BaseDocs):
    def __init__(self, base_path: Path):
        self.base_path = base_path
        
    @ir_datasets.util.use_docstore
    def docs_iter(self):
        for date_path in sorted(self.base_path.glob("*/*")):
            if not date_path.name.endswith("documents.jsonl.gz"):
                continue
            daily_documents = DynamicDatasetDocuments(
                ExtractedPath(date_path)
            )
            for document in daily_documents.docs_iter():
                yield document
    
    def docs_store(self):
        return PickleLz4FullStore(
            path=self.base_path / "all-documents.pklz4",
            init_iter_fn=self.docs_iter,
            data_cls=DynamicDatasetDocument,
            lookup_field="doc_id",
            index_fields=["doc_id"],
        )            
    
    def docs_cls(self):
        return DynamicDatasetDocument
class DynamicSnapshotDataset(Dataset):
    def __init__(self, base_path: Path, qrels_set: str, snapshot: int, queries_emerged: dict[str, datetime], documents: DynamicDatasetAllDocs, prior_datasets: tuple[Dataset, ...] = ()):
        self.prior_datasets = prior_datasets
        self.snapshot = snapshot
        
        # Topics
        queries = DynamicDatasetSnapshotQueries(base_path, snapshot, queries_emerged)
        self.queries = queries
        
        # Qrels
        qrels = DynamicDatasetSnapshotQrels(base_path, qrels_set, snapshot, queries_emerged)
        self.qrels = qrels
        super().__init__(queries, qrels, documents)
        
    def get_timestamp(self):
        # snapshot contains multiple timestamps
        return None

    def get_snapshot(self):
        return self.snapshot

    def get_datasets(self):
        return None

    def get_prior_datasets(self):
        if not self.prior_datasets:
            return []
        return list(self.prior_datasets)

def register_dataset_by_date(base_path: Path, topic_type: str, qrels_type: str):
    datasets = []
    for date_path in sorted((base_path.absolute() / topic_type).iterdir(), key=lambda p: p.name):
        if not date_path.is_dir():
            continue
        dataset = DynamicDataset(date_path, qrels_type.lower(), tuple(reversed(datasets)))
        datasets.append(dataset)
        
        tag = f"{NAME}/{topic_type}/days/{date_path.name}/{qrels_type}"
        if tag not in registry:
            registry.register(tag, dataset)
    
    meta_dataset = MetaDataset(datasets)
    meta_tag = f"{NAME}/{topic_type}/days/{qrels_type}"
    if meta_tag not in registry:
        registry.register(meta_tag, meta_dataset)
        
    return meta_dataset


def get_emerging_dates(md_dynamic_in: MetaDataset):
    queries_emerged = {}
    for dataset in md_dynamic_in.get_datasets():
        for query in dataset.queries_iter():
            queries_emerged[query.query_id] = query.topic.valid_from
    return queries_emerged

def get_n_snapshots(queries_emerged: Dict[str, datetime], base_path: Path):
    first_date = min(queries_emerged.values())
    available_dates = sorted([d.name for d in (base_path.absolute()).iterdir() if d.is_dir()])
    available_dates = [datetime.strptime(d, "%Y-%m-%d") for d in available_dates]
    last_date = max(available_dates)
    n_snapshots = (last_date - first_date).days + 1
    return n_snapshots
    

def register_dataset_by_snapshot(base_path: Path, topic_type: str, qrels_type: str, queries_emerged: Dict[str, datetime], documents: DynamicDatasetAllDocs):
    n_snapshots = get_n_snapshots(queries_emerged, base_path)
    datasets = []
    for snapshot in range(n_snapshots):
        dataset = DynamicSnapshotDataset(base_path, qrels_type.lower(), snapshot, queries_emerged, documents, reversed(tuple(datasets)))
        datasets.append(dataset)
        tag = f"{NAME}/{topic_type}/snapshot/{snapshot}/{qrels_type}"
        if tag not in registry:
            registry.register(tag, dataset)
            
    meta_dataset = MetaDataset(datasets)
    meta_tag = f"{NAME}/{topic_type}/snapshot/{qrels_type}"
    if meta_tag not in registry:
        registry.register(meta_tag, meta_dataset)
    return meta_dataset
            
def register():
    base_path = home_path() / NAME
    dlc = DownloadConfig.context(NAME, base_path)
    base_path = ZipExtractCache(dlc[NAME], home_path() / NAME).path()
    
    # day
    md_dynamic_in = register_dataset_by_date(base_path, "dynamic", "IN")
    md_dynamic_re = register_dataset_by_date(base_path, "dynamic", "RE")
    md_static_in = register_dataset_by_date(base_path, "static", "IN")
    md_static_re = register_dataset_by_date(base_path, "static", "RE")
    
    # snapshot
    queries_emerged = get_emerging_dates(md_dynamic_in)
    dataset_path = base_path / "dynamic"
    documents_all = DynamicDatasetAllDocs(dataset_path)
    register_dataset_by_snapshot(dataset_path, "dynamic", "RE", queries_emerged, documents_all)
    register_dataset_by_snapshot(dataset_path, "dynamic", "IN", queries_emerged, documents_all)
    
    dataset_path = base_path / "static"
    documents_all = DynamicDatasetAllDocs(dataset_path)
    register_dataset_by_snapshot(dataset_path, "static", "RE", queries_emerged, documents_all)
    register_dataset_by_snapshot(dataset_path, "static", "IN", queries_emerged, documents_all)
    