"""Dynamic emerging-topics dataset models and local package reader support."""

import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import ir_datasets
from ir_datasets import registry
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import BaseDocs, BaseQrels, BaseQueries, TrecQueries
from ir_datasets.indices import PickleLz4FullStore
from ir_datasets.util import DownloadConfig as IrdDownloadConfig
from ir_datasets.util import ZipExtractCache, home_path

from ir_datasets_longeval.formats import MetaDataset
from ir_datasets_longeval.util import DownloadConfig

QREL_DEFS = {2: "highly relevant", 1: "relevant", 0: "not relevant"}
NAME = "dynamic-dataset"
SNAPSHOT_DAYS = range(21)
STATIC_TOPIC_IDS = {
    "321",
    "336",
    "341",
    "397",
    "408",
    "422",
    "427",
    "433",
    "801",
    "804",
    "806",
    "808",
    "809",
    "814",
    "818",
    "823"}
class DynamicTopic(NamedTuple):
    """A topic definition with its collection validity interval."""

    topic_id: str
    title: str
    description: str
    narrative: str
    topic_type: Optional[str]
    valid_from: str
    valid_to: Optional[str]
    llm: Optional[str]
    prompt: Optional[str]
    debate: Optional[str]

    def default_text(self) -> str:
        return " ".join(part for part in (self.title, self.description, self.narrative) if part)


class DynamicDocument(NamedTuple):
    """One daily document observation in the dynamic collection.

    ``doc_id`` is the date-bearing docno used by ir_datasets and qrels.  The
    URL-derived identity is exposed independently as ``stable_doc_id``.
    Archive assets remain references: normal retrieval needs only title/text.
    """

    doc_id: str
    stable_doc_id: str
    url: str
    title: Optional[str]
    text: str
    publish_date: Optional[str]
    language: Optional[str]
    content_version_id: str
    valid_from: str
    valid_to: str
    warc: Optional[str]
    warc_trec_id: Optional[str]
    screenshot: Optional[str]
    history: Optional["DynamicDocumentHistory"] = None

    @property
    def docno(self) -> str:
        return self.doc_id

    def default_text(self) -> str:
        return " ".join(part for part in (self.title, self.text) if part)

    def previous_versions(self) -> list["DynamicDocument"]:
        """Return older content versions of this URL, newest first."""
        if self.history is None:
            return []
        return self.history.previous_versions(self)


class DynamicSerpResult(NamedTuple):
    """A document reference at one position in a dated SERP."""

    rank: Optional[int]
    source: str
    url: str
    stable_doc_id: str
    capture_docno: str
    docno: Optional[str]
    title: Optional[str]
    snippet: Optional[str]


class DynamicQuery(NamedTuple):
    """A dated retrieval query, its topic, and its observed SERP."""

    query_id: str
    topic_id: str
    text: str
    date: str
    fetched_at: Optional[str]
    serp: tuple[DynamicSerpResult, ...]
    topic: Optional[DynamicTopic] = None

    def default_text(self) -> str:
        return self.text


class DynamicQrel(NamedTuple):
    """The standard ir_datasets qrel shape, joined through a date-bearing docno."""

    query_id: str
    doc_id: str
    relevance: int
    relevance_rater_1: Optional[int]
    relevance_rater_2: Optional[int]
    relevance_rater_3: Optional[int]

class DynamicQueries(BaseQueries):
    def __init__(self, path: Path, topics: dict[str, DynamicTopic]):
        self.path = path
        self.topics = topics

    def queries_iter(self) -> Iterator[DynamicQuery]:
        for record in _jsonl_gz(self.path):
            results = tuple(
                DynamicSerpResult(
                    item.get("rank"), item["source"], item["url"], item["doc_id"],
                    item["capture_docno"], item.get("docno"), item.get("title"), item.get("snippet"),
                )
                for item in record.get("results", [])
            )
            yield DynamicQuery(
                record["query_id"], record["topic_id"], record["text"], record["date"],
                record.get("fetched_at"), results, self.topics.get(record["topic_id"]),
            )

    def queries_cls(self):
        return DynamicQuery


class DynamicQrels(BaseQrels):
    def __init__(self, paths: list[Path], query_ids: set[str], document_ids: set[str]):
        self.paths = paths
        self.query_ids = query_ids
        self.document_ids = document_ids

    def qrels_iter(self) -> Iterator[DynamicQrel]:
        for path in self.paths:
            for record in _jsonl_gz(path):
                if record["query_id"] not in self.query_ids or record["docno"] not in self.document_ids:
                    continue
                yield DynamicQrel(
                    record["query_id"], record["docno"], int(record["relevance"]),
                    _optional_int(record.get("rater_1")), _optional_int(record.get("rater_2")),
                    _optional_int(record.get("rater_3")),
                )

    def qrels_defs(self):
        return QREL_DEFS

    def qrels_path(self):
        return self.paths[0]

    def qrels_cls(self):
        return DynamicQrel


class DynamicDocumentHistory:
    """Lazy resolver for the text-free document version index."""

    def __init__(self, documents_path: Path):
        self.documents_path = documents_path
        self.index_path = documents_path / "index.jsonl.gz"
        self._versions: Optional[dict[str, list[dict]]] = None
        self._daily_records: dict[str, dict[str, dict]] = {}

    def _load_index(self) -> dict[str, list[dict]]:
        if self._versions is None:
            self._versions = {}
            if self.index_path.is_file():
                for record in _jsonl_gz(self.index_path):
                    self._versions.setdefault(record["doc_id"], []).append(record)
                for versions in self._versions.values():
                    versions.sort(key=_version_start)
        return self._versions

    def _load_daily_records(self, snapshot_date: str) -> dict[str, dict]:
        if snapshot_date not in self._daily_records:
            path = self.documents_path / f"{snapshot_date}.jsonl.gz"
            self._daily_records[snapshot_date] = (
                {record["docno"]: record for record in _jsonl_gz(path)}
                if path.is_file()
                else {}
            )
        return self._daily_records[snapshot_date]

    def _document_for_version(self, version: dict) -> Optional[DynamicDocument]:
        snapshot_date = _version_start(version)
        record = self._load_daily_records(snapshot_date).get(version["docno"])
        if record is None:
            return None
        interval = version["valid_intervals"][0]
        return _document_from_record(
            record,
            self,
            valid_from=interval["valid_from"],
            valid_to=interval["valid_to"],
        )

    def previous_versions(self, document: DynamicDocument) -> list[DynamicDocument]:
        versions = self._load_index().get(document.stable_doc_id, [])
        current = next(
            (version for version in versions if version["docno"] == document.content_version_id),
            None,
        )
        current_start = _version_start(current) if current else document.valid_from
        previous = []
        for version in reversed(versions):
            if _version_start(version) >= current_start:
                continue
            resolved = self._document_for_version(version)
            if resolved is not None:
                previous.append(resolved)
        return previous


class DynamicDocs(BaseDocs):
    def __init__(
        self,
        path: Path,
        document_ids: Optional[set[str]] = None,
        history: Optional[DynamicDocumentHistory] = None,
    ):
        self.path = path
        self.document_ids = document_ids
        self.history = history or DynamicDocumentHistory(path.parent)

    @ir_datasets.util.use_docstore
    def docs_iter(self) -> Iterator[DynamicDocument]:
        yield from self._docs_iter_raw()

    def _docs_iter_raw(self) -> Iterator[DynamicDocument]:
        for record in _jsonl_gz(self.path):
            if self.document_ids is not None and record["docno"] not in self.document_ids:
                continue
            yield _document_from_record(record, self.history)

    def docs_count(self) -> int:
        return sum(1 for _ in self.docs_iter())

    def docs_cls(self):
        return DynamicDocument

    def docs_store(self):
        snapshot_date = self.path.name.removesuffix(".jsonl.gz")
        return PickleLz4FullStore(
            path=self.path.parent / "docstores" / f"day-{snapshot_date}.pklz4",
            init_iter_fn=self._docs_iter_raw,
            data_cls=self.docs_cls(),
            lookup_field="doc_id",
            index_fields=["doc_id"],
        )


class DynamicSnapshotQueries(BaseQueries):
    """Query/SERP records for topic-relative snapshot days."""

    def __init__(self, topic_dates: dict[str, str], base_path: Path, topics: dict[str, DynamicTopic]):
        self.topic_dates = topic_dates
        self.base_path = base_path
        self.topics = topics

    def queries_iter(self) -> Iterator[DynamicQuery]:
        for topic_id, snapshot_date in sorted(self.topic_dates.items()):
            path = self.base_path / "queries" / f"{snapshot_date}.jsonl.gz"
            for record in _jsonl_gz(path):
                if record["topic_id"] != topic_id:
                    continue
                results = tuple(
                    DynamicSerpResult(
                        item.get("rank"), item["source"], item["url"], item["doc_id"],
                        item["capture_docno"], item.get("docno"), item.get("title"), item.get("snippet"),
                    )
                    for item in record.get("results", [])
                )
                yield DynamicQuery(
                    record["query_id"], topic_id, record["text"], record["date"],
                    record.get("fetched_at"), results, self.topics[topic_id],
                )

    def queries_cls(self):
        return DynamicQuery


class DynamicSnapshotDocs(BaseDocs):
    """Documents referenced by a topic-relative snapshot's dated SERPs."""

    def __init__(
        self,
        paths: list[Path],
        document_ids: set[str],
        snapshot_day: int,
        documents_path: Optional[Path] = None,
    ):
        self.paths = paths
        self.document_ids = document_ids
        self.snapshot_day = snapshot_day
        self.documents_path = documents_path or (paths[0].parent if paths else None)
        self.history = DynamicDocumentHistory(paths[0].parent) if paths else None

    @ir_datasets.util.use_docstore
    def docs_iter(self) -> Iterator[DynamicDocument]:
        yield from self._docs_iter_raw()

    def _docs_iter_raw(self) -> Iterator[DynamicDocument]:
        seen: set[str] = set()
        for path in self.paths:
            for document in DynamicDocs(path, self.document_ids, self.history)._docs_iter_raw():
                if document.doc_id not in seen:
                    seen.add(document.doc_id)
                    yield document

    def docs_count(self) -> int:
        return sum(1 for _ in self.docs_iter())

    def docs_cls(self):
        return DynamicDocument

    def docs_store(self):
        if self.documents_path is None:
            raise ValueError("documents_path is required for an empty snapshot")
        return PickleLz4FullStore(
            path=self.documents_path / "docstores" / f"snapshot-{self.snapshot_day}.pklz4",
            init_iter_fn=self._docs_iter_raw,
            data_cls=self.docs_cls(),
            lookup_field="doc_id",
            index_fields=["doc_id"],
        )


def _jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _optional_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _version_start(version: dict) -> str:
    return version["valid_intervals"][0]["valid_from"]


def _document_from_record(
    record: dict,
    history: Optional[DynamicDocumentHistory],
    *,
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None,
) -> DynamicDocument:
    return DynamicDocument(
        record["docno"], record["doc_id"], record["url"], record.get("title"), record["text"],
        record.get("publish_date"), record.get("language"), record["content_version_id"],
        valid_from or record["valid_from"], valid_to or record["valid_to"],
        record.get("warc"), record.get("warc_trec_id"), record.get("screenshot"), history,
    )


def _topic_from_record(record: dict, topic_type: str) -> DynamicTopic:
    return DynamicTopic(
        record["topic_id"], record.get("title", ""), record.get("description", ""),
        record.get("narrative", ""), topic_type, record["valid_from"], record.get("valid_to"),
        record.get("llm_model") or record.get("llm"), record.get("prompt"), record.get("debate_id") or record.get("debate"), 
    )
    
def get_static_topics(base_path: Path) -> dict[str, DynamicTopic]:
    static_topics = {}
    wapo_dlc = IrdDownloadConfig.context("wapo", base_path)
    queries = TrecQueries(wapo_dlc['trec-core-2018/queries'])
    
    for wapo_topic in queries.queries_iter():
        topic_id = wapo_topic.query_id[:3]
        if topic_id in STATIC_TOPIC_IDS:
            static_topics[topic_id] = _topic_from_record({            
            "title": wapo_topic.title,
            "description": wapo_topic.description,
            "narrative": wapo_topic.narrative,
            "topic_id": topic_id,
            "debate_id": "wapo/v2/trec-core-2018",
            "debate_date": "2026-05-19",
            "generated_at": None,
            "llm_profile": "TREC",
            "llm_model": "human",
            "valid_from": "2026-09-01",
            "valid_to": "2026-09-15",
            }, topic_type="static")
    return static_topics
class DynamicDataset(Dataset):
    """A conventional ir_datasets dataset view over one package day."""

    def __init__(self, base_path: Path, at: str, qrel_set: Optional[str] = None):
        self.base_path = Path(base_path)
        self.at = at
        self.qrel_set = qrel_set.lower() if qrel_set else None
        if not self.base_path.is_dir():
            raise FileNotFoundError(self.base_path)
        query_path = self.base_path / "queries" / f"{at}.jsonl.gz"
        document_path = self.base_path / "documents" / f"{at}.jsonl.gz"
        if not query_path.is_file() or not document_path.is_file():
            raise ValueError(f"No complete dataset view for {at}")

        self._topics = {}
        
        # Load dynamic topics
        raw_records = _jsonl_gz(self.base_path / "topics.jsonl.gz")
        for row in raw_records:
            topic = _topic_from_record(row, topic_type="dynamic")
            is_active = topic.valid_from <= at and (topic.valid_to is None or at < topic.valid_to)
            
            if is_active:
                self._topics[topic.topic_id] = topic
        
        # Add static topics from TREC Core 2018
        static_topics = get_static_topics(self.base_path)
        self._topics.update(static_topics)
                
        docs = DynamicDocs(document_path)
        queries = DynamicQueries(query_path, self._topics)
        qrels = None
        if self.qrel_set:
            if self.qrel_set == "in":
                qrels_paths = sorted((self.base_path / "qrels").glob("in*.jsonl.gz"))
            else:
                qrels_paths = [self.base_path / "qrels" / f"{self.qrel_set}.jsonl.gz"]
            if not qrels_paths or not all(path.is_file() for path in qrels_paths):
                raise ValueError(f"Unknown qrel set: {qrel_set}")
            qrels = DynamicQrels(
                qrels_paths,
                {query.query_id for query in queries.queries_iter()},
                {document.doc_id for document in docs.docs_iter()},
            )
        super().__init__(docs, queries, qrels)

    def topics_iter(self) -> Iterator[DynamicTopic]:
        yield from self._topics.values()

    def get_timestamp(self):
        return datetime.strptime(self.at, "%Y-%m-%d")

    def get_snapshot(self):
        return None

    def get_datasets(self):
        return None

    def get_prior_datasets(self):
        query_dates = {
            path.name.removesuffix(".jsonl.gz")
            for path in (self.base_path / "queries").glob("*.jsonl.gz")
        }
        document_dates = {
            path.name.removesuffix(".jsonl.gz")
            for path in (self.base_path / "documents").glob("*.jsonl.gz")
        }
        return [
            DynamicDataset(self.base_path, previous_date, self.qrel_set)
            for previous_date in sorted(query_dates & document_dates, reverse=True)
            if previous_date < self.at
        ]


class DynamicSnapshotDataset(Dataset):
    """A topic-relative, rather than calendar-relative, collection view."""

    def __init__(self, base_path: Path, snapshot_day: int, qrel_set: Optional[str] = None):
        self.base_path = Path(base_path)
        self.snapshot_day = snapshot_day
        self.qrel_set = qrel_set.lower() if qrel_set else None
        if not self.base_path.is_dir():
            raise FileNotFoundError(self.base_path)

        all_topics = {
            topic.topic_id: topic
            for topic in (_topic_from_record(row, topic_type="dynamic") for row in _jsonl_gz(self.base_path / "topics.jsonl.gz"))
        }
        # Add static topics from TREC Core 2018
        static_topics = get_static_topics(self.base_path)
        all_topics.update(static_topics)
                
        query_dates = {
            path.name.removesuffix(".jsonl.gz")
            for path in (self.base_path / "queries").glob("*.jsonl.gz")
        }
        document_dates = {
            path.name.removesuffix(".jsonl.gz")
            for path in (self.base_path / "documents").glob("*.jsonl.gz")
            if path.name != "index.jsonl.gz"
        }
        available_dates = query_dates & document_dates
        topic_dates = {
            topic_id: (datetime.strptime(topic.valid_from, "%Y-%m-%d").date() + timedelta(days=snapshot_day)).isoformat()
            for topic_id, topic in all_topics.items()
        }
        self._topic_dates = {
            topic_id: target_date
            for topic_id, target_date in topic_dates.items()
            if target_date in available_dates
        }
        self._topics = {topic_id: all_topics[topic_id] for topic_id in self._topic_dates}
        queries = DynamicSnapshotQueries(self._topic_dates, self.base_path, self._topics)
        query_records = list(queries.queries_iter())
        document_ids = {
            result.docno
            for query in query_records
            for result in query.serp
            if result.docno is not None
        }
        docs = DynamicSnapshotDocs(
            [self.base_path / "documents" / f"{day}.jsonl.gz" for day in sorted(set(self._topic_dates.values()))],
            document_ids,
            snapshot_day,
            self.base_path / "documents",
        )
        qrels = None
        if self.qrel_set:
            qrel_path = self.base_path / "qrels" / f"{self.qrel_set}.jsonl.gz"
            if not qrel_path.is_file():
                raise ValueError(f"Unknown qrel set: {qrel_set}")
            qrels = DynamicQrels(
                [qrel_path],
                {query.query_id for query in query_records},
                document_ids,
            )
        super().__init__(docs, queries, qrels)

    def topics_iter(self) -> Iterator[DynamicTopic]:
        yield from self._topics.values()

    def get_timestamp(self):
        return None

    def get_snapshot(self):
        return self.snapshot_day

    def get_datasets(self):
        return None

    def get_prior_datasets(self):
        return None


def register(base_path: Optional[Path] = None, name: str = NAME) -> None:
    """Register one standard dataset tag per complete calendar-day view.

    ``<name>/<date>`` includes the IN qrels. Snapshot tags are relative to
    each topic's emergence day: ``snapshot-N`` uses INN. RE judgments are
    attached to ``snapshot-0`` because they reassess that frozen document
    collection at a later day. ``base_path`` is optional so tests and local
    callers can register an already-extracted package directly.
    """
    if base_path is None:
        cache_path = home_path() / name
        dlc = DownloadConfig.context(name, cache_path)
        dataset_path = ZipExtractCache(dlc[name], cache_path).path()
    else:
        dataset_path = Path(base_path)
    
    query_dates = {
        path.name.removesuffix(".jsonl.gz")
        for path in (dataset_path / "queries").glob("*.jsonl.gz")
    }
    document_dates = {
        path.name.removesuffix(".jsonl.gz")
        for path in (dataset_path / "documents").glob("*.jsonl.gz")
    }
    re_sets = [
        path.name.removesuffix(".jsonl.gz")
        for path in (dataset_path / "qrels").glob("re*.jsonl.gz")
    ]
    day_datasets = []
    for snapshot_date in sorted(query_dates & document_dates):
        tag = f"{name}/{snapshot_date}"
        if tag not in registry:
            registry.register(tag, DynamicDataset(dataset_path, snapshot_date, qrel_set="in"))
        day_datasets.append(registry[tag])
        for re_set in re_sets:
            if f"{tag}/{re_set}" in registry:
                continue
            registry.register(
                f"{tag}/{re_set}",
                DynamicDataset(dataset_path, snapshot_date, qrel_set=re_set),
            )
    if f"{name}/days" not in registry:
        registry.register(f"{name}/days", MetaDataset(day_datasets))

    snapshot_datasets = []
    for snapshot_day in SNAPSHOT_DAYS:
        tag = f"{name}/snapshot-{snapshot_day}"
        if tag not in registry:
            qrel_set = f"in{snapshot_day}"
            if not (dataset_path / "qrels" / f"{qrel_set}.jsonl.gz").is_file():
                qrel_set = None
            registry.register(tag, DynamicSnapshotDataset(dataset_path, snapshot_day, qrel_set=qrel_set))
        snapshot_datasets.append(registry[tag])
    if f"{name}/snapshots" not in registry:
        registry.register(f"{name}/snapshots", MetaDataset(snapshot_datasets))

    for re_path in (dataset_path / "qrels").glob("re*.jsonl.gz"):
        re_set = re_path.name.removesuffix(".jsonl.gz")
        tag = f"{name}/snapshot-0/{re_set}"
        if tag not in registry:
            registry.register(
                tag,
                DynamicSnapshotDataset(dataset_path, 0, qrel_set=re_set),
            )
