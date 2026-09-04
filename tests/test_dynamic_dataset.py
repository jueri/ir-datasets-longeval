import gzip
import json
from pathlib import Path

from ir_datasets import registry

from ir_datasets_longeval import load
from ir_datasets_longeval.datasets.dynamic_dataset import (
    DynamicDataset,
    DynamicDocument,
    DynamicQrel,
    DynamicQuery,
    DynamicSerpResult,
    DynamicTopic,
    register,
)


def _write_jsonl_gz(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_temporal_topic_default_text_and_open_interval():
    topic = DynamicTopic(
        "t1", "Title", "Description", "Narrative", "2026-09-01", None,
        "model-name", "prompt-v1", "debate-1",
    )
    assert topic.default_text() == "Title Description Narrative"
    assert topic.valid_to is None


def test_temporal_document_uses_docno_for_ir_datasets_joins():
    document = DynamicDocument(
        "abc.2026-09-01", "abc", "https://example.test", "Title", "Body", "2026-08-31", "en",
        "abc.2026-09-01", "2026-09-01", "2026-09-02", "warcs/day.warc.gz",
        "abc.2026-09-01", "screenshots/day/abc.png",
    )
    assert document.doc_id == document.docno == "abc.2026-09-01"
    assert document.stable_doc_id == "abc"
    assert document.default_text() == "Title Body"
    qrel = DynamicQrel("q1", document.doc_id, 2, 2, 1, 2)
    assert qrel.doc_id == document.doc_id
    assert qrel.relevance_rater_2 == 1


def test_dynamic_query_keeps_topic_and_serp_outside_default_text():
    topic = DynamicTopic("t1", "Title", "Description", "Narrative", "2026-09-01", None, None, None, None)
    result = DynamicSerpResult(
        1, "Web", "https://example.test", "abc", "abc.2026-09-01",
        "abc.2026-09-01", "Result", "Snippet",
    )
    query = DynamicQuery("q1", "t1", "example query", "2026-09-01", None, (result,), topic)
    assert query.default_text() == "example query"
    assert query.topic is topic
    assert query.serp[0].docno == "abc.2026-09-01"


def test_dynamic_dataset_loads_one_daily_view_and_optional_qrels(tmp_path: Path):
    _write_jsonl_gz(tmp_path / "topics.jsonl.gz", [{
        "topic_id": "t1", "title": "Title", "description": "Description", "narrative": "Narrative",
        "valid_from": "2026-09-01", "valid_to": None, "llm_model": "model", "prompt": "prompt", "debate_id": "debate",
    }])
    _write_jsonl_gz(tmp_path / "documents" / "2026-09-01.jsonl.gz", [{
        "doc_id": "abc", "docno": "abc.2026-09-01", "url": "https://example.test", "title": "Title", "text": "Body",
        "content_version_id": "abc.2026-09-01", "valid_from": "2026-09-01", "valid_to": "2026-09-02",
        "warc": None, "warc_trec_id": None, "screenshot": None,
    }])
    _write_jsonl_gz(tmp_path / "queries" / "2026-09-01.jsonl.gz", [{
        "query_id": "q1", "topic_id": "t1", "text": "query", "date": "2026-09-01", "fetched_at": None,
        "results": [{"rank": 1, "source": "Web", "url": "https://example.test", "doc_id": "abc", "capture_docno": "abc.2026-09-01", "docno": "abc.2026-09-01"}],
    }])
    _write_jsonl_gz(tmp_path / "qrels" / "in0.jsonl.gz", [{
        "query_id": "q1", "docno": "abc.2026-09-01", "relevance": 2, "rater_1": "2", "rater_2": "1", "rater_3": "2",
    }])
    _write_jsonl_gz(tmp_path / "documents" / "2026-08-31.jsonl.gz", [{
        "doc_id": "abc", "docno": "abc.2026-08-31", "url": "https://example.test", "title": "Older", "text": "Older body",
        "content_version_id": "abc.2026-08-31", "valid_from": "2026-08-31", "valid_to": "2026-09-01",
        "warc": None, "warc_trec_id": None, "screenshot": None,
    }])
    _write_jsonl_gz(tmp_path / "documents" / "index.jsonl.gz", [
        {
            "doc_id": "abc", "docno": "abc.2026-08-31", "url": "https://example.test",
            "content_hash": "old", "valid_intervals": [{"valid_from": "2026-08-31", "valid_to": "2026-09-01"}],
        },
        {
            "doc_id": "abc", "docno": "abc.2026-09-01", "url": "https://example.test",
            "content_hash": "new", "valid_intervals": [{"valid_from": "2026-09-01", "valid_to": "2026-09-02"}],
        },
    ])
    _write_jsonl_gz(tmp_path / "queries" / "2026-08-31.jsonl.gz", [{
        "query_id": "q0", "topic_id": "t1", "text": "older query", "date": "2026-08-31", "fetched_at": None, "results": [],
    }])

    dataset = DynamicDataset(tmp_path, "2026-09-01", qrel_set="IN0")

    [topic] = dataset.topics_iter()
    [query] = dataset.queries_iter()
    [document] = dataset.docs_iter()
    [qrel] = dataset.qrels_iter()
    assert topic.topic_id == query.topic_id == "t1"
    assert query.topic is topic
    assert query.serp[0].docno == document.doc_id == qrel.doc_id
    assert qrel.relevance_rater_2 == 1
    [previous] = document.previous_versions()
    assert previous.doc_id == "abc.2026-08-31"
    assert previous.text == "Older body"
    assert previous.valid_to == "2026-09-01"
    assert previous.previous_versions() == []
    assert [prior.at for prior in dataset.get_prior_datasets()] == ["2026-08-31"]

    register(tmp_path, name="test-dynamic")
    assert "test-dynamic/2026-09-01" in registry
    assert "test-dynamic/2026-09-01/re7" not in registry

def test_tag():
    dataset = load("dynamic-dataset/2026-08-18")
    assert dataset.at == "2026-08-18"
