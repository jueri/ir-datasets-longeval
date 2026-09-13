from pathlib import Path
from typing import NamedTuple

import pytest

from ir_datasets_longeval import load


class DatasetCase(NamedTuple):
    topic_type: str
    timeline: str
    snapshot: str | int
    qrels_type: str
    topics: int
    queries: int
    documents: int
    qrels: int

    @property
    def id(self):
        return (
            f"{self.topic_type}/"
            f"{self.timeline}/"
            f"{self.snapshot}/"
            f"{self.qrels_type}"
        )

def snapshot_id(param):
    return f"""{param["topic_type"]}/{param["timeline"]}/{param["snapshot"]}/{param["qrels_type"]}"""

DATASET_CASES = [
                # type      timeline          snapshot      qrels   topics      queries     documents       qrels
    DatasetCase( "dynamic",   "days",     "2026-07-28",      "IN",       1,           5,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-07-29",      "IN",       1,           5,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-07-30",      "IN",       3,          15,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-07-31",      "IN",       4,          20,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-01",      "IN",       4,          20,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-02",      "IN",       4,          20,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-03",      "IN",       4,          20,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-04",      "IN",       6,          30,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-05",      "IN",       6,          30,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-06",      "IN",       7,          35,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-07",      "IN",       7,          35,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-08",      "IN",       7,          35,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-09",      "IN",       7,          35,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-10",      "IN",       7,          35,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-11",      "IN",       8,          40,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-12",      "IN",      10,          50,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-13",      "IN",      10,          50,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-14",      "IN",      12,          60,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-15",      "IN",      12,          60,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-16",      "IN",      12,          60,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-17",      "IN",      12,          60,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-18",      "IN",      15,          75,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-19",      "IN",      15,          75,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-20",      "IN",      16,          80,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-21",      "IN",      17,          85,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-22",      "IN",      17,          85,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-23",      "IN",      17,          85,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-24",      "IN",      19,          95,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-25",      "IN",      19,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-26",      "IN",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-27",      "IN",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-28",      "IN",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-29",      "IN",      20,         100,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-30",      "IN",      20,         100,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-31",      "IN",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-01",      "IN",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-02",      "IN",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-03",      "IN",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-04",      "IN",      20,         100,           1,         1),
    # DatasetCase( "dynamic",   "days",     "2026-09-05",      "IN",      21,         105,           1,         0),
    # DatasetCase( "dynamic",   "days",     "2026-09-06",      "IN",      21,         105,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-07",      "IN",      21,         105,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-08",      "IN",      21,         105,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-09",      "IN",      22,         110,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-10",      "IN",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-11",      "IN",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-12",      "IN",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-13",      "IN",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-14",      "IN",      22,         110,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-15",      "IN",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-16",      "IN",      22,         110,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-17",      "IN",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-18",      "IN",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-19",      "IN",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-20",      "IN",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-21",      "IN",      22,         110,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-22",      "IN",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-23",      "IN",      22,         110,           1,         1),
    
    # RE
                # type      timeline          snapshot      qrels   topics      queries     documents       qrels
    DatasetCase( "dynamic",   "days",     "2026-07-28",      "RE",       1,           5,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-07-29",      "RE",       1,           5,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-07-30",      "RE",       3,          15,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-07-31",      "RE",       4,          20,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-01",      "RE",       4,          20,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-02",      "RE",       4,          20,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-03",      "RE",       4,          20,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-04",      "RE",       6,          30,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-05",      "RE",       6,          30,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-06",      "RE",       7,          35,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-07",      "RE",       7,          35,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-08",      "RE",       7,          35,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-09",      "RE",       7,          35,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-10",      "RE",       7,          35,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-11",      "RE",       8,          40,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-12",      "RE",      10,          50,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-13",      "RE",      10,          50,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-14",      "RE",      12,          60,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-15",      "RE",      12,          60,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-16",      "RE",      12,          60,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-17",      "RE",      12,          60,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-18",      "RE",      15,          75,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-19",      "RE",      15,          75,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-20",      "RE",      16,          80,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-21",      "RE",      17,          85,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-22",      "RE",      17,          85,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-23",      "RE",      17,          85,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-24",      "RE",      19,          95,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-25",      "RE",      19,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-26",      "RE",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-27",      "RE",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-28",      "RE",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-08-29",      "RE",      20,         100,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-30",      "RE",      20,         100,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-08-31",      "RE",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-01",      "RE",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-02",      "RE",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-03",      "RE",      20,         100,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-04",      "RE",      20,         100,           1,         1),
    # DatasetCase( "dynamic",   "days",     "2026-09-05",      "RE",      21,         105,           1,         0),
    # DatasetCase( "dynamic",   "days",     "2026-09-06",      "RE",      21,         105,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-07",      "RE",      21,         105,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-08",      "RE",      21,         105,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-09",      "RE",      22,         110,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-10",      "RE",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-11",      "RE",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-12",      "RE",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-13",      "RE",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-14",      "RE",      22,         110,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-15",      "RE",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-16",      "RE",      22,         110,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-17",      "RE",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-18",      "RE",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-19",      "RE",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-20",      "RE",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-21",      "RE",      22,         110,           1,         1),
    DatasetCase( "dynamic",   "days",     "2026-09-22",      "RE",      22,         110,           1,         0),
    DatasetCase( "dynamic",   "days",     "2026-09-23",      "RE",      22,         110,           1,         1),
    
    
    # Static
    # IN
                 # type      timeline          snapshot      qrels   topics      queries     documents       qrels
    DatasetCase( "static",     "days",     "2026-09-01",      "IN",      10,          50,            1,         1),
    DatasetCase( "static",     "days",     "2026-09-02",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-03",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-04",      "IN",      10,          50,            1,         0),
    # DatasetCase( "static",     "days",     "2026-09-05",      "IN",      10,          50,            1,         0),
    # DatasetCase( "static",     "days",     "2026-09-06",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-07",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-08",      "IN",      10,          50,            1,         1),
    DatasetCase( "static",     "days",     "2026-09-09",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-10",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-11",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-12",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-13",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-14",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-15",      "IN",      10,          50,            1,         1),
    DatasetCase( "static",     "days",     "2026-09-16",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-17",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-18",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-19",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-20",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-21",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-22",      "IN",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-23",      "IN",      10,          50,            1,         0),
    
        # RE
                 # type      timeline          snapshot      qrels   topics      queries     documents       qrels
    DatasetCase( "static",     "days",     "2026-09-01",      "RE",      10,          50,            1,         1),
    DatasetCase( "static",     "days",     "2026-09-02",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-03",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-04",      "RE",      10,          50,            1,         0),
    # DatasetCase( "static",     "days",     "2026-09-05",      "RE",      10,          50,            1,         0),
    # DatasetCase( "static",     "days",     "2026-09-06",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-07",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-08",      "RE",      10,          50,            1,         1),
    DatasetCase( "static",     "days",     "2026-09-09",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-10",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-11",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-12",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-13",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-14",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-15",      "RE",      10,          50,            1,         1),
    DatasetCase( "static",     "days",     "2026-09-16",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-17",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-18",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-19",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-20",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-21",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-22",      "RE",      10,          50,            1,         0),
    DatasetCase( "static",     "days",     "2026-09-23",      "RE",      10,          50,            1,         0),

    ####
    # snapshot
            # RE
                 # type          timeline   snapshot      qrels    topics      queries     documents       qrels
    DatasetCase( "dynamic",    "snapshot",         0,       "RE",      22,         110,            1,         1),
    DatasetCase( "dynamic",    "snapshot",         1,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         2,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         3,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         4,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         5,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         6,       "RE",      22,         110,            1,         1),
    DatasetCase( "dynamic",    "snapshot",         7,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         8,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         9,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        10,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        11,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        12,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        13,       "RE",      22,         110,            1,         1),
    DatasetCase( "dynamic",    "snapshot",        14,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        15,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        16,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        17,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        18,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        19,       "RE",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        20,       "RE",      22,         110,            1,         0),
    
    # IN    
                     # type          timeline   snapshot      qrels    topics      queries     documents       qrels
    DatasetCase( "dynamic",    "snapshot",         0,       "IN",      22,         110,            1,         1),
    DatasetCase( "dynamic",    "snapshot",         1,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         2,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         3,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         4,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         5,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         6,       "IN",      22,         110,            1,         1),
    DatasetCase( "dynamic",    "snapshot",         7,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         8,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",         9,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        10,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        11,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        12,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        13,       "IN",      22,         110,            1,         1),
    DatasetCase( "dynamic",    "snapshot",        14,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        15,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        16,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        17,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        18,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        19,       "IN",      22,         110,            1,         0),
    DatasetCase( "dynamic",    "snapshot",        20,       "IN",      22,         110,            1,         0),
]

@pytest.fixture(
    scope="class",
    params=DATASET_CASES,
    ids=lambda case: case.id,
)
def dataset_case(request):
    case = request.param

    dataset = load(
        f"dynamic-dataset/"
        f"{case.topic_type}/"
        f"{case.timeline}/"
        f"{case.snapshot}/"
        f"{case.qrels_type}"
    )
    return dataset, case
    
def test_dataset_exists(dataset_case):
    dataset, expected = dataset_case
    assert dataset is not None
    assert dataset.get_snapshot() == expected.snapshot

def test_qrels(dataset_case):
    dataset, expected = dataset_case
    if expected.qrels > 0:
        assert dataset.has_qrels()
        qrels = list(dataset.qrels_iter())
        assert len(qrels) > 0
    elif expected.qrels == 0:
        assert not dataset.has_qrels()
    
def test_queries(dataset_case):
    dataset, expected = dataset_case
    queries = list(dataset.queries_iter())
    print(f"Found {len(queries)} topics in dataset {dataset.get_snapshot()}")
    topic_ids = [topic.topic_id for topic in queries]
    assert len(set(topic_ids)) == expected.topics
    assert all(query.topic_id == query.topic.topic_id for query in queries)



def test_serp(dataset_case):
    dataset, expected = dataset_case
    queries = list(dataset.queries_iter())
    for query in queries:
        assert query.serp_file.exists()

def test_docs(dataset_case):
    dataset, expected = dataset_case
    docs = list(dataset.docs_iter())
    print(f"Found {len(docs)} docs in dataset {dataset.get_snapshot()}")
    assert len(docs) > 0