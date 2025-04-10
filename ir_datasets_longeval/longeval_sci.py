import contextlib
import json
import os
from datetime import datetime
from pathlib import Path
from pkgutil import get_data
from shutil import copyfile
from typing import Dict, List, NamedTuple, Optional

from ir_datasets import registry
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import JsonlDocs, TrecQrels, TsvQueries
from ir_datasets.util import ZipExtractCache, home_path

from ir_datasets_longeval.formats import MetaDataset
from ir_datasets_longeval.util import DownloadConfig, YamlDocumentation

NAME = "longeval-sci"
QREL_DEFS = {
    2: "highly relevant",
    1: "relevant",
    0: "not relevant",
}
SUB_COLLECTIONS = ["2024-11"]
MAPPING = {
    "doc_id": "id",
    "title": "title",
    "abstract": "abstract",
    "authors": "authors",
    "createdDate": "createdDate",
    "doi": "doi",
    "arxivId": "arxivId",
    "pubmedId": "pubmedId",
    "magId": "magId",
    "oaiIds": "oaiIds",
    "links": "links",
    "publishedDate": "publishedDate",
    "updatedDate": "updatedDate",
}


class LongEvalSciDoc(NamedTuple):
    doc_id: str
    title: str
    abstract: str
    authors: List[Dict[str, str]]
    createdDate: Optional[str]
    doi: Optional[str]
    arxivId: Optional[str]
    pubmedId: Optional[str]
    magId: Optional[str]
    oaiIds: Optional[List[str]]
    links: List[Dict[str, str]]
    publishedDate: str
    updatedDate: str

    def default_text(self):
        ret = self.title
        if self.abstract:
            ret += " " + self.abstract
        return ret


class ExtractedPath:
    def __init__(self, path):
        self._path = path

    @contextlib.contextmanager
    def stream(self):
        with open(self._path, "rb") as f:
            yield f


class LongEvalSciDataset(Dataset):
    def __init__(
        self,
        base_path: Path,
        yaml_documentation: str = "longeval_sci.yaml",
        timestamp: Optional[str] = None,
        prior_datasets: Optional[List[str]] = None,
        lag: Optional[str] = None,
    ):
        documentation = YamlDocumentation(yaml_documentation)
        self.base_path = base_path

        if not base_path or not base_path.exists() or not base_path.is_dir():
            raise FileNotFoundError(
                f"I expected that the directory {base_path} exists. But the directory does not exist."
            )

        if not timestamp:
            timestamp = self.read_property_from_metadata("timestamp")

        self.timestamp = datetime.strptime(timestamp, "%Y-%m")

        if not lag:
            try:
                print("read lag from metadata")
                lag = self.read_property_from_metadata("lag")
            except KeyError:
                lag = None
        self.lag = lag

        if prior_datasets is None:
            prior_datasets = self.read_property_from_metadata("prior-datasets")

        self.prior_datasets = prior_datasets

        docs_path = base_path / "documents"

        if not docs_path.exists() or not docs_path.is_dir():
            raise FileNotFoundError(
                f"I expected that the directory {docs_path} exists. But the directory does not exist."
            )

        jsonl_doc_files = os.listdir(docs_path)
        if len(jsonl_doc_files) == 0:
            raise ValueError(
                f"The directory {docs_path} has no jsonl files. This is likely an arror."
            )

        docs_paths = [
            ExtractedPath(base_path / "documents" / split) for split in jsonl_doc_files
        ]
        docs = JsonlDocs(
            docs_paths,
            doc_cls=LongEvalSciDoc,
            docstore_path=f"{docs_path}/docstore.pklz4",
            mapping=MAPPING,
        )

        queries_path = base_path / "queries.txt"
        if not queries_path.exists() or not queries_path.is_file():
            raise FileNotFoundError(
                f"I expected that the file {queries_path} exists. But the directory does not exist."
            )

        queries = TsvQueries(ExtractedPath(queries_path))

        qrels = None
        qrels_path = base_path / "qrels.txt"
        if qrels_path.exists() and qrels_path.is_file():
            qrels = TrecQrels(ExtractedPath(qrels_path), QREL_DEFS)

        super().__init__(docs, queries, qrels, documentation)

    def get_timestamp(self):
        return self.timestamp

    def get_lag(self):
        return self.lag

    def get_lags(self):
        return None

    def get_past_datasets(self):
        return [LongEvalSciDataset(self.base_path / i) for i in self.prior_datasets]

    def read_property_from_metadata(self, property):
        try:
            return json.load(open(self.base_path / "metadata.json", "r"))[property]
        except FileNotFoundError:
            metadata = json.loads(get_data("ir_datasets_longeval", "metadata.json"))
            return metadata[f"longeval-sci/{self.timestamp.strftime('%Y-%m')}/train"][
                property
            ]


def register_spot_check_datasets():
    if f"{NAME}/spot-check/no-prior-data" in registry:
        return

    base_path = home_path() / NAME / "spot-check"
    dlc = DownloadConfig.context(NAME, base_path)

    def extract_from_tira(name):
        return ZipExtractCache(dlc[name], base_path / name).path() / name.replace(
            "-inputs", ""
        ).replace("-truths", "")

    no_prior_data_inputs = extract_from_tira(
        "sci-spot-check-no-prior-data-20250322-training-inputs"
    )
    with_prior_data_inputs = extract_from_tira(
        "sci-spot-check-with-prior-data-20250322-training-inputs"
    )
    no_prior_data_truths = extract_from_tira(
        "sci-spot-check-no-prior-data-20250322-training-truths"
    )
    with_prior_data_truths = extract_from_tira(
        "sci-spot-check-with-prior-data-20250322-training-truths"
    )

    if not (no_prior_data_inputs / "qrels.txt").exists():
        copyfile(no_prior_data_truths / "qrels.txt", no_prior_data_inputs / "qrels.txt")

    if not (with_prior_data_inputs / "qrels.txt").exists():
        copyfile(
            with_prior_data_truths / "qrels.txt", with_prior_data_inputs / "qrels.txt"
        )

    no_prior = LongEvalSciDataset(no_prior_data_inputs, lag="lag-3")
    prior_data = LongEvalSciDataset(with_prior_data_inputs, lag="lag-4")
    registry.register(f"{NAME}/spot-check/no-prior-data", no_prior)
    registry.register(f"{NAME}/spot-check/with-prior-data", prior_data)
    registry.register(f"{NAME}/spot-check/*", MetaDataset([no_prior, prior_data]))


def register():
    if f"{NAME}/2024-11/train" in registry:
        # Already registered.
        return

    base_path = home_path() / NAME

    dlc = DownloadConfig.context(NAME, base_path)
    base_path = home_path() / NAME

    data_path = (
        ZipExtractCache(
            dlc["longeval_sci_training_2025"], base_path / "longeval_sci_training_2025"
        ).path()
        / "longeval_sci_training_2025_abstract"
    )

    subsets = {}

    subsets["2024-11/train"] = LongEvalSciDataset(
        data_path, "2024-11/train", "2024-11", []
    )

    for s in sorted(subsets):
        registry.register(f"{NAME}/{s}", subsets[s])

    if f"{NAME}/*" in registry:
        # Already registered.
        return
    registry.register(f"{NAME}/*", MetaDataset(list(subsets.values())))
