import unittest

from ir_datasets_longeval import load


class TestOfficialDatasets(unittest.TestCase):
    def test_sci_dataset(self):
        dataset = load("longeval-sci/2024-11/train")

        expected_queries = {"ce5bfacf-8652-4bc1-a5b0-6144a917fb1c": "streptomyces"}
        expected_qrels = [
            {"doc_id": "140120179", "query_id": "1234-1234-1234-1234-1234", "rel": 2}
        ]

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(393, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(4262, len(list(dataset.qrels_iter())))

        # Docs
        self.assertIsNotNone(example_doc.doc_id)
        self.assertEqual("127164364", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("68859258", docs_store.get("68859258").doc_id)

        # Timestamp
        self.assertEqual(2024, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual([], dataset.get_past_datasets())

        # Lag
        self.assertEqual("lag-0", dataset.get_lag())

    def test_web_dataset(self):
        dataset = load("longeval-web/2022-06")

        expected_queries = {"8": "4 mariages 1 enterrement"}

        # Dataset
        self.assertIsNotNone(dataset)
        example_doc = dataset.docs_iter().__next__()

        # Queries
        actual_queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
        self.assertEqual(75427, len(actual_queries))
        for k, v in expected_queries.items():
            self.assertEqual(v, actual_queries[k])

        # Qrels
        self.assertEqual(85776, len(list(dataset.qrels_iter())))
        
        # Docs
        self.assertEqual("938880", example_doc.doc_id)

        # Docstore
        docs_store = dataset.docs_store()
        self.assertEqual("44971", docs_store.get("44971").doc_id)
        
        # Timestamp
        self.assertEqual(2022, dataset.get_timestamp().year)

        # Prior datasets
        self.assertEqual([], dataset.get_past_datasets())
        
        # Lag
        self.assertEqual("lag-0", dataset.get_lag())


    def test_all_sci_datasets(self):
        dataset_id = "longeval-sci/*"
        meta_dataset = load(dataset_id)

        with self.assertRaises(AttributeError):
            meta_dataset.queries_iter()

        with self.assertRaises(AttributeError):
            meta_dataset.docs_iter()

        lags = meta_dataset.get_lags()
        self.assertEqual(1, len(lags))
        self.assertEqual("lag-0", lags[0].get_lag())


    def test_all_web_datasets(self):
        dataset_id = "longeval-web/*"
        meta_dataset = load(dataset_id)

        with self.assertRaises(AttributeError):
            meta_dataset.queries_iter()

        with self.assertRaises(AttributeError):
            meta_dataset.docs_iter()

        lags = meta_dataset.get_lags()
        self.assertEqual(9, len(lags))
        self.assertEqual("lag-0", lags[0].get_lag())
