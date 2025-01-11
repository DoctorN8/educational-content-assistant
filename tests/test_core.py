import unittest
import numpy as np
from src.core.document import Document
from src.core.rag_engine import RAGEngine

class TestDocument(unittest.TestCase):
    def test_document_creation(self):
        doc = Document(
            content="Test content",
            metadata={"type": "test"},
            embedding=np.array([1.0, 2.0, 3.0])
        )
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata["type"], "test")
        self.assertTrue(np.array_equal(doc.embedding, np.array([1.0, 2.0, 3.0])))

    def test_invalid_document(self):
        with self.assertRaises(ValueError):
            Document(
                content=123,  # Should be string
                metadata={"type": "test"},
                embedding=np.array([1.0, 2.0, 3.0])
            )

class TestRAGEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RAGEngine()
        
    def test_embedding_generation(self):
        text = "This is a test document"
        embedding = self.engine._get_embedding(text)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding.shape), 1)  # Should be 1D array

if __name__ == '__main__':
    unittest.main()
