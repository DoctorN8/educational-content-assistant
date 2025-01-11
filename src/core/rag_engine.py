from typing import List, Dict, Optional
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from .document import Document

class RAGEngine:
    """
    Core engine that handles document retrieval and embeddings.
    """
    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the RAG engine with specified models.

        Args:
            embedding_model (str): Name of the Hugging Face model to use for embeddings
        """
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.knowledge_base = []
        self.index = None
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a piece of text.

        Args:
            text (str): Text to generate embedding for

        Returns:
            np.ndarray: Vector representation of the text
        """
        inputs = self.tokenizer(text, return_tensors="pt", 
                              truncation=True, max_length=512, 
                              padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()
            
        return embedding[0]
        
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the knowledge base.

        Args:
            documents (List[Dict[str, str]]): List of document dictionaries
        """
        for doc in documents:
            content = doc['question_text']
            if 'explanation' in doc and doc['explanation']:
                content += f"\nExplanation: {doc['explanation']}"
                
            embedding = self._get_embedding(content)
            
            self.knowledge_base.append(Document(
                content=content,
                metadata=doc,
                embedding=embedding
            ))
            
        embeddings = np.array([doc.embedding for doc in self.knowledge_base])
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query (str): Search query
            k (int): Number of documents to retrieve

        Returns:
            List[Document]: List of most relevant documents
        """
        query_embedding = self._get_embedding(query)
        
        D, I = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            k
        )
        
        return [self.knowledge_base[i] for i in I[0]]
