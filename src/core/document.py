from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class Document:
    """
    Represents a document in the knowledge base.
    
    Attributes:
        content (str): The actual text content of the document
        metadata (Dict): Additional information about the document
        embedding (np.ndarray, optional): Vector representation of the content
    """
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate document attributes after initialization."""
        if not isinstance(self.content, str):
            raise ValueError("Content must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            raise ValueError("Embedding must be a numpy array")
