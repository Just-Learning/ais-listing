"""Text chunking methods for real estate listings."""

import re
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
import numpy as np


class ChunkingMethod(ABC):
    """Abstract base class for chunking methods."""
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Chunk text into smaller pieces.
        
        Args:
            text: Input text to chunk
            **kwargs: Additional parameters for chunking
            
        Returns:
            List of text chunks
        """
        pass
    
    def get_method_name(self) -> str:
        """Get the name of the chunking method."""
        return self.__class__.__name__


class SentenceChunking(ChunkingMethod):
    """Chunk text by sentences."""
    
    def __init__(self, min_chunk_length: int = 10, max_chunk_length: int = 500):
        """Initialize sentence chunking.
        
        Args:
            min_chunk_length: Minimum number of characters per chunk
            max_chunk_length: Maximum number of characters per chunk
        """
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Chunk text by sentences.
        
        Args:
            text: Input text to chunk
            **kwargs: Additional parameters (min_length, max_length)
            
        Returns:
            List of sentence chunks
        """
        min_length = kwargs.get('min_length', self.min_chunk_length)
        max_length = kwargs.get('max_length', self.max_chunk_length)
        
        if not text or len(text.strip()) == 0:
            return []
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed max length, save current chunk
            if current_chunk and len(current_chunk + " " + sentence) > max_length:
                if len(current_chunk) >= min_length:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it meets minimum length
        if current_chunk and len(current_chunk) >= min_length:
            chunks.append(current_chunk.strip())
        
        return chunks


class WordCountChunking(ChunkingMethod):
    """Chunk text by word count with overlap."""
    
    def __init__(self, chunk_size: int = 100, overlap: int = 20):
        """Initialize word count chunking.
        
        Args:
            chunk_size: Number of words per chunk
            overlap: Number of overlapping words between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Chunk text by word count.
        
        Args:
            text: Input text to chunk
            **kwargs: Additional parameters (chunk_size, overlap)
            
        Returns:
            List of word-based chunks
        """
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        overlap = kwargs.get('overlap', self.overlap)
        
        if not text or len(text.strip()) == 0:
            return []
        
        # Tokenize into words
        words = word_tokenize(text)
        
        if len(words) <= chunk_size:
            return [" ".join(words)]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(words):
                break
        
        return chunks


class SemanticChunking(ChunkingMethod):
    """Chunk text based on semantic similarity using embeddings."""
    
    def __init__(self, embedding_model=None, similarity_threshold: float = 0.7):
        """Initialize semantic chunking.
        
        Args:
            embedding_model: Pre-trained embedding model
            similarity_threshold: Threshold for combining similar sentences
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Chunk text based on semantic similarity.
        
        Args:
            text: Input text to chunk
            **kwargs: Additional parameters (similarity_threshold)
            
        Returns:
            List of semantically grouped chunks
        """
        if not self.embedding_model:
            raise ValueError("Embedding model is required for semantic chunking")
        
        similarity_threshold = kwargs.get('similarity_threshold', self.similarity_threshold)
        
        if not text or len(text.strip()) == 0:
            return []
        
        # Split into sentences
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return sentences
        
        # Get embeddings for all sentences
        embeddings = self.embedding_model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i]
            
            # Calculate similarity
            similarity = self._cosine_similarity(current_embedding, sentence_embedding)
            
            if similarity >= similarity_threshold:
                # Add to current chunk
                current_chunk.append(sentence)
                # Update current embedding (average of all sentences in chunk)
                current_embedding = np.mean(embeddings[len(chunks):i+1], axis=0)
            else:
                # Start new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_embedding = sentence_embedding
        
        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class ChunkingFactory:
    """Factory for creating chunking methods."""
    
    @staticmethod
    def create_chunker(method: str, **kwargs) -> ChunkingMethod:
        """Create a chunking method instance.
        
        Args:
            method: Chunking method name
            **kwargs: Parameters for the chunking method
            
        Returns:
            ChunkingMethod instance
        """
        if method == "sentence":
            return SentenceChunking(**kwargs)
        elif method == "word_count":
            return WordCountChunking(**kwargs)
        elif method == "semantic":
            return SemanticChunking(**kwargs)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available chunking methods."""
        return ["sentence", "word_count", "semantic"]


def chunk_property_descriptions(property_listings: List[Dict[str, Any]], 
                               chunking_method: str = "sentence",
                               **kwargs) -> List[Dict[str, Any]]:
    """Chunk property descriptions using specified method.
    
    Args:
        property_listings: List of property dictionaries
        chunking_method: Method to use for chunking
        **kwargs: Parameters for the chunking method
        
    Returns:
        List of property dictionaries with chunked descriptions
    """
    chunker = ChunkingFactory.create_chunker(chunking_method, **kwargs)
    
    chunked_properties = []
    
    for prop in property_listings:
        description = prop.get('description', '')
        if not description:
            continue
        
        chunks = chunker.chunk_text(description)
        
        for i, chunk in enumerate(chunks):
            chunked_prop = prop.copy()
            chunked_prop['description'] = chunk
            chunked_prop['chunk_id'] = f"{prop.get('id', 'unknown')}_{i}"
            chunked_prop['chunk_index'] = i
            chunked_prop['total_chunks'] = len(chunks)
            chunked_prop['chunking_method'] = chunking_method
            
            chunked_properties.append(chunked_prop)
    
    return chunked_properties 