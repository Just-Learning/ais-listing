"""Vector store management for real estate listings."""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Union
import numpy as np
import faiss
import chromadb
from chromadb.config import Settings
import pandas as pd


class VectorStore:
    """Base vector store class for managing embeddings."""
    
    def __init__(self, store_path: str):
        """Initialize vector store.
        
        Args:
            store_path: Path to store the vector database
        """
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add embeddings to the store.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
        """
        raise NotImplementedError
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        raise NotImplementedError
    
    def save(self):
        """Save the vector store to disk."""
        raise NotImplementedError
    
    def load(self):
        """Load the vector store from disk."""
        raise NotImplementedError


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for fast similarity search."""
    
    def __init__(self, store_path: str, dimension: int = 384):
        """Initialize FAISS vector store.
        
        Args:
            store_path: Path to store the FAISS index
            dimension: Dimension of embeddings
        """
        super().__init__(store_path)
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.index_path = os.path.join(store_path, "faiss_index.bin")
        self.metadata_path = os.path.join(store_path, "metadata.json")
        
        # Try to load existing index
        if os.path.exists(self.index_path):
            self.load()
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add embeddings to FAISS index.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
        """
        if self.index is None:
            # Initialize index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings using FAISS.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            List of search results with metadata and scores
        """
        if self.index is None:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Return results with metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def save(self):
        """Save FAISS index and metadata."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load(self):
        """Load FAISS index and metadata."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store for persistent storage."""
    
    def __init__(self, store_path: str, collection_name: str = "real_estate_listings"):
        """Initialize ChromaDB vector store.
        
        Args:
            store_path: Path to store ChromaDB
            collection_name: Name of the collection
        """
        super().__init__(store_path)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=store_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(collection_name)
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], 
                      ids: Optional[List[str]] = None):
        """Add embeddings to ChromaDB.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
            ids: Optional list of IDs for the embeddings
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(embeddings))]
        
        # Convert embeddings to list format
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            metadatas=metadata,
            ids=ids
        )
    
    def search(self, query_embedding: np.ndarray, k: int = 5, 
               where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings using ChromaDB.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            where: Optional filter conditions
            
        Returns:
            List of search results with metadata and scores
        """
        # Convert embedding to list format
        query_embedding_list = query_embedding.tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=k,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i, (doc_id, metadata, distance) in enumerate(zip(
                results['ids'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                result = metadata.copy()
                result['id'] = doc_id
                result['distance'] = float(distance)
                result['score'] = 1.0 - float(distance)  # Convert distance to similarity
                result['rank'] = i + 1
                formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        return {
            'name': self.collection.name,
            'count': self.collection.count(),
            'metadata': self.collection.metadata
        }
    
    def delete_embeddings(self, ids: List[str]):
        """Delete embeddings by IDs.
        
        Args:
            ids: List of IDs to delete
        """
        self.collection.delete(ids=ids)
    
    def update_embeddings(self, ids: List[str], embeddings: np.ndarray, 
                         metadata: List[Dict[str, Any]]):
        """Update existing embeddings.
        
        Args:
            ids: List of IDs to update
            embeddings: New embeddings
            metadata: New metadata
        """
        embeddings_list = embeddings.tolist()
        
        self.collection.update(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadata
        )


class RealEstateVectorStore:
    """Specialized vector store for real estate listings."""
    
    def __init__(self, store_path: str, use_chroma: bool = True):
        """Initialize real estate vector store.
        
        Args:
            store_path: Path to store the vector database
            use_chroma: Whether to use ChromaDB (True) or FAISS (False)
        """
        self.store_path = store_path
        self.use_chroma = use_chroma
        
        if use_chroma:
            self.vector_store = ChromaVectorStore(store_path)
        else:
            self.vector_store = FAISSVectorStore(store_path)
    
    def add_listings(self, listings: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add real estate listings with their embeddings.
        
        Args:
            listings: List of listing dictionaries
            embeddings: Numpy array of embeddings
        """
        # Prepare metadata
        metadata = []
        ids = []
        
        for i, listing in enumerate(listings):
            meta = {
                'title': listing.get('title', ''),
                'price': listing.get('price', ''),
                'location': listing.get('location', ''),
                'bedrooms': listing.get('bedrooms', ''),
                'bathrooms': listing.get('bathrooms', ''),
                'square_feet': listing.get('square_feet', ''),
                'property_type': listing.get('property_type', ''),
                'listing_id': listing.get('id', f'listing_{i}'),
                'description_length': len(listing.get('description', ''))
            }
            metadata.append(meta)
            # Ensure ID is a string
            listing_id = listing.get('id', f'listing_{i}')
            ids.append(str(listing_id))
        
        # Add to vector store
        if self.use_chroma:
            self.vector_store.add_embeddings(embeddings, metadata, ids)
        else:
            self.vector_store.add_embeddings(embeddings, metadata)
    
    def search_similar_listings(self, query_text: str, embedding_model, 
                               k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar listings.
        
        Args:
            query_text: Query text
            embedding_model: Embedding model instance
            k: Number of results
            filters: Optional filters for ChromaDB
            
        Returns:
            List of similar listings
        """
        # Encode query
        query_embedding = embedding_model.encode_single(query_text)
        
        # Search
        if self.use_chroma:
            results = self.vector_store.search(query_embedding, k, filters)
        else:
            results = self.vector_store.search(query_embedding, k)
        
        return results
    
    def search_by_filters(self, filters: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
        """Search listings by metadata filters (ChromaDB only).
        
        Args:
            filters: Filter conditions
            k: Number of results
            
        Returns:
            List of filtered listings
        """
        if not self.use_chroma:
            raise ValueError("Filter search is only available with ChromaDB")
        
        # Get all results with filters
        results = self.vector_store.collection.get(
            where=filters,
            limit=k
        )
        
        # Format results
        formatted_results = []
        for i, (doc_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
            result = metadata.copy()
            result['id'] = doc_id
            result['rank'] = i + 1
            formatted_results.append(result)
        
        return formatted_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the stored listings.
        
        Returns:
            Dictionary with statistics
        """
        if self.use_chroma:
            info = self.vector_store.get_collection_info()
            return {
                'total_listings': info['count'],
                'store_type': 'ChromaDB',
                'collection_name': info['name']
            }
        else:
            return {
                'total_listings': len(self.vector_store.metadata),
                'store_type': 'FAISS',
                'index_size': self.vector_store.index.ntotal if self.vector_store.index else 0
            }
    
    def save(self):
        """Save the vector store."""
        # ChromaDB saves automatically, no need for explicit save
        pass
    
    def load(self):
        """Load the vector store."""
        self.vector_store.load() 