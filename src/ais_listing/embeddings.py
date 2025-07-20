"""Embedding model handling for real estate listings."""

import os
from typing import List, Optional, Union, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import json
import time
from .chunking import ChunkingFactory, chunk_property_descriptions


class EmbeddingModel:
    """Handle embedding generation and operations for real estate listings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode text(s) to embeddings.
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Handle empty or None texts
        texts = [text if text else "" for text in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of embedding
        """
        return self.encode([text])[0]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        embeddings = self.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def batch_similarity(self, query_text: str, candidate_texts: List[str]) -> List[float]:
        """Calculate similarities between a query and multiple candidates.
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            
        Returns:
            List of similarity scores
        """
        all_texts = [query_text] + candidate_texts
        embeddings = self.encode(all_texts)
        
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        similarities = []
        for candidate_embedding in candidate_embeddings:
            similarity = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )
            similarities.append(float(similarity))
        
        return similarities
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """Find the most similar texts to a query.
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with text, index, and similarity score
        """
        similarities = self.batch_similarity(query_text, candidate_texts)
        
        # Create list of (index, similarity) pairs
        indexed_similarities = list(enumerate(similarities))
        
        # Sort by similarity (descending)
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = []
        for idx, similarity in indexed_similarities[:top_k]:
            results.append({
                'index': idx,
                'text': candidate_texts[idx],
                'similarity': similarity
            })
        
        return results
    
    def cluster_embeddings(self, texts: List[str], n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster texts based on their embeddings.
        
        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with cluster assignments and centroids
        """
        from sklearn.cluster import KMeans
        
        embeddings = self.encode(texts)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group texts by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'index': i,
                'text': texts[i]
            })
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centroids': kmeans.cluster_centers_.tolist(),
            'clusters': clusters
        }
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file.
        
        Args:
            embeddings: Numpy array of embeddings
            filepath: Path to save file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if filepath.endswith('.npy'):
            np.save(filepath, embeddings)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
        else:
            raise ValueError("Filepath must end with .npy or .pkl")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file.
        
        Args:
            filepath: Path to embeddings file
            
        Returns:
            Numpy array of embeddings
        """
        if filepath.endswith('.npy'):
            return np.load(filepath)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Filepath must end with .npy or .pkl")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'max_seq_length': self.model.max_seq_length
        }


class RealEstateEmbeddingModel(EmbeddingModel):
    """Specialized embedding model for real estate listings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the real estate embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        super().__init__(model_name)
        
        # Real estate specific keywords for enhanced embeddings
        self.real_estate_keywords = [
            'bedroom', 'bathroom', 'kitchen', 'living room', 'dining room',
            'garage', 'parking', 'garden', 'balcony', 'terrace', 'pool',
            'fireplace', 'hardwood', 'carpet', 'granite', 'stainless steel',
            'central air', 'heating', 'cooling', 'washer', 'dryer', 'dishwasher',
            'modern', 'updated', 'renovated', 'new', 'old', 'vintage',
            'luxury', 'affordable', 'spacious', 'cozy', 'bright', 'sunny',
            'quiet', 'noisy', 'safe', 'convenient', 'walkable', 'transit'
        ]
    
    def encode_with_context(self, text: str, context: Dict[str, Any] = None) -> np.ndarray:
        """Encode text with additional real estate context.
        
        Args:
            text: Input text
            context: Additional context (price, location, etc.)
            
        Returns:
            Enhanced embedding
        """
        # Add context information to text if provided
        if context:
            context_text = " ".join([f"{k}: {v}" for k, v in context.items() if v])
            enhanced_text = f"{text} {context_text}"
        else:
            enhanced_text = text
        
        return self.encode_single(enhanced_text)
    
    def find_similar_properties(self, query_text: str, property_listings: List[Dict[str, Any]], 
                               top_k: int = 5, min_similarity: float = 0.5) -> List[Dict[str, Any]]:
        """Find similar properties based on listing descriptions.
        
        Args:
            query_text: Query description
            property_listings: List of property dictionaries with 'description' key
            top_k: Number of top results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar properties with similarity scores
        """
        descriptions = [prop.get('description', '') for prop in property_listings]
        similarities = self.batch_similarity(query_text, descriptions)
        
        # Create results with property info and similarity
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= min_similarity:
                result = property_listings[i].copy()
                result['similarity_score'] = similarity
                results.append(result)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def find_similar_properties_with_chunking(self, query_text: str, property_listings: List[Dict[str, Any]],
                                            chunking_method: str = "sentence", top_k: int = 5, 
                                            min_similarity: float = 0.5, **chunking_kwargs) -> List[Dict[str, Any]]:
        """Find similar properties using chunked descriptions.
        
        Args:
            query_text: Query description
            property_listings: List of property dictionaries
            chunking_method: Method to use for chunking descriptions
            top_k: Number of top results
            min_similarity: Minimum similarity threshold
            **chunking_kwargs: Parameters for the chunking method
            
        Returns:
            List of similar properties with similarity scores and chunk information
        """
        # Chunk the property descriptions
        chunked_properties = chunk_property_descriptions(
            property_listings, 
            chunking_method=chunking_method, 
            **chunking_kwargs
        )
        
        # Find similar chunks
        chunk_descriptions = [prop.get('description', '') for prop in chunked_properties]
        similarities = self.batch_similarity(query_text, chunk_descriptions)
        
        # Create results with chunk info and similarity
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= min_similarity:
                result = chunked_properties[i].copy()
                result['similarity_score'] = similarity
                results.append(result)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def compare_chunking_methods(self, query_text: str, property_listings: List[Dict[str, Any]],
                               methods: List[str] = None, top_k: int = 5, 
                               min_similarity: float = 0.5) -> Dict[str, Any]:
        """Compare search performance across different chunking methods.
        
        Args:
            query_text: Query description
            property_listings: List of property dictionaries
            methods: List of chunking methods to compare
            top_k: Number of top results
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with comparison results for each method
        """
        if methods is None:
            methods = ["sentence", "word_count"]
        
        comparison_results = {}
        
        for method in methods:
            start_time = time.time()
            
            try:
                # Get results for this method
                if method == "no_chunking":
                    results = self.find_similar_properties(
                        query_text, property_listings, top_k, min_similarity
                    )
                    chunk_count = len(property_listings)
                else:
                    results = self.find_similar_properties_with_chunking(
                        query_text, property_listings, method, top_k, min_similarity
                    )
                    chunk_count = len([r for r in results if 'chunk_id' in r])
                
                end_time = time.time()
                
                comparison_results[method] = {
                    'results': results,
                    'execution_time': end_time - start_time,
                    'total_chunks': chunk_count,
                    'avg_similarity': np.mean([r.get('similarity_score', 0) for r in results]) if results else 0,
                    'max_similarity': max([r.get('similarity_score', 0) for r in results]) if results else 0,
                    'min_similarity': min([r.get('similarity_score', 0) for r in results]) if results else 0,
                    'result_count': len(results)
                }
                
            except Exception as e:
                comparison_results[method] = {
                    'error': str(e),
                    'execution_time': time.time() - start_time,
                    'results': []
                }
        
        return comparison_results
    
    def get_chunking_performance_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate a performance report for chunking method comparison.
        
        Args:
            comparison_results: Results from compare_chunking_methods
            
        Returns:
            Formatted performance report
        """
        report = "Chunking Method Performance Comparison\n"
        report += "=" * 50 + "\n\n"
        
        for method, results in comparison_results.items():
            report += f"Method: {method}\n"
            report += "-" * 20 + "\n"
            
            if 'error' in results:
                report += f"Error: {results['error']}\n"
            else:
                report += f"Execution Time: {results['execution_time']:.4f} seconds\n"
                report += f"Total Chunks: {results['total_chunks']}\n"
                report += f"Results Found: {results['result_count']}\n"
                report += f"Average Similarity: {results['avg_similarity']:.4f}\n"
                report += f"Max Similarity: {results['max_similarity']:.4f}\n"
                report += f"Min Similarity: {results['min_similarity']:.4f}\n"
            
            report += "\n"
        
        return report
    
    def find_similar_properties_hybrid(self, query_text: str, property_listings: List[Dict[str, Any]],
                                     chunking_method: str = "sentence", top_k: int = 5,
                                     min_similarity: float = 0.5, **chunking_kwargs) -> List[Dict[str, Any]]:
        """Find similar properties using a hybrid approach (both chunked and full descriptions).
        
        Args:
            query_text: Query description
            property_listings: List of property dictionaries
            chunking_method: Method to use for chunking
            top_k: Number of top results
            min_similarity: Minimum similarity threshold
            **chunking_kwargs: Parameters for chunking
            
        Returns:
            List of similar properties with hybrid similarity scores
        """
        # Get results from both approaches
        full_results = self.find_similar_properties(query_text, property_listings, top_k, min_similarity)
        chunked_results = self.find_similar_properties_with_chunking(
            query_text, property_listings, chunking_method, top_k, min_similarity, **chunking_kwargs
        )
        
        # Create a mapping of property IDs to their best scores
        property_scores = {}
        
        # Process full description results
        for result in full_results:
            prop_id = result.get('id', result.get('chunk_id', 'unknown'))
            if prop_id not in property_scores:
                property_scores[prop_id] = {
                    'property': result,
                    'full_score': result.get('similarity_score', 0),
                    'chunk_score': 0,
                    'best_score': result.get('similarity_score', 0)
                }
        
        # Process chunked results
        for result in chunked_results:
            # Extract original property ID from chunk ID
            chunk_id = result.get('chunk_id', '')
            prop_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            
            chunk_score = result.get('similarity_score', 0)
            
            if prop_id in property_scores:
                property_scores[prop_id]['chunk_score'] = max(
                    property_scores[prop_id]['chunk_score'], chunk_score
                )
                property_scores[prop_id]['best_score'] = max(
                    property_scores[prop_id]['best_score'], chunk_score
                )
            else:
                property_scores[prop_id] = {
                    'property': result,
                    'full_score': 0,
                    'chunk_score': chunk_score,
                    'best_score': chunk_score
                }
        
        # Create final results sorted by best score
        hybrid_results = []
        for prop_id, scores in property_scores.items():
            if scores['best_score'] >= min_similarity:
                result = scores['property'].copy()
                result['hybrid_similarity_score'] = scores['best_score']
                result['full_description_score'] = scores['full_score']
                result['chunked_score'] = scores['chunk_score']
                hybrid_results.append(result)
        
        # Sort by hybrid score and return top_k
        hybrid_results.sort(key=lambda x: x['hybrid_similarity_score'], reverse=True)
        return hybrid_results[:top_k] 