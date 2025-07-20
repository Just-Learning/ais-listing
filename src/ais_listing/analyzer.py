"""Main analysis module for real estate listings."""

import os
import json
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .preprocessor import TextPreprocessor
from .embeddings import EmbeddingModel, RealEstateEmbeddingModel
from .vector_store import RealEstateVectorStore


class ListingAnalyzer:
    """Main analyzer for real estate listings."""
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 spacy_model: str = "en_core_web_sm",
                 vector_store_path: str = "./data/vector_store",
                 use_chroma: bool = True):
        """Initialize the listing analyzer.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            spacy_model: Name of the spaCy model to use
            vector_store_path: Path to store vector database
            use_chroma: Whether to use ChromaDB for vector storage
        """
        self.preprocessor = TextPreprocessor(spacy_model)
        self.embedding_model = RealEstateEmbeddingModel(embedding_model_name)
        self.vector_store = RealEstateVectorStore(vector_store_path, use_chroma)
        
        # Analysis cache
        self.analysis_cache = {}
    
    def analyze_single_listing(self, listing_text: str, listing_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a single listing.
        
        Args:
            listing_text: Raw listing text
            listing_id: Optional ID for the listing
            
        Returns:
            Complete analysis results
        """
        # Preprocess text
        preprocessed = self.preprocessor.process_listing(listing_text)
        
        # Generate embedding
        embedding = self.embedding_model.encode_single(preprocessed['processed_for_embeddings'])
        
        # Create analysis result
        analysis = {
            'listing_id': listing_id or f"listing_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'preprocessing': preprocessed,
            'embedding': embedding.tolist(),
            'embedding_dimension': len(embedding),
            'model_info': self.embedding_model.get_model_info()
        }
        
        # Add insights
        analysis['insights'] = self._generate_insights(preprocessed)
        
        return analysis
    
    def analyze_multiple_listings(self, listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze multiple listings.
        
        Args:
            listings: List of listing dictionaries with 'description' key
            
        Returns:
            List of analysis results
        """
        results = []
        
        for listing in listings:
            listing_id = listing.get('id', None)
            description = listing.get('description', '')
            
            analysis = self.analyze_single_listing(description, listing_id)
            
            # Add original listing data
            analysis['original_listing'] = listing
            
            results.append(analysis)
        
        return results
    
    def build_vector_database(self, listings: List[Dict[str, Any]], 
                            save_embeddings: bool = True) -> Dict[str, Any]:
        """Build a vector database from listings.
        
        Args:
            listings: List of listing dictionaries
            save_embeddings: Whether to save embeddings to disk
            
        Returns:
            Database statistics
        """
        # Extract descriptions
        descriptions = [listing.get('description', '') for listing in listings]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(descriptions)
        
        # Add to vector store
        self.vector_store.add_listings(listings, embeddings)
        
        # Save if requested
        if save_embeddings:
            self.vector_store.save()
        
        # Return statistics
        stats = self.vector_store.get_statistics()
        stats['embeddings_generated'] = len(embeddings)
        stats['embedding_dimension'] = embeddings.shape[1]
        
        return stats
    
    def search_similar_listings(self, query_text: str, k: int = 5, 
                               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar listings.
        
        Args:
            query_text: Query text
            k: Number of results
            filters: Optional filters for ChromaDB
            
        Returns:
            List of similar listings
        """
        return self.vector_store.search_similar_listings(
            query_text, self.embedding_model, k, filters
        )
    
    def get_listing_clusters(self, listings: List[Dict[str, Any]], 
                           n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster listings based on their descriptions.
        
        Args:
            listings: List of listing dictionaries
            n_clusters: Number of clusters
            
        Returns:
            Clustering results
        """
        descriptions = [listing.get('description', '') for listing in listings]
        
        # Preprocess descriptions
        processed_descriptions = [
            self.preprocessor.preprocess_for_embeddings(desc) 
            for desc in descriptions
        ]
        
        # Perform clustering
        clustering_result = self.embedding_model.cluster_embeddings(
            processed_descriptions, n_clusters
        )
        
        # Add listing data to clusters
        for cluster_id, cluster_items in clustering_result['clusters'].items():
            for item in cluster_items:
                item['listing'] = listings[item['index']]
        
        return clustering_result
    
    def analyze_market_trends(self, listings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market trends from listings.
        
        Args:
            listings: List of listing dictionaries
            
        Returns:
            Market analysis results
        """
        # Extract price data
        prices = []
        price_per_sqft = []
        bedrooms = []
        bathrooms = []
        locations = []
        
        for listing in listings:
            # Extract price
            price_str = str(listing.get('price', '0')).replace('$', '').replace(',', '')
            try:
                price = float(price_str)
                prices.append(price)
            except:
                continue
            
            # Extract square footage
            sqft_str = str(listing.get('square_feet', '0')).replace(',', '')
            try:
                sqft = float(sqft_str)
                if sqft > 0:
                    price_per_sqft.append(price / sqft)
            except:
                pass
            
            # Extract bedrooms and bathrooms
            try:
                bed = float(listing.get('bedrooms', 0))
                bedrooms.append(bed)
            except:
                pass
            
            try:
                bath = float(listing.get('bathrooms', 0))
                bathrooms.append(bath)
            except:
                pass
            
            # Extract location
            location = listing.get('location', '')
            if location:
                locations.append(location)
        
        # Calculate statistics
        analysis = {
            'total_listings': len(listings),
            'price_analysis': {
                'mean_price': np.mean(prices) if prices else 0,
                'median_price': np.median(prices) if prices else 0,
                'min_price': np.min(prices) if prices else 0,
                'max_price': np.max(prices) if prices else 0,
                'price_std': np.std(prices) if prices else 0
            },
            'price_per_sqft_analysis': {
                'mean_price_per_sqft': np.mean(price_per_sqft) if price_per_sqft else 0,
                'median_price_per_sqft': np.median(price_per_sqft) if price_per_sqft else 0,
                'min_price_per_sqft': np.min(price_per_sqft) if price_per_sqft else 0,
                'max_price_per_sqft': np.max(price_per_sqft) if price_per_sqft else 0
            },
            'property_features': {
                'mean_bedrooms': np.mean(bedrooms) if bedrooms else 0,
                'mean_bathrooms': np.mean(bathrooms) if bathrooms else 0,
                'most_common_bedrooms': int(np.median(bedrooms)) if bedrooms else 0,
                'most_common_bathrooms': int(np.median(bathrooms)) if bathrooms else 0
            }
        }
        
        # Location analysis
        if locations:
            from collections import Counter
            location_counts = Counter(locations)
            analysis['location_analysis'] = {
                'top_locations': location_counts.most_common(10),
                'unique_locations': len(set(locations))
            }
        
        return analysis
    
    def _generate_insights(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from preprocessed data.
        
        Args:
            preprocessed: Preprocessed listing data
            
        Returns:
            Dictionary of insights
        """
        insights = {}
        
        # Sentiment insights
        sentiment = preprocessed['sentiment']
        if sentiment['polarity'] > 0.1:
            insights['sentiment'] = 'positive'
        elif sentiment['polarity'] < -0.1:
            insights['sentiment'] = 'negative'
        else:
            insights['sentiment'] = 'neutral'
        
        insights['sentiment_confidence'] = abs(sentiment['polarity'])
        
        # Text quality insights
        features = preprocessed['features']
        if features['word_count'] < 50:
            insights['description_quality'] = 'brief'
        elif features['word_count'] < 200:
            insights['description_quality'] = 'moderate'
        else:
            insights['description_quality'] = 'detailed'
        
        # Real estate terms insights
        real_estate_terms = preprocessed['real_estate_terms']
        insights['extracted_features'] = {
            'bedrooms': real_estate_terms.get('bedrooms', []),
            'bathrooms': real_estate_terms.get('bathrooms', []),
            'amenities_count': len(real_estate_terms.get('amenities', [])),
            'has_price': bool(real_estate_terms.get('price', [])),
            'has_address': bool(real_estate_terms.get('address', []))
        }
        
        return insights
    
    def save_analysis(self, analysis: Dict[str, Any], filepath: str):
        """Save analysis results to file.
        
        Args:
            analysis: Analysis results
            filepath: Path to save file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def load_analysis(self, filepath: str) -> Dict[str, Any]:
        """Load analysis results from file.
        
        Args:
            filepath: Path to analysis file
            
        Returns:
            Analysis results
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def export_to_dataframe(self, analyses: List[Dict[str, Any]]) -> pd.DataFrame:
        """Export analysis results to pandas DataFrame.
        
        Args:
            analyses: List of analysis results
            
        Returns:
            DataFrame with analysis data
        """
        rows = []
        
        for analysis in analyses:
            row = {
                'listing_id': analysis['listing_id'],
                'timestamp': analysis['timestamp'],
                'word_count': analysis['preprocessing']['features']['word_count'],
                'sentiment_polarity': analysis['preprocessing']['sentiment']['polarity'],
                'sentiment_subjectivity': analysis['preprocessing']['sentiment']['subjectivity'],
                'embedding_dimension': analysis['embedding_dimension']
            }
            
            # Add insights
            insights = analysis['insights']
            row.update({
                'sentiment': insights['sentiment'],
                'description_quality': insights['description_quality'],
                'amenities_count': insights['extracted_features']['amenities_count'],
                'has_price': insights['extracted_features']['has_price']
            })
            
            # Add original listing data if available
            if 'original_listing' in analysis:
                original = analysis['original_listing']
                row.update({
                    'title': original.get('title', ''),
                    'price': original.get('price', ''),
                    'location': original.get('location', ''),
                    'bedrooms': original.get('bedrooms', ''),
                    'bathrooms': original.get('bathrooms', ''),
                    'property_type': original.get('property_type', '')
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows) 