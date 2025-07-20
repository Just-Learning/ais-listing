"""Real estate data extraction methods using AI/ML techniques."""

import re
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import json

# AI/ML imports
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering
)
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.tokens import Doc
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


@dataclass
class PropertyEntity:
    """Data class for extracted property entities."""
    entity_type: str
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""


class ExtractionMethod(ABC):
    """Abstract base class for extraction methods."""
    
    @abstractmethod
    def extract_entities(self, text: str, **kwargs) -> List[PropertyEntity]:
        """Extract entities from text.
        
        Args:
            text: Input text to extract from
            **kwargs: Additional parameters for extraction
            
        Returns:
            List of extracted PropertyEntity objects
        """
        pass
    
    def get_method_name(self) -> str:
        """Get the name of the extraction method."""
        return self.__class__.__name__


class TransformerNERExtraction(ExtractionMethod):
    """Extract entities using transformer-based Named Entity Recognition."""
    
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        """Initialize transformer NER extraction.
        
        Args:
            model_name: HuggingFace model name for NER
        """
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback to a smaller model
            self.model_name = "dslim/bert-base-NER"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
    
    def extract_entities(self, text: str, **kwargs) -> List[PropertyEntity]:
        """Extract entities using transformer NER.
        
        Args:
            text: Input text to extract from
            **kwargs: Additional parameters
            
        Returns:
            List of extracted PropertyEntity objects
        """
        entities = []
        
        # Use transformer NER pipeline
        ner_results = self.ner_pipeline(text)
        
        for result in ner_results:
            entity_type = self._map_ner_label(result['entity_group'])
            value = result['word']
            confidence = result['score']
            
            # Get context around the entity
            start_pos = result['start']
            end_pos = result['end']
            context_start = max(0, start_pos - 100)
            context_end = min(len(text), end_pos + 100)
            context = text[context_start:context_end]
            
            entity = PropertyEntity(
                entity_type=entity_type,
                value=value,
                confidence=confidence,
                start_pos=start_pos,
                end_pos=end_pos,
                context=context
            )
            entities.append(entity)
        
        return entities
    
    def _map_ner_label(self, label: str) -> str:
        """Map NER labels to property-specific entity types."""
        mapping = {
            'LOC': 'location',
            'ORG': 'organization',
            'PER': 'person',
            'MISC': 'miscellaneous',
            'O': 'other'
        }
        return mapping.get(label, label.lower())


class SemanticFeatureExtraction(ExtractionMethod):
    """Extract features using semantic similarity with sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize semantic feature extraction.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Define feature categories with example phrases
        self.feature_categories = {
            'transport_access': [
                "close to train station", "walking distance to bus stop", "near public transport",
                "easy access to transport", "transport hub nearby", "commuting convenience"
            ],
            'living_space': [
                "spacious living area", "open plan layout", "high ceilings", "large windows",
                "natural light", "entertaining space", "generous proportions"
            ],
            'neighborhood_quality': [
                "quiet neighborhood", "family friendly area", "safe location", "established community",
                "tree lined street", "peaceful area", "well maintained neighborhood"
            ],
            'outdoor_features': [
                "private garden", "landscaped garden", "balcony", "deck", "patio", "outdoor space",
                "bbq area", "garden shed"
            ],
            'lifestyle_amenities': [
                "cafe nearby", "restaurant close by", "shop walking distance", "park nearby",
                "school nearby", "gym close by", "hospital accessible"
            ],
            'modern_features': [
                "smart home", "solar panels", "energy efficient", "ducted air conditioning",
                "security system", "home automation", "integrated technology"
            ],
            'quality_finishes': [
                "stone benchtops", "stainless steel appliances", "hardwood floors", "premium finishes",
                "granite benchtops", "european appliances", "quality materials"
            ],
            'storage_solutions': [
                "built in wardrobes", "walk in wardrobe", "storage room", "linen press",
                "pantry", "ample storage", "garage parking"
            ]
        }
        
        # Pre-compute embeddings for feature categories
        self.category_embeddings = {}
        for category, phrases in self.feature_categories.items():
            embeddings = self.model.encode(phrases, convert_to_tensor=True)
            self.category_embeddings[category] = embeddings
    
    def extract_entities(self, text: str, **kwargs) -> List[PropertyEntity]:
        """Extract features using semantic similarity.
        
        Args:
            text: Input text to extract from
            **kwargs: Additional parameters (similarity_threshold, max_features_per_category)
            
        Returns:
            List of extracted PropertyEntity objects
        """
        similarity_threshold = kwargs.get('similarity_threshold', 0.7)
        max_features_per_category = kwargs.get('max_features_per_category', 3)
        
        entities = []
        
        # Split text into sentences for analysis
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            # Encode the sentence
            sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)
            
            # Check similarity with each feature category
            for category, category_phrases in self.feature_categories.items():
                category_embeddings = self.category_embeddings[category]
                
                # Calculate cosine similarities
                similarities = util.pytorch_cos_sim(sentence_embedding, category_embeddings)[0]
                
                # Find the best matches
                best_indices = torch.argsort(similarities, descending=True)[:max_features_per_category]
                
                for idx in best_indices:
                    similarity = similarities[idx].item()
                    
                    if similarity >= similarity_threshold:
                        # Find the position of the sentence in the original text
                        start_pos = text.find(sentence)
                        end_pos = start_pos + len(sentence)
                        
                        entity = PropertyEntity(
                            entity_type=category,
                            value=category_phrases[idx],
                            confidence=similarity,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            context=sentence
                        )
                        entities.append(entity)
        
        return entities


class QuestionAnsweringExtraction(ExtractionMethod):
    """Extract specific information using question-answering models."""
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        """Initialize QA extraction.
        
        Args:
            model_name: HuggingFace QA model name
        """
        self.model_name = model_name
        try:
            self.qa_pipeline = pipeline("question-answering", model=model_name)
        except Exception as e:
            print(f"Error loading QA model {model_name}: {e}")
            # Fallback to a smaller model
            self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        
        # Define questions for different property features
        self.feature_questions = {
            'bedrooms': [
                "How many bedrooms does this property have?",
                "What is the number of bedrooms?",
                "How many beds are there?"
            ],
            'bathrooms': [
                "How many bathrooms does this property have?",
                "What is the number of bathrooms?",
                "How many ensuites are there?"
            ],
            'price': [
                "What is the price of this property?",
                "How much does this property cost?",
                "What is the asking price?"
            ],
            'location': [
                "Where is this property located?",
                "What is the address?",
                "Which suburb is this in?"
            ],
            'property_type': [
                "What type of property is this?",
                "Is this a house, apartment, or unit?",
                "What kind of dwelling is this?"
            ],
            'features': [
                "What special features does this property have?",
                "What amenities are included?",
                "What makes this property unique?"
            ]
        }
    
    def extract_entities(self, text: str, **kwargs) -> List[PropertyEntity]:
        """Extract entities using question answering.
        
        Args:
            text: Input text to extract from
            **kwargs: Additional parameters (confidence_threshold)
            
        Returns:
            List of extracted PropertyEntity objects
        """
        confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        entities = []
        
        for feature_type, questions in self.feature_questions.items():
            for question in questions:
                try:
                    result = self.qa_pipeline(question=question, context=text)
                    
                    if result['score'] >= confidence_threshold:
                        answer = result['answer']
                        confidence = result['score']
                        
                        # Find the position of the answer in the text
                        start_pos = result['start']
                        end_pos = result['end']
                        
                        # Get context around the answer
                        context_start = max(0, start_pos - 100)
                        context_end = min(len(text), end_pos + 100)
                        context = text[context_start:context_end]
                        
                        entity = PropertyEntity(
                            entity_type=feature_type,
                            value=answer,
                            confidence=confidence,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            context=context
                        )
                        entities.append(entity)
                        
                except Exception as e:
                    print(f"Error processing question '{question}': {e}")
                    continue
        
        return entities


class TextClassificationExtraction(ExtractionMethod):
    """Extract features using text classification approach."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize text classification extraction.
        
        Args:
            model_name: Base model for classification
        """
        self.model_name = model_name
        
        # Define feature categories for classification
        self.feature_categories = {
            'luxury_features': [
                "premium finishes", "high end", "luxury", "exclusive", "prestigious",
                "designer", "custom", "bespoke", "architectural", "sophisticated"
            ],
            'family_features': [
                "family friendly", "child safe", "playground", "school nearby",
                "quiet street", "safe neighborhood", "backyard", "garden"
            ],
            'investment_features': [
                "rental potential", "investment opportunity", "high yield",
                "capital growth", "rental income", "tenant demand"
            ],
            'lifestyle_features': [
                "walking distance", "cafe culture", "restaurant precinct",
                "entertainment", "nightlife", "shopping", "recreation"
            ],
            'modern_features': [
                "smart home", "automation", "energy efficient", "solar",
                "modern design", "contemporary", "new build"
            ]
        }
        
        # Initialize TF-IDF vectorizer for feature detection
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=1000,
            stop_words='english'
        )
        
        # Prepare training data for classification
        self._prepare_classification_data()
    
    def _prepare_classification_data(self):
        """Prepare training data for feature classification."""
        self.training_texts = []
        self.training_labels = []
        
        for category, keywords in self.feature_categories.items():
            for keyword in keywords:
                self.training_texts.append(keyword)
                self.training_labels.append(category)
        
        # Fit the vectorizer
        self.vectorizer.fit(self.training_texts)
    
    def extract_entities(self, text: str, **kwargs) -> List[PropertyEntity]:
        """Extract entities using text classification.
        
        Args:
            text: Input text to extract from
            **kwargs: Additional parameters (similarity_threshold)
            
        Returns:
            List of extracted PropertyEntity objects
        """
        similarity_threshold = kwargs.get('similarity_threshold', 0.3)
        entities = []
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            
            # Vectorize the sentence
            sentence_vector = self.vectorizer.transform([sentence])
            
            # Calculate similarity with training examples
            similarities = cosine_similarity(sentence_vector, self.vectorizer.transform(self.training_texts))[0]
            
            # Find the best matching category
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= similarity_threshold:
                category = self.training_labels[best_idx]
                keyword = self.training_texts[best_idx]
                
                # Find position in original text
                start_pos = text.find(sentence)
                end_pos = start_pos + len(sentence)
                
                entity = PropertyEntity(
                    entity_type=category,
                    value=keyword,
                    confidence=best_similarity,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    context=sentence
                )
                entities.append(entity)
        
        return entities


class ClusteringExtraction(ExtractionMethod):
    """Extract features using clustering of similar phrases."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize clustering extraction.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def extract_entities(self, text: str, **kwargs) -> List[PropertyEntity]:
        """Extract entities using clustering approach.
        
        Args:
            text: Input text to extract from
            **kwargs: Additional parameters (eps, min_samples)
            
        Returns:
            List of extracted PropertyEntity objects
        """
        eps = kwargs.get('eps', 0.3)
        min_samples = kwargs.get('min_samples', 2)
        
        entities = []
        
        # Split text into phrases (sentences and noun phrases)
        phrases = self._extract_phrases(text)
        
        if len(phrases) < 2:
            return entities
        
        # Encode phrases
        embeddings = self.model.encode(phrases, convert_to_tensor=True)
        
        # Convert to numpy for clustering
        embeddings_np = embeddings.cpu().numpy()
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings_np)
        
        # Group phrases by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Skip noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(phrases[i])
        
        # Create entities for each cluster
        for cluster_id, cluster_phrases in clusters.items():
            if len(cluster_phrases) >= min_samples:
                # Use the most representative phrase (longest)
                representative_phrase = max(cluster_phrases, key=len)
                
                # Find position in original text
                start_pos = text.find(representative_phrase)
                end_pos = start_pos + len(representative_phrase)
                
                # Calculate confidence based on cluster size
                confidence = min(0.9, 0.5 + len(cluster_phrases) * 0.1)
                
                entity = PropertyEntity(
                    entity_type='clustered_feature',
                    value=representative_phrase,
                    confidence=confidence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    context=f"Cluster: {', '.join(cluster_phrases)}"
                )
                entities.append(entity)
        
        return entities
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text."""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Add noun phrases using spaCy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) >= 2]
        except:
            noun_phrases = []
        
        # Combine and filter phrases
        all_phrases = sentences + noun_phrases
        filtered_phrases = [phrase.strip() for phrase in all_phrases if len(phrase.strip()) > 10]
        
        return filtered_phrases


class ExtractionFactory:
    """Factory for creating AI/ML extraction methods."""
    
    @staticmethod
    def create_extractor(method: str, **kwargs) -> ExtractionMethod:
        """Create an extraction method instance.
        
        Args:
            method: Extraction method name
            **kwargs: Parameters for the extraction method
            
        Returns:
            ExtractionMethod instance
        """
        if method == "transformer_ner":
            return TransformerNERExtraction(**kwargs)
        elif method == "semantic_features":
            return SemanticFeatureExtraction(**kwargs)
        elif method == "qa_extraction":
            return QuestionAnsweringExtraction(**kwargs)
        elif method == "text_classification":
            return TextClassificationExtraction(**kwargs)
        elif method == "clustering":
            return ClusteringExtraction(**kwargs)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available AI/ML extraction methods."""
        return [
            "transformer_ner", 
            "semantic_features", 
            "qa_extraction", 
            "text_classification", 
            "spacy_ai", 
            "clustering"
        ]


def extract_property_entities(property_listings: List[Dict[str, Any]], 
                            extraction_method: str = "semantic_features",
                            **kwargs) -> List[Dict[str, Any]]:
    """Extract entities from property listings using AI/ML methods.
    
    Args:
        property_listings: List of property dictionaries
        extraction_method: AI/ML method to use for extraction
        **kwargs: Parameters for the extraction method
        
    Returns:
        List of property dictionaries with extracted entities
    """
    extractor = ExtractionFactory.create_extractor(extraction_method, **kwargs)
    
    extracted_properties = []
    
    for prop in property_listings:
        description = prop.get('description', '')
        if not description:
            continue
        
        entities = extractor.extract_entities(description, **kwargs)
        
        # Convert entities to dictionaries for JSON serialization
        entity_dicts = []
        for entity in entities:
            entity_dict = {
                'entity_type': entity.entity_type,
                'value': entity.value,
                'confidence': entity.confidence,
                'start_pos': entity.start_pos,
                'end_pos': entity.end_pos,
                'context': entity.context
            }
            entity_dicts.append(entity_dict)
        
        # Add extracted entities to property
        prop_with_entities = prop.copy()
        prop_with_entities['extracted_entities'] = entity_dicts
        prop_with_entities['extraction_method'] = extraction_method
        
        extracted_properties.append(prop_with_entities)
    
    return extracted_properties


def extract_from_text(text: str, 
                     extraction_method: str = "semantic_features",
                     **kwargs) -> List[PropertyEntity]:
    """Extract entities from a single text string using AI/ML.
    
    Args:
        text: Input text to extract from
        extraction_method: AI/ML method to use for extraction
        **kwargs: Parameters for the extraction method
        
    Returns:
        List of extracted PropertyEntity objects
    """
    extractor = ExtractionFactory.create_extractor(extraction_method, **kwargs)
    return extractor.extract_entities(text, **kwargs)


def combine_extraction_results(entities_list: List[List[PropertyEntity]]) -> List[PropertyEntity]:
    """Combine results from multiple AI/ML extraction methods.
    
    Args:
        entities_list: List of entity lists from different extraction methods
        
    Returns:
        Combined list of unique entities
    """
    all_entities = []
    seen_entities = set()
    
    for entities in entities_list:
        for entity in entities:
            # Create a unique identifier for the entity
            entity_id = f"{entity.entity_type}:{entity.value}:{entity.start_pos}:{entity.end_pos}"
            
            if entity_id not in seen_entities:
                all_entities.append(entity)
                seen_entities.add(entity_id)
    
    return all_entities


def extract_comprehensive_features(text: str) -> List[PropertyEntity]:
    """Extract comprehensive property features using multiple AI/ML methods.
    
    Args:
        text: Input text to extract from
        
    Returns:
        List of extracted PropertyEntity objects from all methods
    """
    # Use multiple AI/ML extraction methods to get comprehensive features
    methods = ["semantic_features", "transformer_ner", "qa_extraction", "spacy_ai"]
    
    all_entities = []
    for method in methods:
        try:
            extractor = ExtractionFactory.create_extractor(method)
            entities = extractor.extract_entities(text)
            all_entities.extend(entities)
        except Exception as e:
            print(f"Error with method {method}: {e}")
            continue
    
    # Remove duplicates and return
    return combine_extraction_results([all_entities])


def extract_with_ensemble(text: str, methods: List[str] = None, **kwargs) -> List[PropertyEntity]:
    """Extract entities using an ensemble of AI/ML methods.
    
    Args:
        text: Input text to extract from
        methods: List of extraction methods to use
        **kwargs: Parameters for extraction methods
        
    Returns:
        List of extracted PropertyEntity objects
    """
    if methods is None:
        methods = ["semantic_features", "transformer_ner", "qa_extraction"]
    
    all_entities = []
    
    for method in methods:
        try:
            extractor = ExtractionFactory.create_extractor(method)
            entities = extractor.extract_entities(text, **kwargs)
            all_entities.append(entities)
        except Exception as e:
            print(f"Error with method {method}: {e}")
            continue
    
    # Combine results from all methods
    return combine_extraction_results(all_entities)
