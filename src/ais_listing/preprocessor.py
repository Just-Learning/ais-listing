"""Text preprocessing module for real estate listings."""

import re
import string
from typing import List, Optional, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from textblob import TextBlob


class TextPreprocessor:
    """Preprocess real estate listing text for analysis."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize the preprocessor.
        
        Args:
            spacy_model: Name of the spaCy model to use
        """
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)
        
        # Download required NLTK data
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
            
        try:
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            nltk.download('wordnet')
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[1-9][\d]{0,15}', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens with stopwords removed
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract basic features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        cleaned_text = self.clean_text(text)
        doc = self.nlp(cleaned_text)
        
        # Basic statistics
        features = {
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents)),
            'avg_word_length': sum(len(token.text) for token in doc) / len(doc) if len(doc) > 0 else 0,
            'unique_words': len(set(token.text.lower() for token in doc)),
            'vocabulary_diversity': len(set(token.text.lower() for token in doc)) / len(doc) if len(doc) > 0 else 0
        }
        
        # Named entities
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        features['entities'] = entities
        
        # POS tags distribution
        pos_tags = {}
        for token in doc:
            if token.pos_ not in pos_tags:
                pos_tags[token.pos_] = 0
            pos_tags[token.pos_] += 1
        
        features['pos_tags'] = pos_tags
        
        return features
    
    def extract_real_estate_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract real estate specific terms and features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of real estate terms by category
        """
        cleaned_text = self.clean_text(text)
        doc = self.nlp(cleaned_text)
        
        # Define real estate related patterns
        patterns = {
            'bedrooms': r'\b(\d+)\s*(?:bed|bedroom|br)\b',
            'bathrooms': r'\b(\d+)\s*(?:bath|bathroom|ba)\b',
            'square_feet': r'\b(\d+)\s*(?:sq\s*ft|square\s*feet|sf)\b',
            'price': r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            'address': r'\b\d+\s+[a-zA-Z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd)\b',
        }
        
        extracted = {}
        for category, pattern in patterns.items():
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            if matches:
                extracted[category] = matches
        
        # Extract amenities and features
        amenities = [
            'kitchen', 'bathroom', 'bedroom', 'living room', 'dining room',
            'garage', 'parking', 'garden', 'balcony', 'terrace', 'pool',
            'fireplace', 'hardwood', 'carpet', 'granite', 'stainless steel',
            'central air', 'heating', 'cooling', 'washer', 'dryer', 'dishwasher'
        ]
        
        found_amenities = []
        for amenity in amenities:
            if amenity in cleaned_text:
                found_amenities.append(amenity)
        
        extracted['amenities'] = found_amenities
        
        return extracted
    
    def preprocess_for_embeddings(self, text: str, max_length: Optional[int] = None) -> str:
        """Preprocess text specifically for embedding models.
        
        Args:
            text: Input text
            max_length: Maximum length of processed text
            
        Returns:
            Preprocessed text ready for embeddings
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Rejoin
        processed = ' '.join(tokens)
        
        # Truncate if needed
        if max_length:
            words = processed.split()
            if len(words) > max_length:
                processed = ' '.join(words[:max_length])
        
        return processed
    
    def get_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment analysis of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def process_listing(self, text: str) -> Dict[str, Any]:
        """Complete preprocessing pipeline for a listing.
        
        Args:
            text: Raw listing text
            
        Returns:
            Dictionary with all processed data
        """
        result = {
            'original_text': text,
            'cleaned_text': self.clean_text(text),
            'features': self.extract_features(text),
            'real_estate_terms': self.extract_real_estate_terms(text),
            'sentiment': self.get_sentiment(text),
            'processed_for_embeddings': self.preprocess_for_embeddings(text)
        }
        
        return result 