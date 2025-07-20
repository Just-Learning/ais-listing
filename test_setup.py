#!/usr/bin/env python3
"""Test script to verify the AIS Listing setup."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from ais_listing import ListingAnalyzer, TextPreprocessor, EmbeddingModel
        print("✓ Core modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import core modules: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✓ Data processing libraries imported successfully")
    except Exception as e:
        print(f"✗ Failed to import data processing libraries: {e}")
        return False
    
    try:
        import spacy
        print("✓ spaCy imported successfully")
    except Exception as e:
        print(f"✗ Failed to import spaCy: {e}")
        return False
    
    try:
        import nltk
        print("✓ NLTK imported successfully")
    except Exception as e:
        print(f"✗ Failed to import NLTK: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence Transformers imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Sentence Transformers: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from ais_listing import TextPreprocessor
        
        # Test text preprocessing
        preprocessor = TextPreprocessor()
        test_text = "Beautiful 3-bedroom home with modern kitchen and granite countertops."
        processed = preprocessor.process_listing(test_text)
        
        print(f"✓ Text preprocessing works")
        print(f"  - Original: {test_text}")
        print(f"  - Cleaned: {processed['cleaned_text'][:50]}...")
        print(f"  - Word count: {processed['features']['word_count']}")
        print(f"  - Sentiment: {processed['sentiment']['polarity']:.3f}")
        
    except Exception as e:
        print(f"✗ Text preprocessing failed: {e}")
        return False
    
    try:
        from ais_listing import EmbeddingModel
        
        # Test embedding generation
        embedding_model = EmbeddingModel()
        test_text = "Modern apartment with amenities"
        embedding = embedding_model.encode_single(test_text)
        
        print(f"✓ Embedding generation works")
        print(f"  - Embedding dimension: {len(embedding)}")
        print(f"  - Model: {embedding_model.model_name}")
        
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False
    
    return True

def test_analyzer():
    """Test the main analyzer."""
    print("\nTesting main analyzer...")
    
    try:
        from ais_listing import ListingAnalyzer
        
        analyzer = ListingAnalyzer()
        test_text = "Stunning 2-bedroom condo with ocean views, updated kitchen, and private balcony."
        
        analysis = analyzer.analyze_single_listing(test_text)
        
        print(f"✓ Main analyzer works")
        print(f"  - Listing ID: {analysis['listing_id']}")
        print(f"  - Sentiment: {analysis['insights']['sentiment']}")
        print(f"  - Quality: {analysis['insights']['description_quality']}")
        print(f"  - Amenities: {analysis['insights']['extracted_features']['amenities_count']}")
        
    except Exception as e:
        print(f"✗ Main analyzer failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("AIS Listing Setup Test")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test analyzer
    if not test_analyzer():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Setup is working correctly.")
        print("\nNext steps:")
        print("1. Activate the virtual environment: source .venv/bin/activate")
        print("2. Try the CLI: ais-listing analyze 'Your listing text here'")
        print("3. Check out the notebooks in the notebooks/ directory")
        print("4. Use the sample data: ais-listing process -i data/sample_listings.csv -o results.json")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 