#!/usr/bin/env python3

"""Test extraction methods on listing_1.txt"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ais_listing.extracting import (
    extract_from_text,
    extract_comprehensive_features,
    extract_with_ensemble,
    ExtractionFactory
)

def main():
    # Read the listing text
    with open('data/listing_1.txt', 'r') as f:
        listing_text = f.read()
    
    print("=" * 80)
    print("PROPERTY LISTING EXTRACTION ANALYSIS")
    print("=" * 80)
    print(f"Listing text length: {len(listing_text)} characters")
    print()
    
    # Test different extraction methods
    methods = [
        "semantic_features",
        "transformer_ner", 
        "qa_extraction",
        "spacy_ai",
        "text_classification",
        "clustering"
    ]
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"EXTRACTION METHOD: {method.upper()}")
        print(f"{'='*60}")
        
        try:
            entities = extract_from_text(listing_text, extraction_method=method)
            
            if entities:
                print(f"Found {len(entities)} entities:")
                for i, entity in enumerate(entities, 1):
                    print(f"  {i}. Type: {entity.entity_type}")
                    print(f"     Value: {entity.value}")
                    print(f"     Confidence: {entity.confidence:.3f}")
                    print(f"     Context: {entity.context[:100]}...")
                    print()
            else:
                print("No entities found.")
                
        except Exception as e:
            print(f"Error with method {method}: {e}")
    
    # Test comprehensive extraction
    print(f"\n{'='*60}")
    print("COMPREHENSIVE EXTRACTION (ALL METHODS)")
    print(f"{'='*60}")
    
    try:
        comprehensive_entities = extract_comprehensive_features(listing_text)
        print(f"Found {len(comprehensive_entities)} entities using comprehensive extraction:")
        
        # Group by entity type
        by_type = {}
        for entity in comprehensive_entities:
            if entity.entity_type not in by_type:
                by_type[entity.entity_type] = []
            by_type[entity.entity_type].append(entity)
        
        for entity_type, entities in by_type.items():
            print(f"\n{entity_type.upper()} ({len(entities)} entities):")
            for entity in entities:
                print(f"  - {entity.value} (confidence: {entity.confidence:.3f})")
                
    except Exception as e:
        print(f"Error with comprehensive extraction: {e}")
    
    # Test ensemble extraction
    print(f"\n{'='*60}")
    print("ENSEMBLE EXTRACTION")
    print(f"{'='*60}")
    
    try:
        ensemble_entities = extract_with_ensemble(listing_text, 
                                                methods=["semantic_features", "transformer_ner", "qa_extraction"])
        print(f"Found {len(ensemble_entities)} entities using ensemble extraction:")
        
        for i, entity in enumerate(ensemble_entities, 1):
            print(f"  {i}. {entity.entity_type}: {entity.value} ({entity.confidence:.3f})")
            
    except Exception as e:
        print(f"Error with ensemble extraction: {e}")

if __name__ == "__main__":
    main() 