#!/usr/bin/env python3
"""Comprehensive property listing analysis using AI extraction methods"""

import sys
import os
import json
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ais_listing.extracting import (
    extract_from_text,
    extract_with_ensemble,
    PropertyEntity
)

def analyze_property_listing(listing_text: str) -> dict:
    """Analyze a property listing using multiple AI extraction methods."""
    
    # Use ensemble extraction for comprehensive results
    entities = extract_with_ensemble(
        listing_text, 
        methods=["transformer_ner", "qa_extraction", "text_classification"]
    )
    
    # Organize entities by type
    analysis = {
        'property_details': {},
        'location_info': [],
        'features': [],
        'amenities': [],
        'lifestyle_features': [],
        'technical_details': [],
        'summary': {}
    }
    
    # Process entities
    for entity in entities:
        if entity.entity_type == 'bedrooms':
            analysis['property_details']['bedrooms'] = entity.value
        elif entity.entity_type == 'bathrooms':
            analysis['property_details']['bathrooms'] = entity.value
        elif entity.entity_type == 'price':
            analysis['property_details']['price'] = entity.value
        elif entity.entity_type == 'location':
            analysis['location_info'].append({
                'name': entity.value,
                'confidence': float(entity.confidence),  # Convert to regular float
                'context': entity.context[:100]
            })
        elif entity.entity_type in ['luxury_features', 'modern_features', 'family_features']:
            analysis['features'].append({
                'type': entity.entity_type,
                'feature': entity.value,
                'confidence': float(entity.confidence)  # Convert to regular float
            })
        elif entity.entity_type in ['amenity', 'lifestyle_features']:
            analysis['amenities'].append({
                'amenity': entity.value,
                'confidence': float(entity.confidence)  # Convert to regular float
            })
        else:
            analysis['technical_details'].append({
                'type': entity.entity_type,
                'value': entity.value,
                'confidence': float(entity.confidence)  # Convert to regular float
            })
    
    # Generate summary statistics
    analysis['summary'] = {
        'total_entities': len(entities),
        'location_count': len(analysis['location_info']),
        'feature_count': len(analysis['features']),
        'amenity_count': len(analysis['amenities']),
        'high_confidence_entities': len([e for e in entities if e.confidence > 0.8])
    }
    
    return analysis

def print_analysis_report(analysis: dict, listing_text: str):
    """Print a formatted analysis report."""
    
    print("=" * 80)
    print("PROPERTY LISTING AI ANALYSIS REPORT")
    print("=" * 80)
    print(f"Text length: {len(listing_text)} characters")
    print(f"Total entities extracted: {analysis['summary']['total_entities']}")
    print(f"High confidence entities: {analysis['summary']['high_confidence_entities']}")
    print()
    
    # Property Details
    if analysis['property_details']:
        print("🏠 PROPERTY DETAILS")
        print("-" * 40)
        for key, value in analysis['property_details'].items():
            print(f"  {key.title()}: {value}")
        print()
    
    # Location Information
    if analysis['location_info']:
        print("📍 LOCATION INFORMATION")
        print("-" * 40)
        # Sort by confidence
        sorted_locations = sorted(analysis['location_info'], 
                                key=lambda x: x['confidence'], reverse=True)
        for loc in sorted_locations:
            print(f"  • {loc['name']} (confidence: {loc['confidence']:.3f})")
        print()
    
    # Features
    if analysis['features']:
        print("✨ PROPERTY FEATURES")
        print("-" * 40)
        # Group by feature type
        feature_groups = defaultdict(list)
        for feature in analysis['features']:
            feature_groups[feature['type']].append(feature)
        
        for feature_type, features in feature_groups.items():
            print(f"  {feature_type.replace('_', ' ').title()}:")
            for feature in features:
                print(f"    • {feature['feature']} (confidence: {feature['confidence']:.3f})")
        print()
    
    # Amenities
    if analysis['amenities']:
        print("🏪 AMENITIES & LIFESTYLE")
        print("-" * 40)
        for amenity in analysis['amenities']:
            print(f"  • {amenity['amenity']} (confidence: {amenity['confidence']:.3f})")
        print()
    
    # Technical Details
    if analysis['technical_details']:
        print("🔧 TECHNICAL DETAILS")
        print("-" * 40)
        for detail in analysis['technical_details']:
            print(f"  • {detail['type']}: {detail['value']} (confidence: {detail['confidence']:.3f})")
        print()

def main():
    # Read the listing text
    with open('data/listing_1.txt', 'r') as f:
        listing_text = f.read()
    
    # Analyze the listing
    analysis = analyze_property_listing(listing_text)
    
    # Print the report
    print_analysis_report(analysis, listing_text)
    
    # Save detailed results to JSON
    with open('extraction_results.json', 'w') as f:
        # Convert PropertyEntity objects to dictionaries for JSON serialization
        entities_data = []
        for entity in extract_with_ensemble(listing_text, methods=["transformer_ner", "qa_extraction", "text_classification"]):
            entities_data.append({
                'entity_type': entity.entity_type,
                'value': entity.value,
                'confidence': float(entity.confidence),  # Convert to regular float
                'start_pos': entity.start_pos,
                'end_pos': entity.end_pos,
                'context': entity.context
            })
        
        results = {
            'listing_text': listing_text,
            'analysis': analysis,
            'raw_entities': entities_data
        }
        json.dump(results, f, indent=2)
    
    print("📄 Detailed results saved to 'extraction_results.json'")

if __name__ == "__main__":
    main() 