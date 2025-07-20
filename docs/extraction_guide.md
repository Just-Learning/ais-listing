# Real Estate Data Extraction Guide

This guide covers the comprehensive real estate data extraction capabilities available in the `extracting.py` module, following the same coding format as `chunking.py`.

## Overview

The extraction module provides multiple methods to extract real estate entities from property listings, including addresses, property features, amenities, location information, and more.

## Available Extraction Methods

### 1. Regex Extraction (`regex`)

**Best for:** Structured data, addresses, numbers, measurements

**Capabilities:**
- **Addresses**: Street addresses with house numbers and street types
- **Bedrooms**: Number of bedrooms (e.g., "4 bedrooms", "4BR")
- **Bathrooms**: Number of bathrooms and ensuites
- **Price**: Monetary values with currency symbols
- **Square Feet**: Property size measurements
- **Year Built**: Construction dates
- **Schools**: Educational institutions
- **Amenities**: Common amenities like parks, stations, airports

**Example Usage:**
```python
from ais_listing.extracting import extract_from_text

entities = extract_from_text(
    listing_text, 
    extraction_method="regex",
    entity_types=["address", "bedrooms", "schools"]
)
```

### 2. SpaCy NER Extraction (`spacy`)

**Best for:** Named entities, locations, organizations, general NLP

**Capabilities:**
- **Locations**: Cities, suburbs, geographic areas (GPE, LOC labels)
- **Organizations**: Schools, companies, institutions (ORG label)
- **Numbers**: Cardinal numbers and quantities
- **Measurements**: Quantities and measurements
- **Money**: Monetary values
- **Dates**: Date expressions
- **Time**: Time expressions

**Example Usage:**
```python
entities = extract_from_text(
    listing_text, 
    extraction_method="spacy",
    entity_labels=["GPE", "LOC", "ORG", "CARDINAL"]
)
```

### 3. Keyword Extraction (`keyword`)

**Best for:** Property features, conditions, property types

**Capabilities:**
- **Property Types**: house, apartment, unit, townhouse, villa, etc.
- **Features**: hardwood floors, stone benchtops, stainless steel, etc.
- **Condition**: new, renovated, modern, contemporary, luxury, etc.
- **Location Types**: valley, suburb, neighborhood, area, etc.

**Example Usage:**
```python
entities = extract_from_text(
    listing_text, 
    extraction_method="keyword",
    entity_types=["features", "condition"]
)
```

### 4. Distance Extraction (`distance`)

**Best for:** Travel times, distances, proximity information

**Capabilities:**
- **Travel Time**: Minutes to destinations (e.g., "15 mins to Airport")
- **Distance**: Kilometer/mile measurements
- **Proximity**: Walkable distances (e.g., "short stroll away")

**Example Usage:**
```python
entities = extract_from_text(
    listing_text, 
    extraction_method="distance"
)
```

## Entity Types Extracted

### Property Information
- **Address**: Street addresses and property locations
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms and ensuites
- **Property Type**: Type of property (house, apartment, etc.)
- **Square Feet**: Property size measurements
- **Year Built**: Construction year

### Features & Amenities
- **Features**: Property features (hardwood floors, stone benchtops, etc.)
- **Amenities**: Nearby amenities (parks, stations, airports, etc.)
- **Condition**: Property condition (new, renovated, luxury, etc.)

### Location & Transportation
- **Location**: Geographic locations and areas
- **Schools**: Educational institutions
- **Travel Time**: Time to destinations
- **Distance**: Distance measurements
- **Proximity**: Walkable distances

### Organizations & Numbers
- **Organization**: Companies, schools, institutions
- **Number**: Cardinal numbers and quantities
- **Measurement**: Various measurements and quantities

## Usage Examples

### Basic Extraction
```python
from ais_listing.extracting import extract_from_text

# Extract all entities using regex
entities = extract_from_text(listing_text, extraction_method="regex")

# Extract specific entity types
entities = extract_from_text(
    listing_text, 
    extraction_method="regex",
    entity_types=["address", "bedrooms", "schools"]
)
```

### Multiple Methods Combined
```python
from ais_listing.extracting import (
    extract_from_text, 
    combine_extraction_results,
    ExtractionFactory
)

# Extract using multiple methods
methods = ExtractionFactory.get_available_methods()
all_entities = []

for method in methods:
    entities = extract_from_text(listing_text, extraction_method=method)
    all_entities.append(entities)

# Combine results
combined_entities = combine_extraction_results(all_entities)
```

### Property Listings Processing
```python
from ais_listing.extracting import extract_property_entities

# Process a list of property dictionaries
property_listings = [
    {"id": "1", "description": "Beautiful 3BR house..."},
    {"id": "2", "description": "Modern apartment..."}
]

extracted_properties = extract_property_entities(
    property_listings,
    extraction_method="regex"
)
```

## Entity Data Structure

Each extracted entity is returned as a `PropertyEntity` object with:

```python
@dataclass
class PropertyEntity:
    entity_type: str      # Type of entity (address, bedrooms, etc.)
    value: str           # Extracted value
    confidence: float    # Confidence score (0.0-1.0)
    start_pos: int       # Start position in text
    end_pos: int         # End position in text
    context: str         # Surrounding context
```

## Factory Pattern

The module uses a factory pattern similar to chunking.py:

```python
from ais_listing.extracting import ExtractionFactory

# Create an extractor
extractor = ExtractionFactory.create_extractor("regex")

# Get available methods
methods = ExtractionFactory.get_available_methods()
# Returns: ["regex", "spacy", "keyword", "distance"]
```

## Performance Considerations

- **Regex**: Fastest, good for structured data
- **SpaCy**: Medium speed, requires model download, best for general NLP
- **Keyword**: Fast, good for specific feature extraction
- **Distance**: Fast, specialized for travel/distance information

## Customization

### Adding Custom Patterns
```python
class CustomRegexExtraction(RegexExtraction):
    def __init__(self):
        super().__init__()
        # Add custom patterns
        self.patterns['custom_entity'] = [
            r'\b(custom_pattern)\b'
        ]
```

### Custom Entity Types
```python
class CustomKeywordExtraction(KeywordExtraction):
    def __init__(self):
        super().__init__()
        # Add custom keyword patterns
        self.keyword_patterns['custom_type'] = [
            'custom_keyword_1',
            'custom_keyword_2'
        ]
```

## Error Handling

The extraction methods include robust error handling:
- Graceful handling of missing spaCy models
- Fallback for failed extractions
- Duplicate entity removal in combined results
- Context validation for extracted entities

## Integration with Chunking

The extraction module works seamlessly with the chunking module:

```python
from ais_listing.chunking import chunk_property_descriptions
from ais_listing.extracting import extract_property_entities

# First chunk the descriptions
chunked_properties = chunk_property_descriptions(
    property_listings, 
    chunking_method="sentence"
)

# Then extract entities from chunks
extracted_chunks = extract_property_entities(
    chunked_properties,
    extraction_method="regex"
)
```

This allows for fine-grained entity extraction from smaller, more focused text chunks. 