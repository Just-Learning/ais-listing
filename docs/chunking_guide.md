# Chunking Methods for Real Estate Search

This guide explains how to use different text chunking methods to improve search performance for real estate listings.

## Overview

Text chunking breaks down long property descriptions into smaller, more manageable pieces. This can improve search accuracy by:

- Allowing more precise matching of specific features
- Reducing noise from irrelevant parts of descriptions
- Enabling better semantic understanding of property characteristics
- Improving search speed for large datasets

## Available Chunking Methods

### 1. Sentence Chunking
Breaks text into sentences while respecting natural language boundaries.

**Parameters:**
- `min_chunk_length`: Minimum characters per chunk (default: 10)
- `max_chunk_length`: Maximum characters per chunk (default: 500)

**Best for:** Maintaining semantic coherence while controlling chunk size.

### 2. Word Count Chunking
Breaks text into fixed-size word chunks with overlap.

**Parameters:**
- `chunk_size`: Number of words per chunk (default: 100)
- `overlap`: Number of overlapping words between chunks (default: 20)

**Best for:** Consistent chunk sizes and ensuring no information is lost.

### 3. Semantic Chunking
Groups sentences based on semantic similarity using embeddings.

**Parameters:**
- `similarity_threshold`: Threshold for combining similar sentences (default: 0.7)
- `embedding_model`: Pre-trained embedding model (required)

**Best for:** Creating semantically coherent chunks that group related information.

### 4. Real Estate Chunking
Specialized chunking for real estate descriptions with preprocessing.

**Parameters:**
- `chunk_method`: Base method to use ("sentence", "word_count", "semantic")
- Additional parameters for the chosen base method

**Best for:** Real estate-specific text normalization and optimization.

## Usage Examples

### Basic Usage

```python
from src.ais_listing.embeddings import RealEstateEmbeddingModel
from src.ais_listing.chunking import ChunkingFactory

# Initialize embedding model
embedding_model = RealEstateEmbeddingModel()

# Create sample listings
listings = [
    {
        'id': 'prop_001',
        'description': 'Beautiful 2-bedroom apartment with modern amenities...',
        'title': 'Modern Apartment',
        'price': '$450,000'
    }
]

# Search with different chunking methods
query = "modern apartment with amenities"

# No chunking (original method)
results_no_chunking = embedding_model.find_similar_properties(
    query, listings, top_k=5, min_similarity=0.5
)

# With sentence chunking
results_sentence = embedding_model.find_similar_properties_with_chunking(
    query, listings, chunking_method="sentence", top_k=5, min_similarity=0.5
)

# With word count chunking
results_word_count = embedding_model.find_similar_properties_with_chunking(
    query, listings, chunking_method="word_count", top_k=5, min_similarity=0.5,
    chunk_size=100, overlap=20
)
```

### Comparing Methods

```python
# Compare multiple chunking methods
methods = ['no_chunking', 'sentence', 'word_count']
comparison_results = embedding_model.compare_chunking_methods(
    query, listings, methods, top_k=5, min_similarity=0.5
)

# Generate performance report
report = embedding_model.get_chunking_performance_report(comparison_results)
print(report)
```

### Using the CLI

```bash
# Compare chunking methods
ais-listing compare-chunking "modern apartment with amenities" \
    --input data/listings.csv \
    --methods sentence word_count \
    --top-k 5 \
    --output results.json

# Search with specific chunking method
ais-listing search-chunked "modern apartment with amenities" \
    --input data/listings.csv \
    --method sentence \
    --top-k 5 \
    --output results.json
```

## Performance Comparison

### Metrics Tracked

1. **Execution Time**: How long each method takes to process
2. **Total Chunks**: Number of chunks created from original descriptions
3. **Results Found**: Number of properties returned above similarity threshold
4. **Average Similarity**: Mean similarity score of returned results
5. **Max/Min Similarity**: Highest and lowest similarity scores

### Example Performance Report

```
Chunking Method Performance Comparison
==================================================

Method: no_chunking
--------------------
Execution Time: 0.1234 seconds
Total Chunks: 100
Results Found: 5
Average Similarity: 0.723
Max Similarity: 0.891
Min Similarity: 0.612

Method: sentence
--------------------
Execution Time: 0.2345 seconds
Total Chunks: 250
Results Found: 8
Average Similarity: 0.756
Max Similarity: 0.923
Min Similarity: 0.634

Method: word_count
--------------------
Execution Time: 0.3456 seconds
Total Chunks: 300
Results Found: 6
Average Similarity: 0.712
Max Similarity: 0.878
Min Similarity: 0.598
```

## Choosing the Right Method

### When to Use Each Method

**No Chunking:**
- Small datasets (< 1000 listings)
- Short descriptions (< 200 words)
- When you want to preserve full context

**Sentence Chunking:**
- Medium to large datasets
- Descriptions with clear sentence structure
- When semantic coherence is important

**Word Count Chunking:**
- Large datasets with very long descriptions
- When you need consistent chunk sizes
- When you want to ensure no information is lost

**Semantic Chunking:**
- When you have access to embedding models
- When you want to group related information
- For high-quality semantic search



### Performance Considerations

1. **Speed vs. Accuracy Trade-off:**
   - No chunking: Fastest, but may miss specific features
   - Semantic chunking: Most accurate, but slowest
   - Sentence/Word count: Good balance

2. **Memory Usage:**
   - More chunks = more memory usage
   - Consider your hardware constraints

3. **Scalability:**
   - For very large datasets, consider word count chunking
   - For real-time search, sentence chunking often works best

## Advanced Features

### Hybrid Search

Combine results from multiple chunking methods:

```python
# Get hybrid results (best of both worlds)
hybrid_results = embedding_model.find_similar_properties_hybrid(
    query, listings, chunking_method="sentence", top_k=5, min_similarity=0.5
)

# Results include scores from both approaches
for result in hybrid_results:
    print(f"Hybrid Score: {result['hybrid_similarity_score']:.3f}")
    print(f"Full Description Score: {result['full_description_score']:.3f}")
    print(f"Chunked Score: {result['chunked_score']:.3f}")
```

### Custom Chunking Parameters

```python
# Custom sentence chunking
results = embedding_model.find_similar_properties_with_chunking(
    query, listings, 
    chunking_method="sentence",
    min_length=20,  # Minimum 20 characters per chunk
    max_length=300  # Maximum 300 characters per chunk
)

# Custom word count chunking
results = embedding_model.find_similar_properties_with_chunking(
    query, listings,
    chunking_method="word_count",
    chunk_size=50,   # 50 words per chunk
    overlap=10       # 10 word overlap
)
```

## Troubleshooting

### Common Issues

1. **No Results Returned:**
   - Lower the `min_similarity` threshold
   - Try different chunking methods
   - Check if descriptions are being chunked properly

2. **Slow Performance:**
   - Use simpler chunking methods for large datasets
   - Reduce chunk overlap
   - Consider using smaller chunk sizes

3. **Poor Search Quality:**
   - Try semantic chunking if available
   - Adjust chunking parameters
   - Use hybrid search approach

### Debugging

```python
# Check chunking results
from src.ais_listing.chunking import chunk_property_descriptions

chunked_properties = chunk_property_descriptions(
    listings, chunking_method="sentence"
)

for prop in chunked_properties[:3]:  # Show first 3 chunks
    print(f"Chunk ID: {prop['chunk_id']}")
    print(f"Chunk Index: {prop['chunk_index']}/{prop['total_chunks']}")
    print(f"Description: {prop['description'][:100]}...")
    print("---")
```

## Best Practices

1. **Start Simple:** Begin with sentence chunking for most use cases
2. **Test Multiple Methods:** Always compare performance across methods
3. **Monitor Performance:** Track execution time and result quality
4. **Tune Parameters:** Adjust chunk sizes and thresholds based on your data
5. **Consider Hybrid Approach:** Combine multiple methods for best results
6. **Cache Results:** Store chunked embeddings for repeated searches
7. **Validate Results:** Manually check search quality with sample queries

## Integration with Existing Code

The chunking functionality is designed to work seamlessly with existing code:

```python
# Existing code (no changes needed)
analyzer = ListingAnalyzer()
results = analyzer.search_similar_listings(query, k=5)

# New code with chunking
embedding_model = RealEstateEmbeddingModel()
results = embedding_model.find_similar_properties_with_chunking(
    query, listings, chunking_method="sentence", top_k=5
)
```

The chunking methods maintain the same interface as the original search methods, making it easy to integrate into existing workflows. 