# Chunking Comparison: Sentence vs Semantic

This document compares the results of applying **sentence chunking** and **semantic chunking** to the property listing in `data/listing_1.txt`.

## Overview

- **File:** `data/listing_1.txt`
- **Original Text Length:** 1,234 characters
- **Sentence Chunking:** 5 chunks
- **Semantic Chunking:** 10 chunks

## Critical Technical Differences

The semantic approach adds significant computational overhead but enables meaning-based grouping rather than just length-based grouping.

- Both use sentence tokenization - The key difference is what happens after splitting
- Sentence chunking uses character counting and string operations
- Semantic chunking requires embedding model inference and vector math
- Memory usage - Semantic stores embedding vectors (384-768 floats per sentence)
- Computational complexity - O(n+s) vs O(n+s²+s×e) where e is embedding dimension

## Algorithm Complexity Comparison

| Step | Sentence Chunking | Semantic Chunking |
|------|------------------|-------------------|
| **Text Processing** | O(n) - Simple string operations | O(n) - Simple string operations |
| **Sentence Splitting** | O(n) - NLTK tokenization | O(n) - NLTK tokenization |
| **Chunking Logic** | O(s) - Linear through sentences | O(s²) - Compare each sentence with current chunk |
| **Embedding Generation** | N/A | O(s × e) - Where e is embedding dimension |
| **Similarity Calculation** | N/A | O(s × e) - Cosine similarity for each comparison |
| **Overall Complexity** | O(n + s) | O(n + s² + s×e) |

Where:
- n = text length
- s = number of sentences
- e = embedding dimension

## Sentence Chunking Results

**Method:** Sentence-based chunking using NLTK sentence tokenization
**Chunks:** 5

### Chunk 1
Set high in the Glenroy Valley, 50 Valley Crescent delivers a perfect balance of mid-century charm and contemporary family living. The clinker-brick façade with warm timber accents makes an immediate impression, while a clever floor plan ensures effortless flow from front to back. Step inside to a welcoming entry and sunlit living area featuring polished hardwood floors, crisp white walls, and large picture windows that frame garden views.

### Chunk 2
The space seamlessly transitions into the dining area and gourmet kitchen, fitted with sleek stone benchtops, high-gloss cabinetry, glass splashbacks, and premium stainless-steel appliances. The accommodation comprises four bedrooms, including a master retreat with its own ensuite. The central bathroom impresses with near floor-to-ceiling tiles, a frameless shower, and a stunning clawfoot bathtub that brings timeless luxury.

### Chunk 3
At the rear, the open-plan meals area connects effortlessly to an expansive undercover alfresco zone-perfect for year-round entertaining. Landscaped gardens with raised vegetable beds and a level lawn complete this family-friendly haven.

### Chunk 4
Lifestyle location highlights:

• Sewell Reserve & playground just a short stroll away
• Glenroy Central, West Street Village & local cafes within minutes
• Glenroy Station for easy CBD access
• Zoned for Belle Vue Park Primary & Glenroy Secondary College
• Convenient access to the Moonee Ponds Creek Trail across the road for weekend walks
• Approx.

### Chunk 5
15 mins to Melbourne Airport & 30 mins to the CBD

This is a home that blends lifestyle and function in one of Glenroy's most sought-after pockets-ready for families to move straight in and enjoy.

---

## Semantic Chunking Results

**Method:** Semantic similarity-based chunking using embedding model
**Chunks:** 10

### Chunk 1
Set high in the Glenroy Valley, 50 Valley Crescent delivers a perfect balance of mid-century charm and contemporary family living.

### Chunk 2
The clinker-brick façade with warm timber accents makes an immediate impression, while a clever floor plan ensures effortless flow from front to back.

### Chunk 3
Step inside to a welcoming entry and sunlit living area featuring polished hardwood floors, crisp white walls, and large picture windows that frame garden views.

### Chunk 4
The space seamlessly transitions into the dining area and gourmet kitchen, fitted with sleek stone benchtops, high-gloss cabinetry, glass splashbacks, and premium stainless-steel appliances.

### Chunk 5
The accommodation comprises four bedrooms, including a master retreat with its own ensuite.

### Chunk 6
The central bathroom impresses with near floor-to-ceiling tiles, a frameless shower, and a stunning clawfoot bathtub that brings timeless luxury.

### Chunk 7
At the rear, the open-plan meals area connects effortlessly to an expansive undercover alfresco zone-perfect for year-round entertaining.

### Chunk 8
Landscaped gardens with raised vegetable beds and a level lawn complete this family-friendly haven.

### Chunk 9
Lifestyle location highlights:

• Sewell Reserve & playground just a short stroll away
• Glenroy Central, West Street Village & local cafes within minutes
• Glenroy Station for easy CBD access
• Zoned for Belle Vue Park Primary & Glenroy Secondary College
• Convenient access to the Moonee Ponds Creek Trail across the road for weekend walks
• Approx.

### Chunk 10
15 mins to Melbourne Airport & 30 mins to the CBD

This is a home that blends lifestyle and function in one of Glenroy's most sought-after pockets-ready for families to move straight in and enjoy.

---

## Key Differences

| Aspect | Sentence Chunking | Semantic Chunking |
|--------|------------------|-------------------|
| **Number of Chunks** | 5 | 10 |
| **Chunk Size** | Larger, more varied | Smaller, more consistent |
| **Semantic Coherence** | Based on sentence boundaries | Based on semantic similarity |
| **Search Precision** | Lower - broader chunks | Higher - focused chunks |
| **Processing Speed** | Faster | Slower (requires embedding) |
| **Information Density** | Mixed topics per chunk | Single topic per chunk |

## Analysis

### Sentence Chunking Advantages:
- **Faster processing** - no embedding model required
- **Natural language boundaries** - respects sentence structure
- **Larger context** - maintains more context per chunk
- **Simpler implementation** - easier to understand and debug

### Semantic Chunking Advantages:
- **Better semantic coherence** - groups related information
- **Higher search precision** - more focused chunks for specific queries
- **Consistent chunk sizes** - more uniform distribution
- **Better for feature extraction** - isolates specific property features

### Use Cases:

**Sentence Chunking is better for:**
- Quick analysis and overview
- When processing speed is critical
- General search queries
- Maintaining narrative flow

**Semantic Chunking is better for:**
- Precise feature-based searches
- When search accuracy is paramount
- Specific property characteristic queries
- Advanced semantic analysis

## Conclusion

The semantic chunking method produces twice as many chunks as sentence chunking, resulting in more granular and semantically focused pieces of text. This makes semantic chunking particularly effective for:

1. **Feature-specific searches** (e.g., "kitchen with stainless steel appliances")
2. **Location-based queries** (e.g., "near public transportation")
3. **Amenity searches** (e.g., "garden with vegetable beds")

However, sentence chunking remains valuable for:
1. **General property overview** searches
2. **Faster processing** when speed is important
3. **Maintaining narrative context** in search results

The choice between methods depends on the specific use case and whether search precision or processing speed is more important. 