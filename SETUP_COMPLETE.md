# AIS Listing - Setup Complete! 🎉

Your real estate listing analysis project has been successfully set up with `uv` and is ready to use!

## ✅ What's Been Set Up

### 1. **Project Structure**
```
ais-listing/
├── src/ais_listing/          # Main package
│   ├── __init__.py
│   ├── analyzer.py           # Main analysis logic
│   ├── embeddings.py         # Embedding model handling
│   ├── preprocessor.py       # Text preprocessing
│   ├── vector_store.py       # Vector database operations
│   └── cli.py               # Command-line interface
├── tests/                   # Test files
├── notebooks/              # Jupyter notebooks
├── data/                   # Sample data and models
│   └── sample_listings.csv # Sample real estate listings
├── pyproject.toml          # Project configuration
├── README.md               # Project documentation
└── .venv/                  # Virtual environment
```

### 2. **Dependencies Installed**
- **Core Data Processing**: pandas, numpy
- **NLP Libraries**: nltk, spacy, textblob, wordcloud
- **Embedding Models**: sentence-transformers, transformers, torch
- **Vector Databases**: faiss-cpu, chromadb
- **Visualization**: matplotlib, seaborn, plotly
- **Web Scraping**: requests, beautifulsoup4, selenium
- **Utilities**: python-dotenv, pydantic, click
- **Development**: jupyter, pytest, black, isort, mypy

### 3. **NLP Models Downloaded**
- spaCy English model (`en_core_web_sm`)
- NLTK data (punkt, stopwords, wordnet, punkt_tab)

## 🚀 How to Use

### 1. **Activate the Environment**
```bash
source .venv/bin/activate
```

### 2. **Command Line Interface**

#### Analyze a Single Listing
```bash
ais-listing analyze "Beautiful 3-bedroom home with modern kitchen and granite countertops"
```

#### Process Multiple Listings from CSV
```bash
ais-listing process -i data/sample_listings.csv -o results.json
```

#### Build Vector Database
```bash
ais-listing build-database -i data/sample_listings.csv
```

#### Search for Similar Listings
```bash
ais-listing similar "modern apartment with amenities" --k 3
```

#### Market Analysis
```bash
ais-listing market-analysis -i data/sample_listings.csv -o market_results.json
```

#### Cluster Listings
```bash
ais-listing cluster -i data/sample_listings.csv -o clusters.json --clusters 3
```

### 3. **Python API**

```python
from ais_listing import ListingAnalyzer

# Initialize analyzer
analyzer = ListingAnalyzer()

# Analyze a single listing
analysis = analyzer.analyze_single_listing(
    "Beautiful 3-bedroom home with modern kitchen and granite countertops"
)

# Process multiple listings
listings = [
    {"id": "1", "description": "Modern apartment...", "price": "$500k"},
    {"id": "2", "description": "Cozy house...", "price": "$400k"}
]
analyses = analyzer.analyze_multiple_listings(listings)

# Build vector database
stats = analyzer.build_vector_database(listings)

# Search for similar listings
similar = analyzer.search_similar_listings("modern apartment", k=5)
```

### 4. **Jupyter Notebooks**
Check out the `notebooks/` directory for interactive analysis examples.

## 🔧 Key Features

### **Text Preprocessing**
- Clean and normalize listing descriptions
- Extract real estate specific terms (bedrooms, bathrooms, amenities)
- Sentiment analysis
- Named entity recognition

### **Embedding Analysis**
- Generate embeddings using sentence transformers
- Calculate similarity between listings
- Find most similar properties
- Cluster listings by description similarity

### **Vector Database**
- Store embeddings in ChromaDB or FAISS
- Fast similarity search
- Metadata filtering
- Persistent storage

### **Market Analysis**
- Price statistics and trends
- Property feature analysis
- Location-based insights
- Market segmentation

## 📊 Sample Output

The system successfully analyzed the sample data:

```
✓ Text preprocessing works
  - Original: Beautiful 3-bedroom home with modern kitchen and granite countertops.
  - Cleaned: beautiful -bedroom home with modern kitchen and gr...
  - Word count: 10
  - Sentiment: 0.525

✓ Embedding generation works
  - Embedding dimension: 384
  - Model: sentence-transformers/all-MiniLM-L6-v2

✓ Main analyzer works
  - Listing ID: listing_1752975086.760955
  - Sentiment: positive
  - Quality: brief
  - Amenities: 3
```

## 🎯 Next Steps

1. **Add Your Data**: Replace `data/sample_listings.csv` with your real estate listings
2. **Customize Analysis**: Modify the preprocessing and analysis logic in the modules
3. **Scale Up**: Use larger embedding models or add more sophisticated analysis
4. **Deploy**: Package the application for production use
5. **Extend**: Add web scraping, API integrations, or visualization dashboards

## 🛠️ Development

### Run Tests
```bash
python test_setup.py
```

### Install Development Dependencies
```bash
uv pip install -e ".[dev]"
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
```

### Type Checking
```bash
mypy src/
```

## 📝 Configuration

Create a `.env` file based on `env.example` to customize:
- Embedding model selection
- Database paths
- Analysis parameters
- API keys (if needed)

---

**Your AIS Listing analysis system is ready to go! 🚀**

Start by analyzing your own real estate listings and exploring the various features available through the CLI and Python API. 