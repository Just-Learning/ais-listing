# AIS Listing - Real Estate Analysis

A Python-based tool for analyzing real estate listing descriptions using embedding models and NLP techniques.

## Features

- **Text Processing**: Clean and preprocess real estate listing descriptions
- **Embedding Analysis**: Generate and analyze embeddings for similarity search
- **Chunking**: Split text into smaller pieces for analysis [report](chunking_comparison.md)
- **NLP Features**: Extract key information, sentiment, and topics from listings
- **Vector Search**: Find similar properties using FAISS and ChromaDB
- **Visualization**: Create insights and visualizations from listing data

## Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ais-listing
```

2. Create and activate virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. Install development dependencies (optional):
```bash
uv pip install -e ".[dev]"
```

5. Download required NLP models:
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

### Basic Analysis

```python
from ais_listing.analyzer import ListingAnalyzer
from ais_listing.embeddings import EmbeddingModel

# Initialize analyzer
analyzer = ListingAnalyzer()
embedding_model = EmbeddingModel()

# Analyze a listing
listing_text = "Beautiful 3-bedroom house with modern kitchen and large backyard..."
analysis = analyzer.analyze(listing_text)
embeddings = embedding_model.encode(listing_text)

print(analysis)
```

### CLI Usage

```bash
# Analyze a single listing
ais-listing analyze "Beautiful 3-bedroom house..."

# Process a CSV file of listings
ais-listing process --input listings.csv --output analysis.json

# Find similar properties
ais-listing similar --query "modern apartment" --database listings.db
```

## Project Structure

```
ais-listing/
├── src/
│   └── ais_listing/
│       ├── __init__.py
│       ├── analyzer.py      # Main analysis logic
│       ├── embeddings.py    # Embedding model handling
│       ├── preprocessor.py  # Text preprocessing
│       ├── vector_store.py  # Vector database operations
│       ├── visualizer.py    # Data visualization
│       └── cli.py          # Command-line interface
├── tests/                  # Test files
├── notebooks/             # Jupyter notebooks for analysis
├── data/                  # Sample data and models
├── pyproject.toml         # Project configuration
└── README.md
```

## Configuration

Create a `.env` file in the project root:

```env
# Model settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SPACY_MODEL=en_core_web_sm

# Database settings
VECTOR_DB_PATH=./data/vector_db
CHROMA_DB_PATH=./data/chroma_db

# Analysis settings
MAX_TEXT_LENGTH=512
SIMILARITY_THRESHOLD=0.7
```

## Development

### Running Tests

```bash
pytest
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
