"""Command-line interface for real estate listing analysis."""

import click
import json
import pandas as pd
from pathlib import Path
from typing import Optional

from .analyzer import ListingAnalyzer
from .embeddings import RealEstateEmbeddingModel
from .chunking import ChunkingFactory


@click.group()
@click.version_option()
def main():
    """AIS Listing - Real Estate Analysis Tool."""
    pass


@main.command()
@click.argument('text')
@click.option('--output', '-o', help='Output file for analysis results')
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'csv']), help='Output format')
def analyze(text, output, output_format):
    """Analyze a single listing description."""
    click.echo("Initializing analyzer...")
    analyzer = ListingAnalyzer()
    
    click.echo("Analyzing listing...")
    analysis = analyzer.analyze_single_listing(text)
    
    if output:
        if output_format == 'json':
            with open(output, 'w') as f:
                json.dump(analysis, f, indent=2)
        else:
            df = analyzer.export_to_dataframe([analysis])
            df.to_csv(output, index=False)
        click.echo(f"Analysis saved to {output}")
    else:
        click.echo(json.dumps(analysis, indent=2))


@main.command()
@click.option('--input', '-i', required=True, help='Input CSV file with listings')
@click.option('--output', '-o', required=True, help='Output file for analysis results')
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'csv']), help='Output format')
@click.option('--description-column', default='description', 
              help='Column name containing listing descriptions')
@click.option('--id-column', default='id', 
              help='Column name containing listing IDs')
def process(input, output, output_format, description_column, id_column):
    """Process multiple listings from a CSV file."""
    click.echo("Loading listings from CSV...")
    df = pd.read_csv(input)
    
    # Convert DataFrame to list of dictionaries
    listings = []
    for _, row in df.iterrows():
        listing = {
            'id': row.get(id_column, f"listing_{len(listings)}"),
            'description': row.get(description_column, ''),
            'title': row.get('title', ''),
            'price': row.get('price', ''),
            'location': row.get('location', ''),
            'bedrooms': row.get('bedrooms', ''),
            'bathrooms': row.get('bathrooms', ''),
            'property_type': row.get('property_type', ''),
            'square_feet': row.get('square_feet', '')
        }
        listings.append(listing)
    
    click.echo(f"Found {len(listings)} listings to analyze")
    
    click.echo("Initializing analyzer...")
    analyzer = ListingAnalyzer()
    
    click.echo("Analyzing listings...")
    analyses = analyzer.analyze_multiple_listings(listings)
    
    if output_format == 'json':
        with open(output, 'w') as f:
            json.dump(analyses, f, indent=2)
    else:
        df_output = analyzer.export_to_dataframe(analyses)
        df_output.to_csv(output, index=False)
    
    click.echo(f"Analysis of {len(analyses)} listings saved to {output}")


@main.command()
@click.option('--input', '-i', required=True, help='Input CSV file with listings')
@click.option('--output-dir', '-o', default='./data/vector_store', 
              help='Output directory for vector database')
@click.option('--description-column', default='description', 
              help='Column name containing listing descriptions')
def build_database(input, output_dir, description_column):
    """Build a vector database from listings."""
    click.echo("Loading listings from CSV...")
    df = pd.read_csv(input)
    
    # Convert DataFrame to list of dictionaries
    listings = []
    for _, row in df.iterrows():
        listing = {
            'id': row.get('id', f"listing_{len(listings)}"),
            'description': row.get(description_column, ''),
            'title': row.get('title', ''),
            'price': row.get('price', ''),
            'location': row.get('location', ''),
            'bedrooms': row.get('bedrooms', ''),
            'bathrooms': row.get('bathrooms', ''),
            'property_type': row.get('property_type', ''),
            'square_feet': row.get('square_feet', '')
        }
        listings.append(listing)
    
    click.echo(f"Found {len(listings)} listings to process")
    
    click.echo("Initializing analyzer...")
    analyzer = ListingAnalyzer(vector_store_path=output_dir)
    
    click.echo("Building vector database...")
    stats = analyzer.build_vector_database(listings)
    
    click.echo("Vector database statistics:")
    for key, value in stats.items():
        click.echo(f"  {key}: {value}")
    
    click.echo(f"Vector database saved to {output_dir}")


@main.command()
@click.argument('query')
@click.option('--database', '-d', default='./data/vector_store', 
              help='Path to vector database')
@click.option('--k', default=5, help='Number of results to return')
@click.option('--output', '-o', help='Output file for results')
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'csv']), help='Output format')
def similar(query, database, k, output, output_format):
    """Search for similar listings."""
    click.echo("Initializing analyzer...")
    analyzer = ListingAnalyzer(vector_store_path=database)
    
    click.echo(f"Searching for listings similar to: {query}")
    results = analyzer.search_similar_listings(query, k)
    
    if output:
        if output_format == 'json':
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            df = pd.DataFrame(results)
            df.to_csv(output, index=False)
        click.echo(f"Results saved to {output}")
    else:
        click.echo("Similar listings:")
        for i, result in enumerate(results, 1):
            click.echo(f"\n{i}. Score: {result.get('score', 0):.3f}")
            click.echo(f"   Title: {result.get('title', 'N/A')}")
            click.echo(f"   Price: {result.get('price', 'N/A')}")
            click.echo(f"   Location: {result.get('location', 'N/A')}")
            click.echo(f"   Bedrooms: {result.get('bedrooms', 'N/A')}")
            click.echo(f"   Bathrooms: {result.get('bathrooms', 'N/A')}")


@main.command()
@click.option('--input', '-i', required=True, help='Input CSV file with listings')
@click.option('--output', '-o', required=True, help='Output file for market analysis')
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'csv']), help='Output format')
@click.option('--description-column', default='description', 
              help='Column name containing listing descriptions')
def market_analysis(input, output, output_format, description_column):
    """Analyze market trends from listings."""
    click.echo("Loading listings from CSV...")
    df = pd.read_csv(input)
    
    # Convert DataFrame to list of dictionaries
    listings = []
    for _, row in df.iterrows():
        listing = {
            'id': row.get('id', f"listing_{len(listings)}"),
            'description': row.get(description_column, ''),
            'title': row.get('title', ''),
            'price': row.get('price', ''),
            'location': row.get('location', ''),
            'bedrooms': row.get('bedrooms', ''),
            'bathrooms': row.get('bathrooms', ''),
            'property_type': row.get('property_type', ''),
            'square_feet': row.get('square_feet', '')
        }
        listings.append(listing)
    
    click.echo(f"Found {len(listings)} listings to analyze")
    
    click.echo("Initializing analyzer...")
    analyzer = ListingAnalyzer()
    
    click.echo("Analyzing market trends...")
    analysis = analyzer.analyze_market_trends(listings)
    
    if output_format == 'json':
        with open(output, 'w') as f:
            json.dump(analysis, f, indent=2)
    else:
        # Convert market analysis to DataFrame format
        market_data = []
        for trend in analysis.get('trends', []):
            market_data.append({
                'trend': trend['trend'],
                'frequency': trend['frequency'],
                'sentiment': trend['sentiment']
            })
        df_output = pd.DataFrame(market_data)
        df_output.to_csv(output, index=False)
    
    click.echo(f"Market analysis saved to {output}")


@main.command()
@click.argument('query')
@click.option('--input', '-i', required=True, help='Input CSV file with listings')
@click.option('--output', '-o', help='Output file for comparison results')
@click.option('--methods', '-m', multiple=True, 
              type=click.Choice(['no_chunking', 'sentence', 'word_count', 'semantic']),
              default=['no_chunking', 'sentence', 'word_count'],
              help='Chunking methods to compare')
@click.option('--top-k', default=5, help='Number of top results to return')
@click.option('--min-similarity', default=0.5, help='Minimum similarity threshold')
@click.option('--description-column', default='description', 
              help='Column name containing listing descriptions')
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'text']), help='Output format')
def compare_chunking(query, input, output, methods, top_k, min_similarity, description_column, output_format):
    """Compare different chunking methods for search performance."""
    click.echo("Loading listings from CSV...")
    df = pd.read_csv(input)
    
    # Convert DataFrame to list of dictionaries
    listings = []
    for _, row in df.iterrows():
        listing = {
            'id': row.get('id', f"listing_{len(listings)}"),
            'description': row.get(description_column, ''),
            'title': row.get('title', ''),
            'price': row.get('price', ''),
            'location': row.get('location', ''),
            'bedrooms': row.get('bedrooms', ''),
            'bathrooms': row.get('bathrooms', ''),
            'property_type': row.get('property_type', ''),
            'square_feet': row.get('square_feet', '')
        }
        listings.append(listing)
    
    click.echo(f"Found {len(listings)} listings to analyze")
    click.echo(f"Query: {query}")
    click.echo(f"Methods to compare: {', '.join(methods)}")
    
    # Initialize embedding model
    click.echo("Initializing embedding model...")
    embedding_model = RealEstateEmbeddingModel()
    
    # Compare chunking methods
    click.echo("Comparing chunking methods...")
    comparison_results = embedding_model.compare_chunking_methods(
        query, listings, list(methods), top_k, min_similarity
    )
    
    # Generate performance report
    report = embedding_model.get_chunking_performance_report(comparison_results)
    
    if output:
        if output_format == 'json':
            # Save detailed results as JSON
            with open(output, 'w') as f:
                json.dump(comparison_results, f, indent=2)
        else:
            # Save performance report as text
            with open(output, 'w') as f:
                f.write(report)
        click.echo(f"Comparison results saved to {output}")
    else:
        # Display performance report
        click.echo("\n" + report)
        
        # Show detailed results for each method
        for method, results in comparison_results.items():
            if 'error' not in results and results['results']:
                click.echo(f"\nTop results for {method}:")
                for i, result in enumerate(results['results'][:3], 1):
                    click.echo(f"  {i}. Score: {result.get('similarity_score', 0):.3f}")
                    click.echo(f"     Description: {result.get('description', 'N/A')[:100]}...")
                    if 'chunk_id' in result:
                        click.echo(f"     Chunk: {result.get('chunk_id', 'N/A')}")


@main.command()
@click.argument('query')
@click.option('--input', '-i', required=True, help='Input CSV file with listings')
@click.option('--method', '-m', default='sentence', 
              type=click.Choice(['sentence', 'word_count', 'semantic']),
              help='Chunking method to use')
@click.option('--top-k', default=5, help='Number of top results to return')
@click.option('--min-similarity', default=0.5, help='Minimum similarity threshold')
@click.option('--description-column', default='description', 
              help='Column name containing listing descriptions')
@click.option('--output', '-o', help='Output file for results')
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'csv']), help='Output format')
def search_chunked(query, input, method, top_k, min_similarity, description_column, output, output_format):
    """Search for similar listings using chunked descriptions."""
    click.echo("Loading listings from CSV...")
    df = pd.read_csv(input)
    
    # Convert DataFrame to list of dictionaries
    listings = []
    for _, row in df.iterrows():
        listing = {
            'id': row.get('id', f"listing_{len(listings)}"),
            'description': row.get(description_column, ''),
            'title': row.get('title', ''),
            'price': row.get('price', ''),
            'location': row.get('location', ''),
            'bedrooms': row.get('bedrooms', ''),
            'bathrooms': row.get('bathrooms', ''),
            'property_type': row.get('property_type', ''),
            'square_feet': row.get('square_feet', '')
        }
        listings.append(listing)
    
    click.echo(f"Found {len(listings)} listings to analyze")
    click.echo(f"Query: {query}")
    click.echo(f"Chunking method: {method}")
    
    # Initialize embedding model
    click.echo("Initializing embedding model...")
    embedding_model = RealEstateEmbeddingModel()
    
    # Search using chunked descriptions
    click.echo("Searching with chunked descriptions...")
    results = embedding_model.find_similar_properties_with_chunking(
        query, listings, method, top_k, min_similarity
    )
    
    if output:
        if output_format == 'json':
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            df_output = pd.DataFrame(results)
            df_output.to_csv(output, index=False)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(f"\nTop {len(results)} results using {method} chunking:")
        for i, result in enumerate(results, 1):
            click.echo(f"\n{i}. Score: {result.get('similarity_score', 0):.3f}")
            click.echo(f"   ID: {result.get('id', 'N/A')}")
            click.echo(f"   Title: {result.get('title', 'N/A')}")
            click.echo(f"   Price: {result.get('price', 'N/A')}")
            click.echo(f"   Description: {result.get('description', 'N/A')[:150]}...")
            if 'chunk_id' in result:
                click.echo(f"   Chunk ID: {result.get('chunk_id', 'N/A')}")
                click.echo(f"   Chunk Index: {result.get('chunk_index', 'N/A')}/{result.get('total_chunks', 'N/A')}")


@main.command()
@click.option('--input', '-i', required=True, help='Input CSV file with listings')
@click.option('--output', '-o', required=True, help='Output file for clustering results')
@click.option('--clusters', '-c', default=5, help='Number of clusters')
@click.option('--description-column', default='description', 
              help='Column name containing listing descriptions')
def cluster(input, output, clusters, description_column):
    """Cluster listings based on their descriptions."""
    click.echo("Loading listings from CSV...")
    df = pd.read_csv(input)
    
    # Convert DataFrame to list of dictionaries
    listings = []
    for _, row in df.iterrows():
        listing = {
            'id': row.get('id', f"listing_{len(listings)}"),
            'description': row.get(description_column, ''),
            'title': row.get('title', ''),
            'price': row.get('price', ''),
            'location': row.get('location', ''),
            'bedrooms': row.get('bedrooms', ''),
            'bathrooms': row.get('bathrooms', ''),
            'property_type': row.get('property_type', ''),
            'square_feet': row.get('square_feet', '')
        }
        listings.append(listing)
    
    click.echo(f"Found {len(listings)} listings to cluster")
    
    click.echo("Initializing analyzer...")
    analyzer = ListingAnalyzer()
    
    click.echo(f"Clustering into {clusters} groups...")
    clustering_results = analyzer.cluster_listings(listings, clusters)
    
    with open(output, 'w') as f:
        json.dump(clustering_results, f, indent=2)
    
    click.echo(f"Clustering results saved to {output}")
    
    # Display cluster summary
    click.echo("\nCluster Summary:")
    for cluster_id, cluster_data in clustering_results['clusters'].items():
        click.echo(f"Cluster {cluster_id}: {len(cluster_data)} listings")


if __name__ == '__main__':
    main() 