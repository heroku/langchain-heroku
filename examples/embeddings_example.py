#!/usr/bin/env python3
# ruff: noqa: T201
"""
Example script demonstrating Heroku Embeddings integration with LangChain.

This script shows how to use the HerokuEmbeddings class for various use cases,
including OpenAI compatibility mode and Heroku advanced features.
"""

import os
import sys
from typing import Dict, Optional  # noqa: F401

# Add the parent directory to the path to import langchain_heroku
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_heroku.embeddings import HerokuEmbeddings


def setup_environment() -> Optional[Dict[str, Optional[str]]]:
    """Set up environment variables and return configuration."""
    config: Dict[str, Optional[str]] = {}

    # Check for required environment variables
    config["inference_url"] = os.getenv("INFERENCE_URL")
    config["api_key"] = os.getenv("INFERENCE_KEY") or os.getenv("INFERENCE_EMBED_KEY")
    config["model"] = os.getenv("INFERENCE_MODEL_ID")

    if not all(config.values()):
        print("❌ Missing required environment variables!")
        print("Please set the following environment variables:")
        print("  - INFERENCE_URL: Your Heroku inference API URL")
        print("  - INFERENCE_KEY or INFERENCE_EMBED_KEY: Your Heroku API key")
        print("  - INFERENCE_MODEL_ID: Your model ID")
        print("\nExample:")
        print("  export INFERENCE_URL='https://your-url.com'")
        print("  export INFERENCE_KEY='your-key'")
        print("  export INFERENCE_EMBED_KEY='your-key'  # Alternative")
        print("  export INFERENCE_MODEL_ID='your-model'")
        return None

    print("✅ Environment configuration complete!")
    return config


def basic_embedding_demo(embeddings: HerokuEmbeddings) -> None:
    """Demonstrate basic embedding functionality."""
    print("\n🔍 Basic Embedding Demo")
    print("=" * 50)

    # Single query embedding
    query = "What is artificial intelligence?"
    print(f"Query: {query}")

    try:
        embedding = embeddings.embed_query(query)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return

    # Batch document embedding
    documents = [
        "Artificial intelligence is the simulation of human intelligence in machines.",
        "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Natural language processing helps computers understand and generate human language.",
    ]

    print(f"\n📚 Processing {len(documents)} documents...")

    try:
        document_embeddings = embeddings.embed_documents(documents)
        print(f"✅ Generated {len(document_embeddings)} document embeddings")

        for i, (doc, emb) in enumerate(zip(documents, document_embeddings)):
            print(f"  Document {i+1}: {len(emb)} dimensions")

    except Exception as e:
        print(f"❌ Error generating document embeddings: {e}")


def advanced_features_demo(embeddings: HerokuEmbeddings) -> None:
    """Demonstrate Heroku's advanced features."""
    print("\n🚀 Advanced Features Demo")
    print("=" * 50)

    # Test different input types
    input_types = ["search_document", "search_query", "classification", "clustering"]

    for input_type in input_types:
        print(f"\n🔧 Testing input_type: {input_type}")

        try:
            # Create embeddings instance with specific input type
            advanced_embeddings = HerokuEmbeddings(
                model=embeddings.model, api_key=embeddings.api_key, inference_url=embeddings.inference_url, input_type=input_type
            )

            # Test the input type
            test_text = f"Test text for {input_type} optimization"
            embedding = advanced_embeddings.embed_query(test_text)
            print(f"  ✅ Success: {len(embedding)} dimensions")

        except Exception as e:
            print(f"  ❌ Not supported: {e}")

    # Test different encoding formats
    print("\n🔧 Testing encoding formats...")

    try:
        base64_embeddings = HerokuEmbeddings(
            model=embeddings.model, api_key=embeddings.api_key, inference_url=embeddings.inference_url, encoding_format="base64"
        )

        test_text = "Test text for base64 encoding"
        embedding = base64_embeddings.embed_query(test_text)
        print(f"  ✅ Base64 encoding: {len(embedding)} dimensions")

    except Exception as e:
        print(f"  ❌ Base64 encoding not supported: {e}")


def metadata_demo(embeddings: HerokuEmbeddings) -> None:
    """Demonstrate metadata retrieval capabilities."""
    print("\n📊 Metadata Demo")
    print("=" * 50)

    query = "Retrieve metadata for this query"

    try:
        # Single query with metadata
        print("🔍 Single query with metadata:")
        result = embeddings.embed_query_with_metadata(query)

        print(f"  Embedding dimensions: {len(result['embedding'])}")
        print(f"  Model: {result.get('model', 'N/A')}")

        if "usage" in result:
            usage = result["usage"]
            print(f"  Usage: {usage}")

        print(f"  Response metadata keys: {list(result.get('response_metadata', {}).keys())}")

    except Exception as e:
        print(f"❌ Error retrieving metadata: {e}")

    # Batch with metadata
    documents = ["Document A", "Document B"]

    try:
        print(f"\n📚 Batch processing with metadata ({len(documents)} documents):")
        results = embeddings.embed_documents_with_metadata(documents)

        for i, result in enumerate(results):
            print(f"  Document {i+1}:")
            print(f"    Text: {result['text']}")
            print(f"    Dimensions: {len(result['embedding'])}")
            print(f"    Model: {result.get('model', 'N/A')}")

    except Exception as e:
        print(f"❌ Error in batch metadata: {e}")


def similarity_demo(embeddings: HerokuEmbeddings) -> None:
    """Demonstrate basic similarity concepts."""
    print("\n🔗 Similarity Demo")
    print("=" * 50)

    # Similar texts (should produce similar embeddings)
    similar_texts = ["The cat is on the mat", "A cat sits on the mat", "The mat has a cat on it"]

    # Different texts (should produce different embeddings)
    different_texts = ["The cat is on the mat", "The weather is sunny today", "Mathematics is the language of science"]

    try:
        print("🐱 Processing similar texts...")
        similar_embeddings = embeddings.embed_documents(similar_texts)

        print("🌤️ Processing different texts...")
        different_embeddings = embeddings.embed_documents(different_texts)

        print("✅ Generated embeddings:")
        print(f"  Similar texts: {len(similar_embeddings)} embeddings, {len(similar_embeddings[0])} dimensions each")
        print(f"  Different texts: {len(different_embeddings)} embeddings, {len(different_embeddings[0])} dimensions each")

        # Note: In a real application, you would calculate cosine similarity or other metrics
        print("\n💡 Note: In production, use cosine similarity or other metrics to compare embeddings")

    except Exception as e:
        print(f"❌ Error in similarity demo: {e}")


def performance_demo(embeddings: HerokuEmbeddings) -> None:
    """Demonstrate performance characteristics."""
    print("\n⚡ Performance Demo")
    print("=" * 50)

    # Test different batch sizes
    batch_sizes = [1, 5, 10, 20]

    for batch_size in batch_sizes:
        print(f"\n📦 Testing batch size: {batch_size}")

        try:
            # Generate test documents
            documents = [f"Test document {i}" for i in range(batch_size)]

            # Time the embedding generation
            import time

            start_time = time.time()

            embeddings_result = embeddings.embed_documents(documents)

            end_time = time.time()
            duration = end_time - start_time

            print(f"  ✅ Generated {len(embeddings_result)} embeddings in {duration:.2f} seconds")
            print(f"  📊 Rate: {len(embeddings_result) / duration:.2f} embeddings/second")

        except Exception as e:
            print(f"  ❌ Batch size {batch_size} failed: {e}")
            break  # Stop testing larger batches if smaller ones fail


def main() -> None:
    """Main function to run all demos."""
    print("🚀 Heroku Embeddings Integration Demo")
    print("=" * 60)

    # Setup environment
    config = setup_environment()
    if not config:
        return

    # Create embeddings instance
    try:
        embeddings = HerokuEmbeddings(model=config["model"], api_key=config["api_key"], inference_url=config["inference_url"])
        print(f"✅ Created embeddings instance with model: {config['model']}")

    except Exception as e:
        print(f"❌ Error creating embeddings instance: {e}")
        return

    # Run demos
    try:
        basic_embedding_demo(embeddings)
        advanced_features_demo(embeddings)
        metadata_demo(embeddings)
        similarity_demo(embeddings)
        performance_demo(embeddings)

    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during demo: {e}")

    print("\n🎉 Demo completed!")
    print("\n💡 Next steps:")
    print("  - Integrate embeddings into your LangChain applications")
    print("  - Use vector databases for persistent storage")
    print("  - Implement similarity search and retrieval")
    print("  - Explore Heroku's advanced features for your use case")


if __name__ == "__main__":
    main()
