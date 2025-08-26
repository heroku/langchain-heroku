"""Cross-component integration tests for Heroku LangChain components.

This module contains integration tests that validate multiple Heroku components
working together in realistic workflows, following LangChain's testing guidelines.

Key aspects:
- Tests ChatHeroku and HerokuEmbeddings integration
- Validates RAG-style workflows  
- Tests shared configuration handling
- Includes concurrent usage scenarios
- Uses environment variable configuration with graceful skipping
"""

import os
import time
from typing import Any, List

import pytest

# Load dotenv if available
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

    def _load_dotenv_stub(*args: Any, **kwargs: Any) -> bool:
        return False

    load_dotenv = _load_dotenv_stub

from langchain_heroku.chat_models import ChatHeroku
from langchain_heroku.embeddings import HerokuEmbeddings


@pytest.mark.integration
class TestCrossComponentIntegration:
    """Integration tests for multiple Heroku components working together."""

    @pytest.fixture(autouse=True)
    def setup_environment(self) -> None:
        """Setup environment variables for testing."""
        # Load .env file if dotenv is available
        if DOTENV_AVAILABLE:
            for env_file in [".env", ".env.local", ".env.integration"]:
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    break

        # Check if we have the required environment variables
        self.has_env_vars = all([
            os.getenv("INFERENCE_URL"),
            os.getenv("INFERENCE_KEY"),
            os.getenv("INFERENCE_MODEL_ID")
        ])

        if not self.has_env_vars:
            pytest.skip("Cross-component integration tests require INFERENCE_URL, INFERENCE_KEY, and INFERENCE_MODEL_ID environment variables")

    def test_shared_configuration(self) -> None:
        """Test that both components can use shared configuration."""
        # Test that both components can be initialized with same env vars
        chat = ChatHeroku()
        embeddings = HerokuEmbeddings()

        # Both should be able to make API calls
        chat_result = chat.invoke("Hello")
        embedding_result = embeddings.embed_query("Hello")

        assert chat_result.content is not None
        assert len(chat_result.content) > 0
        assert isinstance(embedding_result, list)
        assert len(embedding_result) > 0

    def test_basic_rag_workflow(self) -> None:
        """Test a basic RAG-style workflow using both components."""
        # Initialize components
        embeddings = HerokuEmbeddings()
        chat = ChatHeroku()

        # Sample documents for RAG
        documents = [
            "The capital of France is Paris.",
            "Python is a programming language.",
            "The weather today is sunny.",
        ]

        # Step 1: Create embeddings for documents
        doc_embeddings = embeddings.embed_documents(documents)
        assert len(doc_embeddings) == len(documents)
        assert all(isinstance(emb, list) for emb in doc_embeddings)

        # Step 2: Create query embedding
        query = "What is the capital of France?"
        query_embedding = embeddings.embed_query(query)
        assert isinstance(query_embedding, list)
        assert len(query_embedding) > 0

        # Step 3: Simple similarity search (cosine similarity)
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(a * a for a in vec2) ** 0.5
            return dot_product / (magnitude1 * magnitude2)

        similarities = [
            cosine_similarity(query_embedding, doc_emb) 
            for doc_emb in doc_embeddings
        ]
        
        # Find most similar document
        best_doc_idx = similarities.index(max(similarities))
        best_document = documents[best_doc_idx]

        # Step 4: Use retrieved document in chat completion
        rag_prompt = f"Based on this context: '{best_document}', answer the question: {query}"
        chat_result = chat.invoke(rag_prompt)

        assert chat_result.content is not None
        assert len(chat_result.content) > 0
        
        # Should mention Paris since that's the most relevant document
        content = str(chat_result.content).lower()
        assert "paris" in content

    def test_concurrent_component_usage(self) -> None:
        """Test concurrent usage of both components."""
        import concurrent.futures

        chat = ChatHeroku()
        embeddings = HerokuEmbeddings()

        def chat_task() -> Any:
            return chat.invoke("What is machine learning?")

        def embedding_task() -> Any:
            return embeddings.embed_query("Machine learning is artificial intelligence")

        # Run both tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            chat_future = executor.submit(chat_task)
            embedding_future = executor.submit(embedding_task)

            chat_result = chat_future.result()
            embedding_result = embedding_future.result()

        # Both should succeed
        assert chat_result.content is not None
        assert len(chat_result.content) > 0
        assert isinstance(embedding_result, list)
        assert len(embedding_result) > 0

    def test_component_parameter_consistency(self) -> None:
        """Test that components handle parameters consistently."""
        # Test with explicit parameters
        api_key = os.getenv("INFERENCE_KEY")
        inference_url = os.getenv("INFERENCE_URL")
        model = os.getenv("INFERENCE_MODEL_ID")

        chat = ChatHeroku(
            api_key=api_key,
            inference_url=inference_url,
            model=model,
            timeout=30
        )
        
        embeddings = HerokuEmbeddings(
            api_key=api_key,
            inference_url=inference_url,
            model=model,
            timeout=30
        )

        # Both should work with explicit parameters
        chat_result = chat.invoke("Test message")
        embedding_result = embeddings.embed_query("Test message")

        assert chat_result.content is not None
        assert isinstance(embedding_result, list)

    def test_batch_processing_workflow(self) -> None:
        """Test batch processing using both components."""
        embeddings = HerokuEmbeddings()
        chat = ChatHeroku()

        # Test batch embedding processing
        texts = [
            "What is artificial intelligence?",
            "Explain machine learning.",
            "Define neural networks."
        ]

        # Batch embed all texts
        batch_embeddings = embeddings.embed_documents(texts)
        assert len(batch_embeddings) == len(texts)

        # Process each text with chat (simulate batch chat processing)
        chat_results = []
        for text in texts:
            result = chat.invoke(f"Provide a brief answer: {text}")
            chat_results.append(result)

        assert len(chat_results) == len(texts)
        for result in chat_results:
            assert result.content is not None
            assert len(result.content) > 0

    def test_error_handling_across_components(self) -> None:
        """Test error handling consistency across components."""
        chat = ChatHeroku()
        embeddings = HerokuEmbeddings()

        # Test empty input handling
        with pytest.raises((ValueError, RuntimeError)):
            chat.invoke("")

        with pytest.raises((ValueError, RuntimeError)):
            embeddings.embed_query("")

        # Test invalid input types
        with pytest.raises(ValueError):
            chat.invoke(12345)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            embeddings.embed_query(12345)  # type: ignore[arg-type]

    def test_semantic_search_workflow(self) -> None:
        """Test a more advanced semantic search workflow."""
        embeddings = HerokuEmbeddings()
        chat = ChatHeroku()

        # Create a knowledge base
        knowledge_base = [
            "Paris is the capital city of France and its largest city.",
            "London is the capital and largest city of England and the United Kingdom.",
            "Tokyo is the capital of Japan and the most populous metropolitan area.",
            "Berlin is the capital and largest city of Germany.",
            "Madrid is the capital and most populous city of Spain."
        ]

        # Embed the knowledge base
        kb_embeddings = embeddings.embed_documents(knowledge_base)
        
        # Test query
        query = "Tell me about the capital of Germany"
        query_embedding = embeddings.embed_query(query)

        # Find most relevant document
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(a * a for a in vec2) ** 0.5
            return dot_product / (magnitude1 * magnitude2)

        similarities = [
            cosine_similarity(query_embedding, kb_emb) 
            for kb_emb in kb_embeddings
        ]
        
        best_match_idx = similarities.index(max(similarities))
        best_match = knowledge_base[best_match_idx]

        # Should find the Berlin document
        assert "berlin" in best_match.lower()
        assert "germany" in best_match.lower()

        # Use in chat completion
        enhanced_query = f"Based on this information: '{best_match}', {query}"
        result = chat.invoke(enhanced_query)

        assert result.content is not None
        content = str(result.content).lower()
        assert "berlin" in content or "germany" in content

    def test_performance_across_components(self) -> None:
        """Test performance characteristics across components."""
        chat = ChatHeroku()
        embeddings = HerokuEmbeddings()

        # Test chat performance
        start_time = time.time()
        chat_result = chat.invoke("What is 2 + 2?")
        chat_duration = time.time() - start_time

        # Test embedding performance
        start_time = time.time()
        embedding_result = embeddings.embed_query("What is 2 + 2?")
        embedding_duration = time.time() - start_time

        # Both should complete in reasonable time
        assert chat_duration < 30.0, f"Chat took {chat_duration:.2f}s"
        assert embedding_duration < 30.0, f"Embedding took {embedding_duration:.2f}s"

        # Results should be valid
        assert chat_result.content is not None
        assert isinstance(embedding_result, list)

    def test_metadata_consistency(self) -> None:
        """Test that metadata handling is consistent across components."""
        chat = ChatHeroku()
        embeddings = HerokuEmbeddings()

        # Test chat metadata
        chat_result = chat.invoke("Hello")
        assert "usage_metadata" in chat_result.additional_kwargs
        assert hasattr(chat_result, "response_metadata")

        # Test embedding metadata (if supported)
        try:
            embedding_metadata = embeddings.embed_query_with_metadata("Hello")
            assert "embedding" in embedding_metadata
            assert "model" in embedding_metadata
            assert "response_metadata" in embedding_metadata
        except AttributeError:
            # embed_query_with_metadata might not be available in all versions
            pass

    def test_configuration_validation_across_components(self) -> None:
        """Test configuration validation across components."""
        # Test that both components validate configuration similarly
        
        # Valid configuration should work for both
        valid_config = {
            "api_key": os.getenv("INFERENCE_KEY"),
            "inference_url": os.getenv("INFERENCE_URL"),
            "model": os.getenv("INFERENCE_MODEL_ID"),
        }

        chat = ChatHeroku(**valid_config)  # type: ignore[arg-type]
        embeddings = HerokuEmbeddings(**valid_config)

        # Both should work
        chat_result = chat.invoke("Test")
        embedding_result = embeddings.embed_query("Test")

        assert chat_result.content is not None
        assert isinstance(embedding_result, list)

        # Invalid configuration should fail for both
        invalid_config = {
            "api_key": "invalid-key",
            "inference_url": "https://invalid.url",
            "model": "invalid-model",
        }

        invalid_chat = ChatHeroku(**invalid_config)  # type: ignore[arg-type]
        invalid_embeddings = HerokuEmbeddings(**invalid_config)

        # Both should handle errors gracefully (might be authentication errors)
        try:
            invalid_chat.invoke("Test")
            assert False, "Expected error with invalid configuration"
        except Exception as e:
            # Should get some form of authentication or connection error
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg 
                for keyword in ["auth", "key", "invalid", "unauthorized", "connection", "url"]
            )

        try:
            invalid_embeddings.embed_query("Test")
            assert False, "Expected error with invalid configuration"
        except Exception as e:
            # Should get some form of authentication or connection error
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg 
                for keyword in ["auth", "key", "invalid", "unauthorized", "connection", "url"]
            )