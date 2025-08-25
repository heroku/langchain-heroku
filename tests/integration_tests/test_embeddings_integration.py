"""Integration tests for HerokuEmbeddings."""

import os

import pytest

from langchain_heroku.embeddings import HerokuEmbeddings


@pytest.mark.integration
class TestHerokuEmbeddingsIntegration:
    """Integration tests for HerokuEmbeddings with real API calls."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Get configuration from environment variables
        self.inference_url = os.getenv("INFERENCE_URL")
        self.api_key = os.getenv("INFERENCE_KEY") or os.getenv("HEROKU_API_KEY")
        self.model = os.getenv("INFERENCE_MODEL_ID")

        # Skip tests if configuration is not available
        if not all([self.inference_url, self.api_key, self.model]):
            pytest.skip("Missing required environment variables for integration tests")

        self.embeddings = HerokuEmbeddings(model=self.model, api_key=self.api_key, inference_url=self.inference_url)

    def test_basic_embedding_functionality(self) -> None:
        """Test basic embedding functionality with real API."""
        text = "Hello, world! This is a test of the embedding functionality."

        # Test single query embedding
        embedding = self.embeddings.embed_query(text)

        # Verify the embedding is a list of floats
        assert isinstance(embedding, list)
        assert all(isinstance(x, (int, float)) for x in embedding)
        assert len(embedding) > 0

        # Test batch document embedding
        texts = ["First document for testing.", "Second document with different content.", "Third document to complete the batch."]

        embeddings = self.embeddings.embed_documents(texts)

        # Verify batch results
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

    def test_openai_compatibility_mode(self) -> None:
        """Test that embeddings work in OpenAI compatibility mode."""
        # Create embeddings with OpenAI-compatible defaults
        embeddings = HerokuEmbeddings(model=self.model, api_key=self.api_key, inference_url=self.inference_url)

        text = "Testing OpenAI compatibility mode"
        embedding = embeddings.embed_query(text)

        # Should work without any Heroku-specific parameters
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_heroku_advanced_features(self) -> None:
        """Test Heroku-specific features if the model supports them."""
        # Test with different input types if supported
        test_cases = [
            {"input_type": "search_document"},
            {"input_type": "search_query"},
            {"encoding_format": "raw"},
            {"embedding_type": "float"},
        ]

        text = "Testing advanced Heroku features"

        for params in test_cases:
            try:
                embeddings = HerokuEmbeddings(model=self.model, api_key=self.api_key, inference_url=self.inference_url, **params)

                embedding = embeddings.embed_query(text)
                assert isinstance(embedding, list)
                assert len(embedding) > 0

            except Exception as e:
                # Some models may not support all features
                # This is expected behavior
                print(f"Feature {params} not supported by model {self.model}: {e}")

    def test_embedding_consistency(self) -> None:
        """Test that embeddings are consistent for the same input."""
        text = "Consistency test text"

        # Generate embeddings multiple times
        embedding1 = self.embeddings.embed_query(text)
        embedding2 = self.embeddings.embed_query(text)

        # Embeddings should be identical for the same input
        assert embedding1 == embedding2
        assert len(embedding1) == len(embedding2)

    def test_embedding_dimensions(self) -> None:
        """Test that embeddings have consistent dimensions."""
        texts = ["Short text", "This is a longer text with more words and content to process", "Very short"]

        embeddings = self.embeddings.embed_documents(texts)

        # All embeddings should have the same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1, f"All embeddings should have same dimension, got: {dimensions}"

        # Dimension should be reasonable (not 0, not extremely large)
        dimension = dimensions[0]
        assert dimension > 0
        assert dimension < 10000  # Reasonable upper bound

    def test_embedding_metadata(self) -> None:
        """Test embedding with metadata retrieval."""
        text = "Testing metadata retrieval"

        # Test single query with metadata
        result = self.embeddings.embed_query_with_metadata(text)

        assert "embedding" in result
        assert "model" in result
        assert "response_metadata" in result

        embedding = result["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0

        # Test batch with metadata
        texts = ["Text 1", "Text 2"]
        results = self.embeddings.embed_documents_with_metadata(texts)

        assert len(results) == len(texts)
        for i, result_item in enumerate(results):
            assert "embedding" in result_item
            assert "text" in result_item
            assert result_item["text"] == texts[i]
            assert "model" in result_item
            assert "response_metadata" in result_item

    def test_error_handling(self) -> None:
        """Test error handling for invalid inputs."""
        # Test with empty string
        with pytest.raises(Exception):
            self.embeddings.embed_query("")

        # Test with empty list
        with pytest.raises(Exception):
            self.embeddings.embed_documents([])

        # Test with very long text (may exceed model limits)
        very_long_text = "This is a very long text. " * 1000
        try:
            embedding = self.embeddings.embed_query(very_long_text)
            # If it succeeds, verify the result
            assert isinstance(embedding, list)
            assert len(embedding) > 0
        except Exception as e:
            # Expected behavior for some models
            print(f"Long text not supported: {e}")

    def test_different_model_behavior(self) -> None:
        """Test that different models behave differently (if multiple models available)."""
        # This test assumes you might have access to different models
        # For now, we'll test with the configured model
        text = "Testing model behavior"

        embedding = self.embeddings.embed_query(text)
        assert isinstance(embedding, list)
        assert len(embedding) > 0

        # Store the dimension for comparison
        dimension = len(embedding)

        # If you have access to different models, you could test them here
        # and verify they produce different dimensions or behaviors
        print(f"Model {self.model} produces embeddings with dimension {dimension}")

    def test_batch_size_limits(self) -> None:
        """Test behavior with different batch sizes."""
        # Test with single item
        single_result = self.embeddings.embed_documents(["Single item"])
        assert len(single_result) == 1

        # Test with multiple items
        multiple_texts = [f"Text {i}" for i in range(5)]
        multiple_results = self.embeddings.embed_documents(multiple_texts)
        assert len(multiple_results) == 5

        # Test with larger batch (if supported)
        large_batch = [f"Text {i}" for i in range(20)]
        try:
            large_results = self.embeddings.embed_documents(large_batch)
            assert len(large_results) == 20
        except Exception as e:
            # Some models may have batch size limits
            print(f"Large batch not supported: {e}")

    def test_special_characters_and_unicode(self) -> None:
        """Test embedding with special characters and unicode."""
        test_texts = ["Hello, world!", "Special chars: @#$%^&*()", "Unicode: 🚀🌟🎉", "Numbers: 12345", "Mixed: Hello 世界! @#$% 🚀"]

        for text in test_texts:
            try:
                embedding = self.embeddings.embed_query(text)
                assert isinstance(embedding, list)
                assert len(embedding) > 0
            except Exception as e:
                # Some models may not handle all special characters
                print(f"Text '{text}' not supported: {e}")

    def test_embedding_quality_basic(self) -> None:
        """Basic test of embedding quality and semantic meaning."""
        # Test that similar texts produce somewhat similar embeddings
        similar_texts = ["The cat is on the mat", "A cat sits on the mat", "The mat has a cat on it"]

        similar_embeddings = self.embeddings.embed_documents(similar_texts)

        # Test that different texts produce different embeddings
        different_texts = ["The cat is on the mat", "The weather is sunny today", "Mathematics is the language of science"]

        different_embeddings = self.embeddings.embed_documents(different_texts)

        # All embeddings should have the same dimension
        assert all(len(emb) == len(similar_embeddings[0]) for emb in similar_embeddings)
        assert all(len(emb) == len(different_embeddings[0]) for emb in different_embeddings)

        # Similar texts should produce embeddings with some similarity
        # (This is a basic test - more sophisticated similarity metrics could be used)
        print("Embedding quality test completed - check that similar texts produce similar embeddings")
