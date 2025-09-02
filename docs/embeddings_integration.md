# Heroku Embeddings Integration with LangChain

This document outlines the strategy and implementation for integrating Heroku's embeddings API with LangChain, addressing the compatibility challenges identified in `EMBEDDINGS_NOTES.md`.

## Overview

The `HerokuEmbeddings` class provides a LangChain-compatible interface to Heroku's Inference API `/v1/embeddings` endpoint. It handles the compatibility issues between Heroku's API and OpenAI's embeddings API while preserving access to Heroku's advanced features.

## Compatibility Strategy

### OpenAI Compatibility Mode (Default)

By default, the embeddings class operates in OpenAI compatibility mode:

```python
from langchain_heroku.embeddings import HerokuEmbeddings

# OpenAI-compatible configuration
embeddings = HerokuEmbeddings(
    model="text-embedding-ada-002",  # OpenAI model name
    api_key="your-key",
    inference_url="https://your-url.com"
)

# Uses OpenAI-compatible defaults:
# - encoding_format: "raw"
# - embedding_type: "float"
# - allow_ignored_params: True
# - input_type: None
```

### Heroku Advanced Features Mode

For advanced use cases, you can enable Heroku-specific features:

```python
# Advanced Heroku configuration
embeddings = HerokuEmbeddings(
    model="cohere-embed-multilingual",
    api_key="your-key",
    inference_url="https://your-url.com",
    input_type="search_document",      # Heroku-specific
    encoding_format="base64",          # Heroku-specific
    embedding_type="int8",             # Heroku-specific
    allow_ignored_params=False         # Heroku-specific
)
```

## Key Compatibility Considerations

### 1. Input Type Optimization

Heroku supports input type specification for optimization:

- `search_document`: For documents to be searched
- `search_query`: For search queries
- `classification`: For classification tasks
- `clustering`: For clustering tasks

**OpenAI Compatibility**: OpenAI doesn't support input types, so this parameter is ignored in compatibility mode.

### 2. Encoding Format

- **Heroku**: Supports `"raw"` and `"base64"`
- **OpenAI**: Only supports `"float"` (equivalent to Heroku's `"raw"`)

**Strategy**: Default to `"raw"` for OpenAI compatibility, but allow `"base64"` for advanced use cases.

### 3. Embedding Type

- **Heroku**: Supports `"float"`, `"int8"`, `"uint8"`, `"binary"`, `"ubinary"`
- **OpenAI**: Only supports `"float"`

**Strategy**: Default to `"float"` for OpenAI compatibility, but allow other types for advanced use cases.

### 4. Parameter Flexibility

- **Heroku**: `allow_ignored_params` parameter for graceful handling of unsupported parameters
- **OpenAI**: Strict parameter validation

**Strategy**: Default to `True` for OpenAI compatibility, but allow `False` for strict mode.

## Interactive Examples

### Jupyter Notebook

For interactive examples and demonstrations, see the [embeddings.ipynb](embeddings.ipynb) notebook in this directory. The notebook includes:

- **Basic Usage**: Single queries and batch processing
- **OpenAI Compatibility**: Drop-in replacement examples
- **Heroku Advanced Features**: Input types, encoding formats, embedding types
- **Metadata Retrieval**: Access to additional API response information
- **Performance Testing**: Batch size optimization and timing
- **Error Handling**: Robust error handling for various scenarios
- **LangChain Integration**: Working with other LangChain components

### Basic Usage

```python
from langchain_heroku.embeddings import HerokuEmbeddings

# Initialize with environment variables
embeddings = HerokuEmbeddings()

# Single query embedding
query_embedding = embeddings.embed_query("What is machine learning?")

# Batch document embedding
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand text."
]
document_embeddings = embeddings.embed_documents(documents)
```

### Advanced Usage with Metadata

```python
# Get embeddings with full response metadata
result = embeddings.embed_query_with_metadata("Advanced query")
print(f"Embedding dimension: {len(result['embedding'])}")
print(f"Model used: {result['model']}")
print(f"Token usage: {result['usage']}")

# Batch with metadata
results = embeddings.embed_documents_with_metadata(documents)
for i, result in enumerate(results):
    print(f"Document {i}: {len(result['embedding'])} dimensions")
```

### Search-Optimized Embeddings

```python
# For search applications
search_embeddings = HerokuEmbeddings(
    model="cohere-embed-multilingual",
    input_type="search_document",  # Optimize for search
    api_key="your-key",
    inference_url="https://your-url.com"
)

# Generate search document embeddings
search_docs = ["Document 1", "Document 2", "Document 3"]
doc_embeddings = search_embeddings.embed_documents(search_docs)

# Generate search query embedding
query_embedding = search_embeddings.embed_query("search query")
```

### Quantized Embeddings for Storage Efficiency

```python
# Use int8 embeddings for storage efficiency
efficient_embeddings = HerokuEmbeddings(
    model="cohere-embed-multilingual",
    embedding_type="int8",  # 8-bit integers instead of 32-bit floats
    api_key="your-key",
    inference_url="https://your-url.com"
)

# Note: These will be converted to float for LangChain compatibility
embeddings = efficient_embeddings.embed_documents(documents)
```

## Environment Configuration

Set these environment variables for automatic configuration:

```bash
export INFERENCE_URL="https://your-inference-api-url"
export INFERENCE_KEY="your-heroku-inference-api-key"
export INFERENCE_MODEL_ID="your-model-id"
```

Or use the `INFERENCE_EMBED_KEY` environment variable as an alternative:

```bash
export INFERENCE_EMBED_KEY="your-heroku-inference-api-key"
```

## Model Selection

### OpenAI-Compatible Models

For maximum compatibility, use these model names:

- `text-embedding-ada-002`
- `text-embedding-3-small`
- `text-embedding-3-large`

### Heroku-Specific Models

For advanced features, consider these models:

- `cohere-embed-multilingual`: Multilingual support with input type optimization
- `cohere-embed-english-v3.0`: English-optimized embeddings
- Custom models deployed on Heroku

## Error Handling and Retries

The implementation includes robust error handling:

- **Automatic retries**: Up to 2 retries for failed requests
- **Timeout handling**: Configurable timeout (default: 30 seconds)
- **Parameter validation**: Comprehensive validation of all parameters
- **Graceful degradation**: Falls back to OpenAI-compatible mode when possible

## Migration from OpenAI

### Simple Migration

```python
# Before (OpenAI)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key="your-key")

# After (Heroku)
from langchain_heroku.embeddings import HerokuEmbeddings
embeddings = HerokuEmbeddings(
    model="text-embedding-ada-002",  # Same model name
    api_key="your-heroku-key",
    inference_url="https://your-heroku-url.com"
)

# Usage remains identical
result = embeddings.embed_query("Hello, world!")
```

### Advanced Migration

```python
# Leverage Heroku-specific features
embeddings = HerokuEmbeddings(
    model="cohere-embed-multilingual",
    input_type="search_document",
    api_key="your-heroku-key",
    inference_url="https://your-heroku-url.com"
)

# Enhanced functionality while maintaining LangChain compatibility
```

## Testing and Validation

### Unit Tests

Run the unit tests to verify functionality:

```bash
pytest tests/unit_tests/test_embeddings.py -v
```

### Integration Tests

Run integration tests with real API calls:

```bash
# Set environment variables first
export INFERENCE_URL="https://your-url.com"
export INFERENCE_KEY="your-key"
export INFERENCE_MODEL_ID="your-model"

# Run integration tests
pytest tests/integration_tests/test_embeddings_integration.py -v -m integration
```

## Performance Considerations

### Batch Processing

- Use `embed_documents()` for multiple texts to reduce API calls
- Monitor batch size limits for your specific model
- Consider parallel processing for large document sets

### Caching

- Implement embedding caching for repeated queries
- Use vector databases for persistent storage
- Consider similarity search for finding existing embeddings

### Model Selection

- **OpenAI models**: Best for English text, consistent dimensions
- **Cohere models**: Better for multilingual content, input type optimization
- **Custom models**: Optimized for specific domains or languages

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Verify `INFERENCE_KEY` or `HEROKU_API_KEY`
2. **Model Not Found**: Check `INFERENCE_MODEL_ID` and model availability
3. **Parameter Errors**: Ensure parameters are supported by your model
4. **Dimension Mismatches**: Different models produce different embedding dimensions

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

embeddings = HerokuEmbeddings(
    model="your-model",
    api_key="your-key",
    inference_url="https://your-url.com"
)
```

## Future Enhancements

### Planned Features

- **Async support**: Non-blocking API calls
- **Streaming embeddings**: Real-time embedding generation
- **Custom model endpoints**: Direct integration with deployed models
- **Advanced similarity metrics**: Built-in similarity calculations

### Community Contributions

- **Model adapters**: Support for additional embedding models
- **Performance optimizations**: Improved batch processing
- **Integration examples**: More use case demonstrations

## Conclusion

The `HerokuEmbeddings` integration provides a robust, LangChain-compatible interface to Heroku's embeddings API. It maintains OpenAI compatibility while unlocking Heroku's advanced features, making it an ideal choice for applications that need both reliability and performance.

The implementation handles all compatibility challenges identified in `EMBEDDINGS_NOTES.md` and provides a smooth migration path from OpenAI embeddings to Heroku's enhanced capabilities.
