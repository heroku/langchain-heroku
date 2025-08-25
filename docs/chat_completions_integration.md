# Heroku Chat Completions Integration with LangChain

This document outlines the strategy and implementation for integrating Heroku's chat completions API with LangChain, providing a seamless interface to Heroku's Inference API `/v1/chat/completions` endpoint.

## Overview

The `ChatHeroku` class provides a LangChain-compatible interface to Heroku's Inference API for chat completions. It offers OpenAI compatibility while leveraging Heroku's powerful inference infrastructure and model selection.

## Key Features

- **OpenAI Compatibility**: Drop-in replacement for OpenAI chat models
- **Streaming Support**: Real-time streaming of responses using Server-Sent Events (SSE)
- **Flexible Configuration**: Support for various models and generation parameters
- **LangChain Integration**: Seamless integration with LangChain ecosystem
- **Error Handling**: Robust error handling with retry logic

## Compatibility Strategy

### OpenAI Compatibility Mode

The chat model operates in full OpenAI compatibility mode:

```python
from langchain_heroku.chat_models import ChatHeroku

# OpenAI-compatible configuration
chat_model = ChatHeroku(
    model="gpt-3.5-turbo",  # OpenAI model name
    api_key="your-key",
    inference_url="https://your-url.com"
)

# All OpenAI parameters are supported:
# - temperature, max_tokens, top_p
# - frequency_penalty, presence_penalty
# - stop sequences, etc.
```

### Heroku-Specific Features

While maintaining OpenAI compatibility, the integration leverages Heroku's infrastructure:

- **Model Selection**: Access to various open-source models (Llama, Mistral, etc.)
- **Inference API**: Direct integration with Heroku's Inference API
- **Performance**: Optimized for Heroku's infrastructure

## Usage Examples

### Interactive Examples

#### Jupyter Notebook

For interactive examples and demonstrations, see the [completions.ipynb](completions.ipynb) notebook in this directory. The notebook includes:

- **Basic Usage**: Simple message handling and responses
- **System Messages and Context**: Setting behavior and maintaining context
- **Multi-turn Conversations**: Managing conversation flow
- **Streaming Responses**: Real-time response generation
- **Custom Configuration**: Parameter tuning and optimization
- **Batch Processing**: Handling multiple messages efficiently
- **Error Handling**: Robust error handling for various scenarios
- **LangChain Integration**: Working with chains, agents, and other components
- **Performance Testing**: Response time analysis and optimization

### Basic Usage

```python
from langchain_heroku.chat_models import ChatHeroku
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize with environment variables
chat_model = ChatHeroku()

# Simple chat completion
message = HumanMessage(content="Hello! How are you today?")
response = chat_model.invoke([message])
print(response.content)
```

### System Messages and Context

```python
# Chat with system message
system_message = SystemMessage(content="You are a helpful AI assistant that specializes in explaining complex topics in simple terms.")
user_message = HumanMessage(content="Can you explain quantum computing in simple terms?")

messages = [system_message, user_message]
response = chat_model.invoke(messages)
print(response.content)
```

### Multi-turn Conversations

```python
# Multi-turn conversation
conversation = [
    SystemMessage(content="You are a helpful coding assistant. Provide clear, concise explanations."),
    HumanMessage(content="What is a function in programming?")
]

# First turn
response1 = chat_model.invoke(conversation)
conversation.append(response1)

# Second turn
conversation.append(HumanMessage(content="Can you give me an example in Python?"))
response2 = chat_model.invoke(conversation)
print(response2.content)
```

### Streaming Responses

```python
# Streaming response
message = HumanMessage(content="Write a short story about a robot learning to paint.")

for chunk in chat_model.stream([message]):
    if chunk.content:
        print(chunk.content, end="")
print()  # New line after streaming
```

### Custom Configuration

```python
# Custom configuration
custom_chat_model = ChatHeroku(
    model="llama-2-13b-chat",
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1
)

response = custom_chat_model.invoke([HumanMessage(content="Write a creative haiku about artificial intelligence.")])
print(response.content)
```

### Batch Processing

```python
# Batch processing
messages = [
    [HumanMessage(content="What is machine learning?")],
    [HumanMessage(content="Explain neural networks.")],
    [HumanMessage(content="What is deep learning?")]
]

responses = chat_model.batch(messages)
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response.content[:100]}...")
```

## LangChain Integration

### LLMChain

```python
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides concise explanations."),
    ("human", "Explain {topic} in {style} style.")
])

# Create LLM chain
chain = LLMChain(llm=chat_model, prompt=prompt)

# Run the chain
result = chain.run({"topic": "blockchain technology", "style": "simple"})
print(result)
```

### Agents

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Create a simple tool
def get_current_time():
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

time_tool = Tool(
    name="get_current_time",
    description="Get the current date and time",
    func=get_current_time
)

# Initialize agent
agent = initialize_agent(
    tools=[time_tool],
    llm=chat_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
result = agent.run("What is the current time?")
print(result)
```

## Configuration Options

### Basic Parameters

- **model**: Model identifier (e.g., "llama-2-70b-chat", "mistral-7b-instruct")
- **api_key**: API key for authentication
- **inference_url**: URL for Heroku Inference API
- **timeout**: Request timeout in seconds

### Generation Parameters

- **temperature**: Controls randomness (0.0 to 2.0)
- **max_tokens**: Maximum number of tokens to generate
- **top_p**: Nucleus sampling parameter (0.0 to 1.0)
- **frequency_penalty**: Penalty for frequent tokens (-2.0 to 2.0)
- **presence_penalty**: Penalty for new tokens (-2.0 to 2.0)
- **stop**: Stop sequences for generation

### Environment Variables

- **INFERENCE_URL**: URL for Heroku Inference API
- **INFERENCE_KEY**: API key for authentication
- **INFERENCE_MODEL_ID**: Default model ID

## Error Handling

The integration includes robust error handling:

```python
try:
    response = chat_model.invoke([HumanMessage(content="Your message here")])
    print(response.content)
except Exception as e:
    print(f"Error: {e}")
    # Handle specific error types as needed
```

### Common Error Scenarios

- **Empty messages**: Validation for empty content
- **Empty message lists**: Validation for empty message arrays
- **Long messages**: Handling of very long input text
- **API errors**: Network issues, rate limiting, authentication failures
- **Model errors**: Unsupported models or parameters

## Performance Considerations

### Response Time Optimization

- **Streaming**: Use streaming for long responses to improve perceived performance
- **Batch processing**: Process multiple messages together when possible
- **Parameter tuning**: Adjust temperature and other parameters for optimal performance

### Resource Management

- **Connection pooling**: Efficient HTTP connection management
- **Timeout handling**: Configurable timeouts for different use cases
- **Retry logic**: Automatic retry for transient failures

## Best Practices

### 1. Message Handling

- Always validate message content before sending
- Use appropriate message types (HumanMessage, SystemMessage, etc.)
- Maintain conversation context when needed

### 2. Error Handling

- Implement comprehensive error handling
- Handle rate limiting and timeouts gracefully
- Provide fallback responses when possible

### 3. Performance Optimization

- Use streaming for long responses
- Implement caching for repeated queries
- Monitor response times and adjust parameters

### 4. OpenAI Compatibility

- Use default settings for drop-in replacement
- Test thoroughly before production deployment
- Leverage LangChain's built-in features

## Migration from OpenAI

### Simple Replacement

```python
# Before (OpenAI)
from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI(model="gpt-3.5-turbo")

# After (Heroku)
from langchain_heroku.chat_models import ChatHeroku
chat_model = ChatHeroku(model="gpt-3.5-turbo")
```

### Parameter Mapping

All OpenAI parameters are directly supported:

```python
chat_model = ChatHeroku(
    model="llama-2-70b-chat",  # Heroku model
    temperature=0.7,            # Same as OpenAI
    max_tokens=1000,            # Same as OpenAI
    top_p=0.9,                 # Same as OpenAI
    # ... all other OpenAI parameters
)
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Verify INFERENCE_KEY and INFERENCE_URL
2. **Model Not Found**: Check INFERENCE_MODEL_ID or model parameter
3. **Timeout Errors**: Adjust timeout parameter or check network connectivity
4. **Rate Limiting**: Implement exponential backoff for retries

### Debug Mode

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed API requests and responses
```

## Summary

The `ChatHeroku` class provides a powerful and flexible way to integrate Heroku's Inference API with LangChain, offering:

- **Full OpenAI compatibility** for easy migration
- **Streaming support** for real-time responses
- **Robust error handling** with retry logic
- **Seamless LangChain integration** with chains, agents, and other components
- **Performance optimization** for production use cases

This integration enables developers to leverage Heroku's powerful inference infrastructure while maintaining compatibility with existing LangChain applications and workflows.


