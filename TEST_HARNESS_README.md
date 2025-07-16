# MiaChat Test Harness

A comprehensive test harness for testing the MiaChat model against the production Heroku Inference API endpoint.

## Overview

The test harness provides a complete testing suite for the MiaChat integration, allowing you to:

- Test all major functionality of the MiaChat model
- Validate production endpoint connectivity
- Measure response times and performance
- Test different parameter configurations
- Generate detailed test reports
- Use `.env` files for environment variable management

## Prerequisites

Before using the test harness, ensure you have:

1. **Environment Variables Set** (choose one method):
   
   **Method A: Using .env file (recommended)**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your values
   INFERENCE_URL=https://your-inference-api-url
   INFERENCE_KEY=your-heroku-inference-api-key
   INFERENCE_MODEL_ID=your-model-id
   ```
   
   **Method B: Using system environment variables**
   ```bash
   export INFERENCE_URL="https://your-inference-api-url"
   export INFERENCE_KEY="your-heroku-inference-api-key"
   export INFERENCE_MODEL_ID="your-model-id"
   ```

2. **Dependencies Installed:**
   ```bash
   pip install -r test_requirements.txt
   ```

## Quick Start

### Run All Tests with .env file
```bash
python test_harness.py --env .env
```

### Run All Tests with system environment variables
```bash
python test_harness.py
```

### Run a Specific Test
```bash
# With .env file
python test_harness.py --env .env --test basic

# With system environment variables
python test_harness.py --test basic
```

### Use Shell Script with .env file
```bash
./run_tests.sh --env .env basic
```

### List Available Tests
```bash
python test_harness.py --list-tests
```

### Use Custom Configuration
```bash
python test_harness.py --config test_config.yaml --env .env
```

### Save Results to Specific File
```bash
python test_harness.py --env .env --save-results my_results.json
```

## Environment Variable Management

The test harness supports multiple ways to manage environment variables:

### 1. .env Files (Recommended)

Create a `.env` file in your project root:

```bash
# .env
INFERENCE_URL=https://your-inference-api-url
INFERENCE_KEY=your-heroku-inference-api-key
INFERENCE_MODEL_ID=your-model-id

# Optional: Additional configuration
TEMPERATURE=0.1
MAX_TOKENS=256
TIMEOUT=60
```

Then use it with:
```bash
python test_harness.py --env .env
```

### 2. System Environment Variables

Set environment variables in your shell:
```bash
export INFERENCE_URL="https://your-inference-api-url"
export INFERENCE_KEY="your-heroku-inference-api-key"
export INFERENCE_MODEL_ID="your-model-id"
```

### 3. Multiple .env Files

The test harness automatically tries these files in order:
- `.env` (specified with `--env`)
- `.env`
- `.env.local`
- `.env.production`

### 4. Shell Script Support

The shell script also supports .env files:
```bash
./run_tests.sh --env .env basic
```

## Available Tests

| Test Name | Description |
|-----------|-------------|
| `basic` | Basic conversation functionality |
| `system` | Conversation with system messages |
| `multi_turn` | Multi-turn conversation testing |
| `string` | Direct string input testing |
| `tool` | Tool message handling |
| `function` | Function message handling |
| `streaming` | Streaming response testing |
| `temperature` | Temperature parameter variation |
| `max_tokens` | Max tokens parameter testing |
| `stop` | Stop sequences testing |
| `error` | Error handling validation |

## Configuration

The test harness supports configuration via YAML files. See `test_config.yaml` for a complete example.

### Model Configuration
```yaml
model:
  temperature: 0.1
  max_tokens: 256
  timeout: 30
  streaming: false
  top_p: 0.95
```

### Test Configuration
```yaml
tests:
  basic_conversation: true
  system_message: true
  # ... other tests
```

## Programmatic Usage

You can also use the test harness programmatically:

```python
from test_harness import TestHarness

# Initialize with .env file
harness = TestHarness(env_file=".env")

# Initialize with custom config and .env file
config = {
    "model": {
        "temperature": 0.0,
        "max_tokens": 100
    }
}
harness = TestHarness(config, env_file=".env")

# Run specific test
result = harness.run_specific_test("basic")
print(f"Test passed: {result.success}")

# Run all tests
results = harness.run_all_tests()
harness.print_summary()
harness.save_results("my_results.json")
```

## Example Usage

See `example_usage.py` for comprehensive examples of how to use the test harness programmatically, including dotenv examples.

## Output

### Console Output
The test harness provides real-time feedback:
```
✅ Loaded environment from: .env
✅ Environment variables validated
✅ MiaChat model initialized
🚀 Starting test suite with 11 tests...
============================================================
🧪 Running test: Basic Conversation
✅ Test passed: Basic Conversation
🧪 Running test: System Message
✅ Test passed: System Message
...
```

### Test Summary
```
============================================================
📊 TEST SUMMARY
============================================================
Total Tests: 11
Passed: 10
Failed: 1
Total Duration: 45.23s
Average Duration: 4.11s

❌ FAILED TESTS:
  - Streaming: Connection timeout

✅ PASSED TESTS:
  - Basic Conversation (2.34s)
    Metadata: {'input_tokens': 5, 'output_tokens': 12, 'total_tokens': 17}
  - System Message (3.12s)
    Metadata: {'input_tokens': 8, 'output_tokens': 15, 'total_tokens': 23}
  ...
```

### JSON Results
Test results are automatically saved to a JSON file with timestamp:
```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "environment": {
    "inference_url": "https://your-inference-api-url",
    "model_id": "your-model-id"
  },
  "results": [
    {
      "name": "Basic Conversation",
      "success": true,
      "duration": 2.34,
      "error": null,
      "response": "Hello! I'm doing well, thank you for asking...",
      "metadata": {
        "input_tokens": 5,
        "output_tokens": 12,
        "total_tokens": 17
      }
    }
  ]
}
```

## Test Details

### Basic Conversation Test
Tests simple human message interaction:
```python
messages = [HumanMessage(content="Hello! How are you today?")]
result = chat_model.invoke(messages)
```

### System Message Test
Tests conversation with system instructions:
```python
messages = [
    SystemMessage(content="You are a helpful assistant that speaks in a friendly tone."),
    HumanMessage(content="What's the weather like?")
]
```

### Multi-turn Conversation Test
Tests conversation context handling:
```python
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 2 + 2?"),
    AIMessage(content="2 + 2 equals 4."),
    HumanMessage(content="What about 3 + 3?")
]
```

### Streaming Test
Tests streaming response functionality:
```python
streaming_model = MiaChat(streaming=True)
for chunk in streaming_model.stream(messages):
    if chunk.message.content:
        full_response += chunk.message.content
```

### Temperature Variation Test
Tests different temperature settings:
```python
# Low temperature (deterministic)
low_temp_model = MiaChat(temperature=0.0)
# High temperature (creative)
high_temp_model = MiaChat(temperature=0.9)
```

## Error Handling

The test harness includes comprehensive error handling:

- **Environment Validation**: Checks for required environment variables
- **Dotenv Loading**: Gracefully handles missing .env files
- **Connection Errors**: Handles network timeouts and connection issues
- **API Errors**: Handles API-specific error responses
- **Invalid Input**: Tests error handling for invalid inputs

## Performance Monitoring

The test harness tracks:
- Response times for each test
- Token usage (input, output, total)
- Success/failure rates
- Error details and stack traces

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```
   ❌ Missing required environment variables: INFERENCE_URL, INFERENCE_KEY
   ```
   Solution: Set the required environment variables or create a `.env` file

2. **Dotenv Not Available**
   ```
   ⚠️  python-dotenv not available. Install with: pip install python-dotenv
   ```
   Solution: Install python-dotenv or use system environment variables

3. **Connection Timeout**
   ```
   ❌ Test failed: Basic Conversation
      Error: Connection timeout
   ```
   Solution: Check your network connection and API endpoint

4. **Authentication Error**
   ```
   ❌ Test failed: Basic Conversation
      Error: 401 Unauthorized
   ```
   Solution: Verify your API key is correct

5. **Model Not Found**
   ```
   ❌ Test failed: Basic Conversation
      Error: 404 Model not found
   ```
   Solution: Check your model ID is correct

### Debug Mode

For detailed debugging, you can modify the test harness to include more verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Environment File Examples

**Basic .env file:**
```bash
INFERENCE_URL=https://your-inference-api-url
INFERENCE_KEY=your-heroku-inference-api-key
INFERENCE_MODEL_ID=your-model-id
```

**Advanced .env file with optional settings:**
```bash
INFERENCE_URL=https://your-inference-api-url
INFERENCE_KEY=your-heroku-inference-api-key
INFERENCE_MODEL_ID=your-model-id

# Optional model parameters
TEMPERATURE=0.1
MAX_TOKENS=256
TIMEOUT=30
TOP_P=0.95
STREAMING=false
LOG_LEVEL=INFO
```

## Contributing

To add new tests to the harness:

1. Add a new test method to the `TestHarness` class
2. Update the test list in `run_all_tests()`
3. Add the test to the test map in `run_specific_test()`
4. Update this documentation

Example new test:
```python
def test_custom_functionality(self) -> tuple[str, Dict[str, Any]]:
    """Test custom functionality."""
    messages = [HumanMessage(content="Your custom prompt")]
    result = self.chat_model.invoke(messages)
    return result.content, {"custom_metadata": "value"}
```

## License

This test harness is part of the langchain-heroku project and follows the same license terms. 