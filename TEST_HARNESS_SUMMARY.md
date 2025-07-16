# MiaChat Test Harness - Summary

I've built a comprehensive test harness for testing the MiaChat model against the production Heroku Inference API endpoint with **dotenv support** for easier environment variable management. Here's what has been created:

## Files Created

### 1. `test_harness.py` - Main Test Harness
- **Purpose**: Comprehensive test suite for MiaChat production endpoint
- **Features**:
  - 11 different test scenarios covering all major functionality
  - **Dotenv support** for environment variable management
  - Environment variable validation with helpful error messages
  - Detailed test reporting with timing and metadata
  - JSON result export with timestamps
  - Command-line interface with various options
  - Error handling and retry logic

### 2. `test_config.yaml` - Configuration Template
- **Purpose**: YAML configuration file for customizing test behavior
- **Features**:
  - Model parameter configuration (temperature, max_tokens, etc.)
  - Test enable/disable options
  - Custom prompts
  - Performance thresholds
  - Logging configuration

### 3. `example_usage.py` - Usage Examples
- **Purpose**: Demonstrates programmatic usage of the test harness
- **Features**:
  - Basic usage examples
  - **Dotenv usage examples**
  - Custom configuration examples
  - Error handling examples
  - Streaming test examples
  - **Environment file creation examples**

### 4. `run_tests.sh` - Shell Script Runner
- **Purpose**: Easy-to-use shell script for running tests
- **Features**:
  - Colored output for better UX
  - **Dotenv file support**
  - Environment variable validation
  - Simple command-line interface
  - Help documentation

### 5. `tests/integration_tests/test_production_endpoint.py` - Integration Tests
- **Purpose**: Pytest-based integration tests for CI/CD
- **Features**:
  - 10 production endpoint tests
  - **Automatic dotenv loading**
  - Automatic skipping if environment variables missing
  - Proper test isolation
  - Detailed assertions

### 6. `test_requirements.txt` - Dependencies
- **Purpose**: Lists additional dependencies needed for test harness
- **Features**:
  - Core dependencies
  - Test-specific dependencies
  - **python-dotenv dependency**
  - Optional enhanced reporting tools

### 7. `TEST_HARNESS_README.md` - Documentation
- **Purpose**: Comprehensive documentation for the test harness
- **Features**:
  - Installation instructions
  - **Dotenv usage guide**
  - Usage examples
  - Configuration guide
  - Troubleshooting section
  - Contributing guidelines

### 8. `.env.example` - Environment Template
- **Purpose**: Template for environment variable configuration
- **Features**:
  - Required environment variables
  - Optional configuration examples
  - Clear documentation

## Test Scenarios Covered

1. **Basic Conversation** - Simple human message interaction
2. **System Message** - Conversation with system instructions
3. **Multi-turn Conversation** - Context handling across multiple turns
4. **String Input** - Direct string input functionality
5. **Tool Messages** - Tool message handling
6. **Function Messages** - Function message handling
7. **Streaming** - Streaming response functionality
8. **Temperature Variation** - Different temperature settings
9. **Max Tokens** - Token limit testing
10. **Stop Sequences** - Stop sequence functionality
11. **Error Handling** - Invalid input validation

## Environment Variable Management

The test harness now supports **multiple ways** to manage environment variables:

### 1. .env Files (Recommended)
```bash
# Create .env file
cp .env.example .env

# Edit .env with your values
INFERENCE_URL=https://your-inference-api-url
INFERENCE_KEY=your-heroku-inference-api-key
INFERENCE_MODEL_ID=your-model-id
```

### 2. System Environment Variables
```bash
export INFERENCE_URL="https://your-inference-api-url"
export INFERENCE_KEY="your-heroku-inference-api-key"
export INFERENCE_MODEL_ID="your-model-id"
```

### 3. Multiple .env Files
The harness automatically tries:
- `.env` (specified with `--env`)
- `.env`
- `.env.local`
- `.env.production`

## Usage Examples

### Quick Start with .env file
```bash
# Set up environment
cp .env.example .env
# Edit .env with your values

# Run all tests
python test_harness.py --env .env

# Run specific test
python test_harness.py --env .env --test basic

# Use shell script
./run_tests.sh --env .env basic
```

### Quick Start with system environment variables
```bash
# Set environment variables
export INFERENCE_URL="https://your-inference-api-url"
export INFERENCE_KEY="your-heroku-inference-api-key"
export INFERENCE_MODEL_ID="your-model-id"

# Run all tests
python test_harness.py

# Run specific test
python test_harness.py --test basic
```

### Programmatic Usage with dotenv
```python
from test_harness import TestHarness

# Initialize with .env file
harness = TestHarness(env_file=".env")

# Initialize with custom config and .env file
config = {"model": {"temperature": 0.0}}
harness = TestHarness(config, env_file=".env")

# Run specific test
result = harness.run_specific_test("basic")
print(f"Test passed: {result.success}")

# Run all tests
results = harness.run_all_tests()
harness.print_summary()
harness.save_results("results.json")
```

### Integration with Existing Test Suite
```bash
# Run production integration tests with dotenv
pytest tests/integration_tests/test_production_endpoint.py -v

# Run with specific .env file
INFERENCE_URL="..." INFERENCE_KEY="..." INFERENCE_MODEL_ID="..." pytest tests/integration_tests/test_production_endpoint.py
```

## Key Features

### 1. Comprehensive Testing
- Tests all major MiaChat functionality
- Covers edge cases and error conditions
- Validates production endpoint connectivity

### 2. **Dotenv Integration**
- **Automatic .env file loading**
- **Multiple .env file support** - `.env`, `.env.local`, `.env.production`
- **Graceful fallback** - Falls back to system environment variables if .env not found
- **Helpful error messages** - Clear guidance when environment variables are missing

### 3. Detailed Reporting
- Real-time test progress
- Timing information for each test
- Token usage metadata
- JSON export with timestamps

### 4. Flexible Configuration
- YAML-based configuration
- **Environment variable support** (both .env and system)
- Customizable test parameters

### 5. Multiple Usage Patterns
- Command-line interface with dotenv support
- Programmatic API with dotenv support
- Shell script wrapper with dotenv support
- Pytest integration with dotenv support

### 6. Error Handling
- Environment validation with dotenv support
- Connection error handling
- API error handling
- Invalid input validation

### 7. Performance Monitoring
- Response time tracking
- Token usage monitoring
- Success/failure rates
- Detailed error reporting

## Environment Variables Required

- `INFERENCE_URL` - Your Heroku Inference API URL
- `INFERENCE_KEY` - Your Heroku Inference API key
- `INFERENCE_MODEL_ID` - Your model ID

## Dependencies

Core dependencies (should be installed with langchain-heroku):
- `langchain-core>=0.1.0`
- `httpx>=0.24.0`

Additional test harness dependencies:
- `pyyaml>=6.0`
- `pytest>=7.0.0`
- `pytest-mock>=3.10.0`
- **`python-dotenv>=1.0.0`**

## Output Examples

### Console Output with dotenv
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
```

### JSON Results
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
      "response": "Hello! I'm doing well...",
      "metadata": {
        "input_tokens": 5,
        "output_tokens": 12,
        "total_tokens": 17
      }
    }
  ]
}
```

## Integration with Existing Codebase

The test harness is designed to integrate seamlessly with the existing langchain-heroku codebase:

1. **Follows existing patterns** - Uses the same import structure and coding style
2. **Compatible with existing tests** - Can be run alongside existing unit tests
3. **Environment variable support** - Uses the same environment variable pattern as MiaChat
4. **Error handling** - Follows the same error handling patterns as the main codebase
5. **Dotenv integration** - Adds convenient environment variable management

## Next Steps

1. **Install dependencies**: `pip install -r test_requirements.txt`
2. **Set up environment**: 
   - Copy `.env.example` to `.env` and update with your values, OR
   - Set system environment variables
3. **Run tests**: Use any of the provided methods to run tests
4. **Customize**: Modify `test_config.yaml` for your specific needs
5. **Integrate**: Add to CI/CD pipeline using the integration tests

## New Features in This Update

### 🔧 **Dotenv Support**
- **Automatic .env file loading** - No more manual environment variable setup
- **Multiple .env file support** - `.env`, `.env.local`, `.env.production`
- **Graceful fallback** - Falls back to system environment variables if .env not found
- **Helpful error messages** - Clear guidance when environment variables are missing

### 🚀 **Enhanced Shell Script**
- **Dotenv file support** - `./run_tests.sh --env .env basic`
- **Better argument parsing** - More flexible command-line interface
- **Improved error handling** - Better validation and error messages

### 📝 **Updated Documentation**
- **Comprehensive dotenv guide** - Complete examples and troubleshooting
- **Multiple usage patterns** - Both .env and system environment variable examples
- **Enhanced troubleshooting** - Solutions for common dotenv issues

### 🧪 **Integration Test Updates**
- **Automatic dotenv loading** - Tests automatically load .env files
- **Dotenv validation test** - Ensures environment variables are properly loaded
- **Better error handling** - Graceful handling of missing .env files

The test harness now provides a **production-ready solution** for validating MiaChat against real endpoints, with **comprehensive coverage**, **detailed reporting**, **multiple usage patterns**, and **convenient environment variable management** through dotenv support. 