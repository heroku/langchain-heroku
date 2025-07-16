#!/usr/bin/env python3
"""
Test Harness for MiaChat Production Endpoint

This script provides a comprehensive test suite for testing the MiaChat model
against the production Heroku Inference API endpoint.

Usage:
    python test_harness.py [--config config.yaml] [--test test_name] [--env .env]

Environment Variables Required:
    - INFERENCE_URL: Your Heroku Inference API URL
    - INFERENCE_KEY: Your Heroku Inference API key
    - INFERENCE_MODEL_ID: Your model ID

Example:
    # Using .env file
    python test_harness.py --env .env
    
    # Using environment variables directly
    export INFERENCE_URL="https://your-inference-api-url"
    export INFERENCE_KEY="your-heroku-inference-api-key"
    export INFERENCE_MODEL_ID="your-model-id"
    python test_harness.py
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

# Load dotenv if available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_heroku.chat_models import MiaChat


@dataclass
class MiaChatTestResult:
    """Represents the result of a single test."""
    name: str
    success: bool
    duration: float
    error: Optional[str] = None
    response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MiaChatTestHarness:
    """Test harness for MiaChat production endpoint testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, env_file: Optional[str] = None):
        self.config = config or {}
        self.results: List[MiaChatTestResult] = []
        self.chat_model = None
        self._load_environment(env_file)
        self._validate_environment()
        self._initialize_chat_model()
    
    def _load_environment(self, env_file: Optional[str] = None):
        """Load environment variables from .env file if available."""
        if not DOTENV_AVAILABLE:
            print("⚠️  python-dotenv not available. Install with: pip install python-dotenv")
            print("   Using system environment variables only.")
            return
        
        # Try to load .env file
        env_files_to_try = []
        if env_file:
            env_files_to_try.append(env_file)
        else:
            # Try common .env file names
            env_files_to_try.extend([".env", ".env.local", ".env.production"])
        
        loaded_env_file = None
        for env_file_path in env_files_to_try:
            if os.path.exists(env_file_path):
                load_dotenv(env_file_path)
                loaded_env_file = env_file_path
                print(f"✅ Loaded environment from: {env_file_path}")
                break
        
        if not loaded_env_file:
            print("ℹ️  No .env file found. Using system environment variables.")
    
    def _validate_environment(self):
        """Validate that required environment variables are set."""
        required_vars = ["INFERENCE_URL", "INFERENCE_KEY", "INFERENCE_MODEL_ID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("Please set the following environment variables:")
            for var in missing_vars:
                print(f"  export {var}='your-value'")
            print("\nOr create a .env file with:")
            for var in missing_vars:
                print(f"  {var}=your-value")
            print("\nExample .env file:")
            print("  INFERENCE_URL=https://your-inference-api-url")
            print("  INFERENCE_KEY=your-heroku-inference-api-key")
            print("  INFERENCE_MODEL_ID=your-model-id")
            sys.exit(1)
        
        print("✅ Environment variables validated")
    
    def _initialize_chat_model(self):
        """Initialize the MiaChat model with configuration."""
        model_config = self.config.get("model", {})
        
        self.chat_model = MiaChat(
            model=os.getenv("INFERENCE_MODEL_ID"),
            api_key=os.getenv("INFERENCE_KEY"),
            inference_url=os.getenv("INFERENCE_URL"),
            temperature=model_config.get("temperature", 0.1),
            max_tokens=model_config.get("max_tokens", 256),
            timeout=model_config.get("timeout", 30),
            streaming=model_config.get("streaming", False),
            top_p=model_config.get("top_p", 0.95),
        )
        print("✅ MiaChat model initialized")
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> MiaChatTestResult:
        """Run a single test and return the result."""
        start_time = time.time()
        success = False
        error = None
        response = None
        metadata = None
        
        try:
            print(f"🧪 Running test: {test_name}")
            response, metadata = test_func(*args, **kwargs)
            success = True
            print(f"✅ Test passed: {test_name}")
        except Exception as e:
            error = str(e)
            print(f"❌ Test failed: {test_name}")
            print(f"   Error: {error}")
            traceback.print_exc()
        
        duration = time.time() - start_time
        result = MiaChatTestResult(
            name=test_name,
            success=success,
            duration=duration,
            error=error,
            response=response,
            metadata=metadata
        )
        self.results.append(result)
        return result
    
    def test_basic_conversation(self) -> tuple[str, Dict[str, Any]]:
        """Test basic conversation functionality."""
        messages = [HumanMessage(content="Hello! How are you today?")]
        result = self.chat_model.invoke(messages)
        return result.content, {
            "input_tokens": result.usage_metadata.get("input_tokens"),
            "output_tokens": result.usage_metadata.get("output_tokens"),
            "total_tokens": result.usage_metadata.get("total_tokens"),
        }
    
    def test_system_message(self) -> tuple[str, Dict[str, Any]]:
        """Test conversation with system message."""
        messages = [
            SystemMessage(content="You are a helpful assistant that speaks in a friendly tone."),
            HumanMessage(content="What's the weather like?")
        ]
        result = self.chat_model.invoke(messages)
        return result.content, {
            "input_tokens": result.usage_metadata.get("input_tokens"),
            "output_tokens": result.usage_metadata.get("output_tokens"),
            "total_tokens": result.usage_metadata.get("total_tokens"),
        }
    
    def test_multi_turn_conversation(self) -> tuple[str, Dict[str, Any]]:
        """Test multi-turn conversation."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is 2 + 2?"),
            AIMessage(content="2 + 2 equals 4."),
            HumanMessage(content="What about 3 + 3?")
        ]
        result = self.chat_model.invoke(messages)
        return result.content, {
            "input_tokens": result.usage_metadata.get("input_tokens"),
            "output_tokens": result.usage_metadata.get("output_tokens"),
            "total_tokens": result.usage_metadata.get("total_tokens"),
        }
    
    def test_string_input(self) -> tuple[str, Dict[str, Any]]:
        """Test direct string input."""
        result = self.chat_model.invoke("Tell me a short joke.")
        return result.content, {
            "input_tokens": result.usage_metadata.get("input_tokens"),
            "output_tokens": result.usage_metadata.get("output_tokens"),
            "total_tokens": result.usage_metadata.get("total_tokens"),
        }
    
    def test_tool_messages(self) -> tuple[str, Dict[str, Any]]:
        """Test conversation with tool messages."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather like in New York?"),
            AIMessage(content="I don't have access to real-time weather data."),
            ToolMessage(content="The weather in New York is sunny with a temperature of 75°F", tool_call_id="call_123"),
            HumanMessage(content="Great! What should I wear?")
        ]
        result = self.chat_model.invoke(messages)
        return result.content, {
            "input_tokens": result.usage_metadata.get("input_tokens"),
            "output_tokens": result.usage_metadata.get("output_tokens"),
            "total_tokens": result.usage_metadata.get("total_tokens"),
        }
    
    def test_function_messages(self) -> tuple[str, Dict[str, Any]]:
        """Test conversation with function messages."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Get the current time."),
            FunctionMessage(content="The current time is 2:30 PM", name="get_time"),
            HumanMessage(content="What time is it?")
        ]
        result = self.chat_model.invoke(messages)
        return result.content, {
            "input_tokens": result.usage_metadata.get("input_tokens"),
            "output_tokens": result.usage_metadata.get("output_tokens"),
            "total_tokens": result.usage_metadata.get("total_tokens"),
        }
    
    def test_streaming(self) -> tuple[str, Dict[str, Any]]:
        """Test streaming functionality."""
        # Create a streaming version of the model
        streaming_model = MiaChat(
            model=os.getenv("INFERENCE_MODEL_ID"),
            api_key=os.getenv("INFERENCE_KEY"),
            inference_url=os.getenv("INFERENCE_URL"),
            streaming=True,
            temperature=0.1,
        )
        
        messages = [HumanMessage(content="Write a short story about a cat.")]
        full_response = ""
        
        for chunk in streaming_model.stream(messages):
            if chunk.message.content:
                full_response += chunk.message.content
        
        return full_response, {"streaming": True}
    
    def test_temperature_variation(self) -> tuple[str, Dict[str, Any]]:
        """Test different temperature settings."""
        # Test with low temperature (deterministic)
        low_temp_model = MiaChat(
            model=os.getenv("INFERENCE_MODEL_ID"),
            api_key=os.getenv("INFERENCE_KEY"),
            inference_url=os.getenv("INFERENCE_URL"),
            temperature=0.0,
        )
        
        result_low = low_temp_model.invoke("What is the capital of France?")
        
        # Test with high temperature (creative)
        high_temp_model = MiaChat(
            model=os.getenv("INFERENCE_MODEL_ID"),
            api_key=os.getenv("INFERENCE_KEY"),
            inference_url=os.getenv("INFERENCE_URL"),
            temperature=0.9,
        )
        
        result_high = high_temp_model.invoke("What is the capital of France?")
        
        return f"Low temp: {result_low.content}\nHigh temp: {result_high.content}", {
            "low_temperature_response": result_low.content,
            "high_temperature_response": result_high.content,
        }
    
    def test_max_tokens(self) -> tuple[str, Dict[str, Any]]:
        """Test max_tokens parameter."""
        short_model = MiaChat(
            model=os.getenv("INFERENCE_MODEL_ID"),
            api_key=os.getenv("INFERENCE_KEY"),
            inference_url=os.getenv("INFERENCE_URL"),
            max_tokens=10,
        )
        
        result = short_model.invoke("Write a detailed explanation of quantum physics.")
        return result.content, {
            "max_tokens": 10,
            "response_length": len(result.content),
        }
    
    def test_stop_sequences(self) -> tuple[str, Dict[str, Any]]:
        """Test stop sequences."""
        stop_model = MiaChat(
            model=os.getenv("INFERENCE_MODEL_ID"),
            api_key=os.getenv("INFERENCE_KEY"),
            inference_url=os.getenv("INFERENCE_URL"),
            stop=["\n\n", "END"],
        )
        
        result = stop_model.invoke("Write a paragraph about dogs. END")
        return result.content, {
            "stop_sequences": ["\n\n", "END"],
        }
    
    def test_error_handling(self) -> tuple[str, Dict[str, Any]]:
        """Test error handling with invalid input."""
        try:
            # This should raise an error
            self.chat_model.invoke(12345)
            return "Unexpected success", {"error": "Expected ValueError"}
        except ValueError as e:
            return str(e), {"expected_error": "ValueError"}
    
    def run_all_tests(self) -> List[MiaChatTestResult]:
        """Run all available tests and return the results."""
        test_methods = [
            ("basic", self.test_basic_conversation),
            ("system", self.test_system_message),
            ("multi_turn", self.test_multi_turn_conversation),
            ("string", self.test_string_input),
            ("tools", self.test_tool_messages),
            ("functions", self.test_function_messages),
            ("streaming", self.test_streaming),
            ("temperature", self.test_temperature_variation),
            ("max_tokens", self.test_max_tokens),
            ("stop_sequences", self.test_stop_sequences),
            ("error_handling", self.test_error_handling),
        ]
        results = []
        for name, method in test_methods:
            result = self.run_test(name, method)
            results.append(result)
        return results
    
    def run_specific_test(self, test_name: str) -> MiaChatTestResult:
        """Run a specific test by name and return the result."""
        test_map = {
            "basic": self.test_basic_conversation,
            "system": self.test_system_message,
            "multi_turn": self.test_multi_turn_conversation,
            "string": self.test_string_input,
            "tools": self.test_tool_messages,
            "functions": self.test_function_messages,
            "streaming": self.test_streaming,
            "temperature": self.test_temperature_variation,
            "max_tokens": self.test_max_tokens,
            "stop_sequences": self.test_stop_sequences,
            "error_handling": self.test_error_handling,
        }
        if test_name not in test_map:
            raise ValueError(f"Unknown test: {test_name}")
        return self.run_test(test_name, test_map[test_name])
    
    def print_summary(self):
        """Print a summary of test results."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        total_duration = sum(r.duration for r in self.results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Average Duration: {total_duration/total_tests:.2f}s")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.name}: {result.error}")
        
        print("\n✅ PASSED TESTS:")
        for result in self.results:
            if result.success:
                print(f"  - {result.name} ({result.duration:.2f}s)")
                if result.metadata:
                    print(f"    Metadata: {result.metadata}")
    
    def save_results(self, filename: str = None):
        """Save test results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        results_data = []
        for result in self.results:
            results_data.append({
                "name": result.name,
                "success": result.success,
                "duration": result.duration,
                "error": result.error,
                "response": result.response,
                "metadata": result.metadata,
            })
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "environment": {
                    "inference_url": os.getenv("INFERENCE_URL"),
                    "model_id": os.getenv("INFERENCE_MODEL_ID"),
                },
                "results": results_data
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for the test harness."""
    parser = argparse.ArgumentParser(description="Test Harness for MiaChat Production Endpoint")
    parser.add_argument("--config", help="Path to configuration YAML file")
    parser.add_argument("--test", help="Run specific test (basic, system, multi_turn, etc.)")
    parser.add_argument("--save-results", help="Save results to specified file")
    parser.add_argument("--list-tests", action="store_true", help="List available tests")
    parser.add_argument("--env", help="Path to .env file (default: .env, .env.local, .env.production)")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
        print(f"✅ Loaded configuration from: {args.config}")
    
    # List available tests
    if args.list_tests:
        print("Available tests:")
        tests = [
            "basic", "system", "multi_turn", "string", "tool", 
            "function", "streaming", "temperature", "max_tokens", 
            "stop", "error"
        ]
        for test in tests:
            print(f"  - {test}")
        return
    
    # Initialize test harness
    harness = MiaChatTestHarness(config, args.env)
    
    try:
        if args.test:
            # Run specific test
            result = harness.run_specific_test(args.test)
            print(f"\nTest '{args.test}' completed in {result.duration:.2f}s")
            if result.success:
                print(f"Response: {result.response}")
                if result.metadata:
                    print(f"Metadata: {result.metadata}")
            else:
                print(f"Error: {result.error}")
        else:
            # Run all tests
            harness.run_all_tests()
            harness.print_summary()
        
        # Save results if requested
        if args.save_results:
            harness.save_results(args.save_results)
        elif not args.test:  # Auto-save for full test suite
            harness.save_results()
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test harness failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 