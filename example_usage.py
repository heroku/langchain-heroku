#!/usr/bin/env python3
"""
Example usage of the MiaChat test harness.

This script demonstrates how to use the test harness programmatically
and shows different ways to run tests with dotenv support.
"""

import os
import sys
from test_harness import TestHarness


def example_basic_usage():
    """Example of basic test harness usage."""
    print("=== Basic Test Harness Usage ===")
    
    # Initialize the test harness
    harness = TestHarness()
    
    # Run a specific test
    print("\n1. Running a specific test:")
    result = harness.run_specific_test("basic")
    print(f"   Test: {result.name}")
    print(f"   Success: {result.success}")
    print(f"   Duration: {result.duration:.2f}s")
    if result.success:
        print(f"   Response: {result.response[:100]}...")
    
    # Run all tests
    print("\n2. Running all tests:")
    results = harness.run_all_tests()
    print(f"   Total tests run: {len(results)}")
    print(f"   Passed: {sum(1 for r in results if r.success)}")
    print(f"   Failed: {sum(1 for r in results if not r.success)}")
    
    # Print summary
    harness.print_summary()
    
    # Save results
    harness.save_results("example_results.json")


def example_dotenv_usage():
    """Example with dotenv file usage."""
    print("\n=== Dotenv Usage Example ===")
    
    # Initialize with .env file
    harness = TestHarness(env_file=".env")
    
    # Run a few specific tests
    tests_to_run = ["basic", "system", "string"]
    for test_name in tests_to_run:
        result = harness.run_specific_test(test_name)
        print(f"   {test_name}: {'✅' if result.success else '❌'} ({result.duration:.2f}s)")
    
    harness.print_summary()


def example_custom_config():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Custom configuration
    config = {
        "model": {
            "temperature": 0.0,  # More deterministic
            "max_tokens": 100,   # Shorter responses
            "timeout": 30,       # Shorter timeout
        }
    }
    
    # Initialize with custom config and .env file
    harness = TestHarness(config, env_file=".env")
    
    # Run a few specific tests
    tests_to_run = ["basic", "system", "string"]
    for test_name in tests_to_run:
        result = harness.run_specific_test(test_name)
        print(f"   {test_name}: {'✅' if result.success else '❌'} ({result.duration:.2f}s)")
    
    harness.print_summary()


def example_error_handling():
    """Example of error handling."""
    print("\n=== Error Handling Example ===")
    
    # Test with missing environment variables
    original_url = os.environ.get("INFERENCE_URL")
    if original_url:
        # Temporarily remove the URL to test error handling
        del os.environ["INFERENCE_URL"]
        
        try:
            harness = TestHarness()
        except SystemExit:
            print("   ✅ Correctly caught missing environment variable")
        finally:
            # Restore the environment
            if original_url:
                os.environ["INFERENCE_URL"] = original_url


def example_streaming_test():
    """Example of streaming test."""
    print("\n=== Streaming Test Example ===")
    
    harness = TestHarness(env_file=".env")
    
    # Run streaming test specifically
    result = harness.run_specific_test("streaming")
    print(f"   Streaming test: {'✅' if result.success else '❌'}")
    if result.success:
        print(f"   Response length: {len(result.response)} characters")
        print(f"   First 100 chars: {result.response[:100]}...")


def example_create_env_file():
    """Example of creating a .env file."""
    print("\n=== Creating .env File Example ===")
    
    env_content = """# MiaChat Test Harness Environment Variables
# Replace these with your actual values

INFERENCE_URL=https://your-inference-api-url
INFERENCE_KEY=your-heroku-inference-api-key
INFERENCE_MODEL_ID=your-model-id

# Optional: Additional configuration
# TEMPERATURE=0.1
# MAX_TOKENS=256
# TIMEOUT=60
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    
    print("   Created .env.example file")
    print("   Copy this file to .env and update with your values:")
    print("     cp .env.example .env")
    print("     # Edit .env with your actual values")


def main():
    """Main example function."""
    print("MiaChat Test Harness Examples")
    print("=" * 50)
    
    # Check if environment variables are set
    required_vars = ["INFERENCE_URL", "INFERENCE_KEY", "INFERENCE_MODEL_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set the required environment variables before running examples:")
        for var in missing_vars:
            print(f"   export {var}='your-value'")
        print("\nOr create a .env file:")
        example_create_env_file()
        return
    
    try:
        # Run examples
        example_basic_usage()
        example_dotenv_usage()
        example_custom_config()
        example_streaming_test()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 