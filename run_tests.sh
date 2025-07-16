#!/bin/bash

# Test Harness Runner Script
# This script provides easy access to common test harness operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to load environment from .env file
load_env_file() {
    local env_file="$1"
    
    if [ -f "$env_file" ]; then
        print_status "Loading environment from: $env_file"
        # Export variables from .env file
        while IFS= read -r line; do
            # Skip comments and empty lines
            if [[ ! "$line" =~ ^[[:space:]]*# ]] && [[ -n "$line" ]]; then
                # Export the variable
                export "$line"
            fi
        done < "$env_file"
        print_success "Environment loaded from $env_file"
        return 0
    else
        print_warning "Environment file not found: $env_file"
        return 1
    fi
}

# Function to check if environment variables are set
check_environment() {
    print_status "Checking environment variables..."
    
    local missing_vars=()
    
    if [ -z "$INFERENCE_URL" ]; then
        missing_vars+=("INFERENCE_URL")
    fi
    
    if [ -z "$INFERENCE_KEY" ]; then
        missing_vars+=("INFERENCE_KEY")
    fi
    
    if [ -z "$INFERENCE_MODEL_ID" ]; then
        missing_vars+=("INFERENCE_MODEL_ID")
    fi
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_error "Missing required environment variables: ${missing_vars[*]}"
        echo
        echo "Please set the following environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  export $var='your-value'"
        done
        echo
        echo "Or create a .env file with:"
        for var in "${missing_vars[@]}"; do
            echo "  $var=your-value"
        done
        echo
        echo "Example .env file:"
        echo "  INFERENCE_URL=https://your-inference-api-url"
        echo "  INFERENCE_KEY=your-heroku-inference-api-key"
        echo "  INFERENCE_MODEL_ID=your-model-id"
        echo
        echo "Then run: $0 --env .env [test_name]"
        exit 1
    fi
    
    print_success "Environment variables validated"
}

# Function to show usage
show_usage() {
    echo "MiaChat Test Harness Runner"
    echo "=========================="
    echo
    echo "Usage: $0 [OPTION] [TEST_NAME]"
    echo
    echo "Options:"
    echo "  --env FILE           Load environment from .env file"
    echo "  --help, -h           Show this help message"
    echo "  list                 List all available tests"
    echo
    echo "Test Names:"
    echo "  all                  Run all tests (default)"
    echo "  basic                Run basic conversation test"
    echo "  system               Run system message test"
    echo "  multi_turn           Run multi-turn conversation test"
    echo "  string               Run string input test"
    echo "  tool                 Run tool messages test"
    echo "  function             Run function messages test"
    echo "  streaming            Run streaming test"
    echo "  temperature          Run temperature variation test"
    echo "  max_tokens           Run max tokens test"
    echo "  stop                 Run stop sequences test"
    echo "  error                Run error handling test"
    echo
    echo "Examples:"
    echo "  $0                           # Run all tests with system env vars"
    echo "  $0 --env .env               # Run all tests with .env file"
    echo "  $0 --env .env basic         # Run basic test with .env file"
    echo "  $0 basic                    # Run basic test with system env vars"
    echo "  $0 list                     # List available tests"
    echo
    echo "Environment Variables:"
    echo "  INFERENCE_URL         Your Heroku Inference API URL"
    echo "  INFERENCE_KEY         Your Heroku Inference API key"
    echo "  INFERENCE_MODEL_ID    Your model ID"
    echo
    echo "Environment File (.env):"
    echo "  INFERENCE_URL=https://your-inference-api-url"
    echo "  INFERENCE_KEY=your-heroku-inference-api-key"
    echo "  INFERENCE_MODEL_ID=your-model-id"
}

# Function to run tests
run_tests() {
    local test_name="$1"
    local env_file="$2"
    
    # Load environment file if specified
    if [ -n "$env_file" ]; then
        load_env_file "$env_file"
    fi
    
    # Check environment variables
    check_environment
    
    if [ -z "$test_name" ]; then
        print_status "Running all tests..."
        if [ -n "$env_file" ]; then
            python test_harness.py --env "$env_file"
        else
            python test_harness.py
        fi
    else
        print_status "Running test: $test_name"
        if [ -n "$env_file" ]; then
            python test_harness.py --env "$env_file" --test "$test_name"
        else
            python test_harness.py --test "$test_name"
        fi
    fi
}

# Main script logic
main() {
    local env_file=""
    local test_name="all"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                env_file="$2"
                shift 2
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            list)
                print_status "Listing available tests..."
                python test_harness.py --list-tests
                exit 0
                ;;
            all|basic|system|multi_turn|string|tool|function|streaming|temperature|max_tokens|stop|error)
                test_name="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                echo
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Run tests
    run_tests "$test_name" "$env_file"
}

# Run main function with all arguments
main "$@" 