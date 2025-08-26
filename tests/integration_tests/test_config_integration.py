"""Integration tests for HerokuConfig following LangChain testing standards.

This module contains integration tests that validate the HerokuConfig component
and its integration with other Heroku components, following LangChain's testing guidelines.

Key aspects:
- Tests configuration validation logic
- Tests environment variable loading scenarios  
- Tests configuration inheritance between components
- Includes comprehensive error scenario testing
- Uses environment variable configuration with graceful skipping
"""

import os
import tempfile
from typing import Any, Dict, Generator
from unittest.mock import patch

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
from langchain_heroku.config import HerokuConfig
from langchain_heroku.embeddings import HerokuEmbeddings


@pytest.mark.integration
class TestHerokuConfigIntegration:
    """Integration tests for HerokuConfig component."""

    @pytest.fixture(autouse=True)
    def setup_environment(self) -> None:
        """Setup environment variables for testing."""
        # Load .env file if dotenv is available
        if DOTENV_AVAILABLE:
            for env_file in [".env", ".env.local", ".env.integration"]:
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    break

    def test_config_validation_with_valid_env_vars(self) -> None:
        """Test configuration validation with valid environment variables."""
        # Check if we have valid environment variables
        inference_url = os.getenv("INFERENCE_URL")
        api_key = os.getenv("INFERENCE_KEY")
        model_id = os.getenv("INFERENCE_MODEL_ID")

        if not all([inference_url, api_key, model_id]):
            pytest.skip("Config validation tests require INFERENCE_URL, INFERENCE_KEY, and INFERENCE_MODEL_ID environment variables")

        # Test that validation passes with valid config
        try:
            HerokuConfig.validate_config(
                inference_url=inference_url,
                api_key=api_key,
                model_id=model_id
            )
        except Exception as e:
            pytest.fail(f"Valid configuration should not raise an exception: {e}")

    def test_config_validation_with_invalid_values(self) -> None:
        """Test configuration validation with invalid values."""
        # Test with None values
        with pytest.raises(ValueError, match="api_key is required"):
            HerokuConfig.validate_config(
                inference_url="https://valid.url",
                api_key=None,
                model_id="valid-model"
            )

        with pytest.raises(ValueError, match="inference_url is required"):
            HerokuConfig.validate_config(
                inference_url=None,
                api_key="valid-key",
                model_id="valid-model"
            )

        with pytest.raises(ValueError, match="model_id is required"):
            HerokuConfig.validate_config(
                inference_url="https://valid.url",
                api_key="valid-key",
                model_id=None
            )

        # Test with empty strings
        with pytest.raises(ValueError, match="api_key is required"):
            HerokuConfig.validate_config(
                inference_url="https://valid.url",
                api_key="",
                model_id="valid-model"
            )

        with pytest.raises(ValueError, match="inference_url is required"):
            HerokuConfig.validate_config(
                inference_url="",
                api_key="valid-key",
                model_id="valid-model"
            )

        with pytest.raises(ValueError, match="model_id is required"):
            HerokuConfig.validate_config(
                inference_url="https://valid.url",
                api_key="valid-key",
                model_id=""
            )

    def test_config_validation_with_invalid_url_format(self) -> None:
        """Test configuration validation with invalid URL formats."""
        with pytest.raises(ValueError, match="inference_url must be a valid URL"):
            HerokuConfig.validate_config(
                inference_url="not-a-url",
                api_key="valid-key",
                model_id="valid-model"
            )

        with pytest.raises(ValueError, match="inference_url must be a valid URL"):
            HerokuConfig.validate_config(
                inference_url="ftp://invalid-protocol.com",
                api_key="valid-key",
                model_id="valid-model"
            )

    def test_env_variable_loading(self) -> None:
        """Test environment variable loading functionality."""
        # Test with existing environment variables
        actual_url = os.getenv("INFERENCE_URL")
        actual_key = os.getenv("INFERENCE_KEY")
        actual_model = os.getenv("INFERENCE_MODEL_ID")

        # Test get_env method
        loaded_url = HerokuConfig.get_env("INFERENCE_URL")
        loaded_key = HerokuConfig.get_env("INFERENCE_KEY")
        loaded_model = HerokuConfig.get_env("INFERENCE_MODEL_ID")

        assert loaded_url == actual_url
        assert loaded_key == actual_key
        assert loaded_model == actual_model

        # Test with default values
        non_existent_var = HerokuConfig.get_env("NON_EXISTENT_VAR", "default_value")
        assert non_existent_var == "default_value"

        # Test with None default (should return None)
        non_existent_var_none = HerokuConfig.get_env("NON_EXISTENT_VAR")
        assert non_existent_var_none is None

    @pytest.fixture
    def temporary_env_vars(self) -> Generator[Dict[str, str], None, None]:
        """Fixture to temporarily set environment variables for testing."""
        temp_vars = {
            "TEST_INFERENCE_URL": "https://test.heroku.com",
            "TEST_INFERENCE_KEY": "test-key-12345",
            "TEST_INFERENCE_MODEL_ID": "test-model-v1"
        }

        # Set temporary environment variables
        original_values = {}
        for key, value in temp_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value

        yield temp_vars

        # Restore original environment variables
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

    def test_config_inheritance_between_components(self, temporary_env_vars: Dict[str, str]) -> None:
        """Test that configuration is properly inherited between components."""
        # Test with components using environment variables
        with patch.dict("os.environ", temporary_env_vars):
            # Create components that should inherit from environment
            chat = ChatHeroku()
            embeddings = HerokuEmbeddings()

            # Both should have loaded the same configuration values
            assert chat.inference_url == temporary_env_vars["TEST_INFERENCE_URL"]
            assert chat.api_key == temporary_env_vars["TEST_INFERENCE_KEY"]
            assert chat.model == temporary_env_vars["TEST_INFERENCE_MODEL_ID"]

            assert embeddings.inference_url == temporary_env_vars["TEST_INFERENCE_URL"]
            assert embeddings.api_key == temporary_env_vars["TEST_INFERENCE_KEY"]
            assert embeddings.model == temporary_env_vars["TEST_INFERENCE_MODEL_ID"]

    def test_explicit_config_override(self, temporary_env_vars: Dict[str, str]) -> None:
        """Test that explicit configuration overrides environment variables."""
        with patch.dict("os.environ", temporary_env_vars):
            # Create component with explicit configuration
            explicit_config = {
                "inference_url": "https://explicit.url",
                "api_key": "explicit-key",
                "model": "explicit-model"
            }

            chat = ChatHeroku(**explicit_config)  # type: ignore[arg-type]

            # Should use explicit values, not environment variables
            assert chat.inference_url == "https://explicit.url"
            assert chat.api_key == "explicit-key"
            assert chat.model == "explicit-model"

            # Should not match environment variables
            assert chat.inference_url != temporary_env_vars["TEST_INFERENCE_URL"]
            assert chat.api_key != temporary_env_vars["TEST_INFERENCE_KEY"]
            assert chat.model != temporary_env_vars["TEST_INFERENCE_MODEL_ID"]

    def test_partial_config_override(self, temporary_env_vars: Dict[str, str]) -> None:
        """Test partial configuration override (some explicit, some from env)."""
        with patch.dict("os.environ", temporary_env_vars):
            # Create component with partial explicit configuration
            chat = ChatHeroku(
                api_key="explicit-key",
                # inference_url and model should come from env vars
            )

            # Should use explicit api_key
            assert chat.api_key == "explicit-key"

            # Should use environment variables for the rest
            assert chat.inference_url == temporary_env_vars["TEST_INFERENCE_URL"]
            assert chat.model == temporary_env_vars["TEST_INFERENCE_MODEL_ID"]

    def test_config_validation_integration_with_components(self) -> None:
        """Test that components properly validate configuration during initialization."""
        # Test with invalid URL
        with pytest.raises(ValueError):
            ChatHeroku(
                inference_url="invalid-url",
                api_key="valid-key",
                model="valid-model"
            )

        with pytest.raises(ValueError):
            HerokuEmbeddings(
                inference_url="invalid-url",
                api_key="valid-key",
                model="valid-model"
            )

        # Test with missing required config
        with pytest.raises(ValueError):
            ChatHeroku(
                inference_url="https://valid.url",
                api_key="",  # Empty key should fail
                model="valid-model"
            )

    def test_config_loading_from_dotenv_file(self) -> None:
        """Test configuration loading from .env file."""
        if not DOTENV_AVAILABLE:
            pytest.skip("dotenv not available for testing")

        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as temp_env:
            temp_env.write("TEST_DOTENV_URL=https://dotenv.test.com\n")
            temp_env.write("TEST_DOTENV_KEY=dotenv-test-key\n")
            temp_env.write("TEST_DOTENV_MODEL=dotenv-test-model\n")
            temp_env_path = temp_env.name

        try:
            # Load the temporary .env file
            load_dotenv(temp_env_path)

            # Verify environment variables were loaded
            assert os.getenv("TEST_DOTENV_URL") == "https://dotenv.test.com"
            assert os.getenv("TEST_DOTENV_KEY") == "dotenv-test-key" 
            assert os.getenv("TEST_DOTENV_MODEL") == "dotenv-test-model"

            # Test HerokuConfig can access these values
            loaded_url = HerokuConfig.get_env("TEST_DOTENV_URL")
            loaded_key = HerokuConfig.get_env("TEST_DOTENV_KEY")
            loaded_model = HerokuConfig.get_env("TEST_DOTENV_MODEL")

            assert loaded_url == "https://dotenv.test.com"
            assert loaded_key == "dotenv-test-key"
            assert loaded_model == "dotenv-test-model"

        finally:
            # Clean up
            os.unlink(temp_env_path)
            # Remove from environment
            for key in ["TEST_DOTENV_URL", "TEST_DOTENV_KEY", "TEST_DOTENV_MODEL"]:
                os.environ.pop(key, None)

    def test_config_precedence_order(self, temporary_env_vars: Dict[str, str]) -> None:
        """Test configuration precedence: explicit params > env vars > defaults."""
        with patch.dict("os.environ", temporary_env_vars):
            # Test explicit parameters take precedence over environment variables
            chat = ChatHeroku(
                inference_url="https://explicit.com",
                # api_key and model should come from env vars
            )

            assert chat.inference_url == "https://explicit.com"  # Explicit wins
            assert chat.api_key == temporary_env_vars["TEST_INFERENCE_KEY"]  # From env
            assert chat.model == temporary_env_vars["TEST_INFERENCE_MODEL_ID"]  # From env

    def test_config_error_messages(self) -> None:
        """Test that configuration errors provide helpful error messages."""
        try:
            HerokuConfig.validate_config(
                inference_url=None,
                api_key=None,
                model_id=None
            )
        except ValueError as e:
            error_msg = str(e)
            # Should mention what's missing
            assert any(
                required_field in error_msg.lower() 
                for required_field in ["api_key", "inference_url", "model_id"]
            )

        try:
            HerokuConfig.validate_config(
                inference_url="not-a-url",
                api_key="valid-key",
                model_id="valid-model"
            )
        except ValueError as e:
            error_msg = str(e).lower()
            assert "url" in error_msg
            assert "valid" in error_msg

    def test_config_validation_with_real_components(self) -> None:
        """Test configuration validation with real component initialization."""
        # Skip if we don't have valid environment variables
        if not all([os.getenv("INFERENCE_URL"), os.getenv("INFERENCE_KEY"), os.getenv("INFERENCE_MODEL_ID")]):
            pytest.skip("Real component testing requires valid environment variables")

        # Test that components can be created with valid environment configuration
        try:
            chat = ChatHeroku()
            embeddings = HerokuEmbeddings()

            # Validate that internal configuration validation was called
            # (This is implicit - if it gets this far, validation passed)
            assert chat.inference_url is not None
            assert chat.api_key is not None
            assert chat.model is not None

            assert embeddings.inference_url is not None
            assert embeddings.api_key is not None
            assert embeddings.model is not None

        except Exception as e:
            pytest.fail(f"Components should initialize successfully with valid env vars: {e}")

    def test_config_validation_thread_safety(self, temporary_env_vars: Dict[str, str]) -> None:
        """Test that configuration validation is thread-safe."""
        import concurrent.futures

        results = []
        errors = []

        def validate_config() -> None:
            try:
                with patch.dict("os.environ", temporary_env_vars):
                    HerokuConfig.validate_config(
                        inference_url=temporary_env_vars["TEST_INFERENCE_URL"],
                        api_key=temporary_env_vars["TEST_INFERENCE_KEY"],
                        model_id=temporary_env_vars["TEST_INFERENCE_MODEL_ID"]
                    )
                    results.append("success")
            except Exception as e:
                errors.append(str(e))

        # Run validation concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(validate_config) for _ in range(10)]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Wait for completion

        # All validations should succeed
        assert len(results) == 10
        assert len(errors) == 0
        assert all(result == "success" for result in results)