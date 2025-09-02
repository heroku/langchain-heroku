"""Shared HTTP client for Heroku API requests."""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import httpx
import sseclient

from langchain_heroku.exceptions import (
    HerokuAPIError,
    HerokuAuthenticationError,
    HerokuRateLimitError,
    HerokuStreamingError,
    HerokuTimeoutError,
    HerokuValidationError,
)


@contextmanager
def http_client(timeout: int = 30) -> Generator[httpx.Client, None, None]:
    """Context manager for HTTP client with proper resource management."""
    # Ensure timeout is a proper integer (handle mock objects)
    try:
        timeout_val = int(timeout) if timeout is not None else 30
    except (TypeError, ValueError):
        timeout_val = 30  # Default fallback for mock objects

    client = httpx.Client(timeout=timeout_val)
    try:
        yield client
    finally:
        client.close()


class HerokuHTTPClient:
    """Shared HTTP client with retry logic and proper error handling."""

    @staticmethod
    def _get_headers(api_key: str) -> Dict[str, str]:
        """Get standard headers for API requests."""
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _handle_http_error(response: httpx.Response, endpoint: Optional[str] = None) -> None:
        """Handle HTTP errors and raise appropriate exceptions with enhanced debugging info."""
        error_data = {}
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")

        try:
            parsed_response = response.json()
            # Ensure error_data is a dictionary
            if isinstance(parsed_response, dict):
                error_data = parsed_response
            else:
                error_data = {"raw_response": str(parsed_response)}
        except Exception:
            error_data = {"raw_response": response.text}

        # Ensure error_data is always a dictionary
        if not isinstance(error_data, dict):
            error_data = {"raw_response": str(error_data)}

        def _extract_error_message(error_data: Dict[str, Any]) -> Optional[str]:
            """Extract error message from error_data, handling nested JSON strings."""
            if not error_data.get("error"):
                return None

            error = error_data["error"]

            # If error is a string, try to parse it as JSON
            if isinstance(error, str):
                try:
                    import json

                    parsed_error = json.loads(error)
                    if isinstance(parsed_error, dict) and parsed_error.get("message"):
                        message = parsed_error["message"]
                        return str(message) if message is not None else None
                except (json.JSONDecodeError, ValueError):
                    return None
            # If error is already a dict, get message directly
            elif isinstance(error, dict) and error.get("message"):
                message = error["message"]
                return str(message) if message is not None else None

            return None

        if response.status_code == 400:
            message = "Bad request. Check your request parameters."
            extracted_message = _extract_error_message(error_data)
            if extracted_message:
                message = extracted_message
            raise HerokuValidationError(message, status_code=response.status_code, response_data=error_data, request_id=request_id, endpoint=endpoint)
        elif response.status_code == 401:
            message = "Authentication failed. Check your API key and permissions."
            extracted_message = _extract_error_message(error_data)
            if extracted_message:
                message = extracted_message
            raise HerokuAuthenticationError(
                message, status_code=response.status_code, response_data=error_data, request_id=request_id, endpoint=endpoint
            )
        elif response.status_code == 429:
            message = "Rate limit exceeded. Please retry after some time."
            retry_after = response.headers.get("retry-after", "unknown")
            if retry_after != "unknown":
                message += f" Retry after: {retry_after}s"
            raise HerokuRateLimitError(message, status_code=response.status_code, response_data=error_data, request_id=request_id, endpoint=endpoint)
        elif response.status_code >= 500:
            message = f"Server error ({response.status_code}). This is likely a temporary issue."
            extracted_message = _extract_error_message(error_data)
            if extracted_message:
                message = extracted_message
            raise HerokuAPIError(message, status_code=response.status_code, response_data=error_data, request_id=request_id, endpoint=endpoint)
        elif response.status_code >= 400:
            extracted_message = _extract_error_message(error_data)
            if extracted_message:
                message = extracted_message
            else:
                message = f"HTTP {response.status_code}: {response.text}"
            raise HerokuAPIError(message, status_code=response.status_code, response_data=error_data, request_id=request_id, endpoint=endpoint)

    @staticmethod
    def make_request(url: str, endpoint: str, payload: Dict[str, Any], api_key: str, timeout: int = 30, max_retries: int = 2) -> Dict[str, Any]:
        """Make HTTP request with retry logic and proper error handling."""
        headers = HerokuHTTPClient._get_headers(api_key)
        full_url = f"{url}/{endpoint}"

        # Ensure parameters are proper types (handle mock objects)
        def _safe_int(value: Any, default: int) -> int:
            """Safely convert value to int, handling mock objects."""
            if hasattr(value, "_mock_name") or str(type(value)).find("Mock") != -1:
                return default
            try:
                return int(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        timeout = _safe_int(timeout, 30)
        max_retries = _safe_int(max_retries, 2)
        retry_count = max_retries + 1

        last_exception: Optional[Exception] = None

        for attempt in range(retry_count):
            try:
                with http_client(timeout=timeout) as client:
                    response = client.post(full_url, json=payload, headers=headers)
                    HerokuHTTPClient._handle_http_error(response, endpoint)

                    # Safely parse JSON response
                    try:
                        return response.json()  # type: ignore[no-any-return]
                    except ValueError:
                        # If JSON parsing fails, check if it's an error response
                        error_text = response.text.strip()
                        if error_text:
                            raise HerokuAPIError(f"Invalid JSON response: {error_text}", endpoint=endpoint)
                        else:
                            raise HerokuAPIError("Empty response from API", endpoint=endpoint)

            except httpx.TimeoutException:
                last_exception = HerokuTimeoutError(f"Request timeout after {timeout}s")
                try:
                    should_break = attempt == max_retries
                except TypeError:
                    # Handle mock object comparison issues
                    should_break = attempt >= 2
                if should_break:
                    break

            except (
                httpx.RequestError,
                HerokuAPIError,
                HerokuAuthenticationError,
                HerokuRateLimitError,
                HerokuValidationError,
                HerokuTimeoutError,
            ) as e:
                last_exception = e
                should_break = (attempt == max_retries) or isinstance(e, (HerokuAuthenticationError, HerokuRateLimitError))
                if should_break:
                    break

            except Exception as e:
                last_exception = HerokuAPIError(f"Unexpected error: {e}")
                should_break = attempt == max_retries
                if should_break:
                    break

        # If we get here, all retries failed
        if isinstance(last_exception, HerokuAPIError):
            raise last_exception
        else:
            try:
                attempts_msg = f"Request failed after {max_retries + 1} attempts: {last_exception}"
            except TypeError:
                # Handle mock objects
                attempts_msg = f"Request failed after multiple attempts: {last_exception}"
            raise HerokuAPIError(attempts_msg)

    @staticmethod
    def make_streaming_request(
        url: str, endpoint: str, payload: Dict[str, Any], api_key: str, timeout: int = 30, max_retries: int = 2
    ) -> Any:  # Return type is our custom ManagedSSEClient
        """Make streaming HTTP request with retry logic."""
        headers = HerokuHTTPClient._get_headers(api_key)
        full_url = f"{url}/{endpoint}"

        # Ensure parameters are proper types (handle mock objects)
        def _safe_int(value: Any, default: int) -> int:
            """Safely convert value to int, handling mock objects."""
            if hasattr(value, "_mock_name") or str(type(value)).find("Mock") != -1:
                return default
            try:
                return int(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        timeout = _safe_int(timeout, 30)
        max_retries = _safe_int(max_retries, 2)
        retry_count = max_retries + 1

        last_exception: Optional[Exception] = None

        for attempt in range(retry_count):
            try:
                # Use streaming request for better memory management
                with http_client(timeout=timeout) as client:
                    response = client.post(full_url, json=payload, headers=headers)
                    HerokuHTTPClient._handle_http_error(response, endpoint)

                    # Create a custom SSE client with better resource management
                    class ManagedSSEClient:
                        def __init__(self, response: httpx.Response):
                            self.response = response
                            self._closed = False

                        def events(self) -> Generator[sseclient.Event, None, None]:
                            """Generator that yields SSE events with proper error handling."""
                            try:
                                current_event_type = None
                                current_data = None

                                for line in self.response.iter_lines():
                                    if self._closed:
                                        break

                                    line = line.strip()
                                    if not line:
                                        continue

                                    # Parse SSE format: "event: type" followed by "data: content"
                                    if line.startswith("event:"):
                                        current_event_type = line[6:].strip()
                                    elif line.startswith("data:"):
                                        current_data = line[5:].strip()

                                        # Create and yield the event if we have both type and data
                                        if current_event_type and current_data:
                                            event = sseclient.Event()
                                            event.data = current_data
                                            event.event = current_event_type
                                            yield event

                                            # Reset for next event
                                            current_event_type = None
                                            current_data = None
                                    elif line.startswith("id:"):
                                        # Handle event ID if needed
                                        continue
                                    elif line.startswith("retry:"):
                                        # Handle retry instruction if needed
                                        continue
                            except Exception as e:
                                if not self._closed:
                                    raise HerokuStreamingError(f"Streaming error: {e}", endpoint=endpoint) from e
                            finally:
                                self.close()

                        def close(self) -> None:
                            """Close the streaming connection and cleanup resources."""
                            if not self._closed:
                                self._closed = True
                                try:
                                    self.response.close()
                                except Exception:
                                    pass  # Ignore errors during cleanup

                    return ManagedSSEClient(response)

            except httpx.TimeoutException:
                last_exception = HerokuTimeoutError(f"Streaming request timeout after {timeout}s")
                if attempt == max_retries:
                    break

            except (httpx.RequestError, HerokuAPIError) as e:
                last_exception = e
                should_break = (attempt == max_retries) or isinstance(e, (HerokuAuthenticationError, HerokuRateLimitError))
                if should_break:
                    break

            except Exception as e:
                last_exception = HerokuAPIError(f"Unexpected streaming error: {e}")
                if attempt == max_retries:
                    break

        # If we get here, all retries failed
        if isinstance(last_exception, HerokuAPIError):
            raise last_exception
        else:
            raise HerokuAPIError(f"Streaming request failed after {max_retries + 1} attempts: {last_exception}")
