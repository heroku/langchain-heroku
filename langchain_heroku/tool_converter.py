"""Tool conversion utilities for Heroku API format."""

import inspect
from functools import lru_cache
from typing import Any, Callable, Dict, List, Sequence, Union

# No additional imports needed
from langchain_core.tools import BaseTool


class ToolConverter:
    """Converts various tool formats to Heroku API format."""

    @staticmethod
    def convert_tools(tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]]) -> List[Dict[str, Any]]:
        """Convert tools to the API format expected by Heroku Inference API.

        Args:
            tools: Sequence of tools in various formats

        Returns:
            List of tools in Heroku API format
        """
        api_tools = []

        for tool in tools:
            if isinstance(tool, dict):
                api_tools.append(ToolConverter._convert_dict_tool(tool))
            elif isinstance(tool, BaseTool):
                api_tools.append(ToolConverter._convert_base_tool(tool))
            elif callable(tool):
                api_tools.append(ToolConverter._convert_callable_tool(tool))
            elif isinstance(tool, type):
                api_tools.append(ToolConverter._convert_class_tool(tool))

        return api_tools

    @staticmethod
    def _convert_dict_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dict tool (already in API format)."""
        return tool

    @staticmethod
    def _convert_base_tool(tool: BaseTool) -> Dict[str, Any]:
        """Convert BaseTool to API format."""
        function_def: Dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
        }

        # Add parameters if available
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                # Get the JSON schema from the Pydantic model
                if hasattr(tool.args_schema, "model_json_schema"):
                    schema = tool.args_schema.model_json_schema()
                    function_def["parameters"] = schema
            except Exception:
                # Fallback if schema extraction fails
                pass

        return {"type": "function", "function": function_def}

    @staticmethod
    def _convert_callable_tool(tool: Callable) -> Dict[str, Any]:
        """Convert callable function to API format."""
        func_name = getattr(tool, "__name__", str(tool))
        func_description = getattr(tool, "__doc__", f"Function {func_name}")

        function_def: Dict[str, Any] = {
            "name": func_name,
            "description": func_description.strip() if func_description else func_name,
        }

        # Try to extract parameters from function signature
        try:
            schema = ToolConverter._get_function_schema(tool)
            if schema["properties"]:
                function_def["parameters"] = schema
        except Exception:
            # If signature extraction fails, continue without parameters
            pass

        return {"type": "function", "function": function_def}

    @staticmethod
    def _convert_class_tool(tool: type) -> Dict[str, Any]:
        """Convert class type to API format."""
        class_name = getattr(tool, "__name__", str(tool))
        class_description = getattr(tool, "__doc__", f"Tool {class_name}")

        function_def: Dict[str, Any] = {
            "name": class_name,
            "description": class_description.strip() if class_description else class_name,
        }

        # Try to extract schema if it's a Pydantic model
        try:
            if hasattr(tool, "model_json_schema"):
                schema = tool.model_json_schema()
                function_def["parameters"] = schema
        except Exception:
            pass

        return {"type": "function", "function": function_def}

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_function_schema(func: Callable) -> Dict[str, Any]:
        """Extract JSON schema from function signature with caching.

        Args:
            func: Function to analyze

        Returns:
            JSON schema dict for the function parameters
        """
        sig = inspect.signature(func)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            if param_name != "self":  # Skip self parameter
                param_info = ToolConverter._get_param_info(param)

                # Add to required if no default value
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

                properties[param_name] = param_info

        return {"type": "object", "properties": properties, "required": required}

    @staticmethod
    def _get_param_info(param: inspect.Parameter) -> Dict[str, Any]:
        """Get parameter info for JSON schema.

        Args:
            param: Function parameter to analyze

        Returns:
            Parameter info dict for JSON schema
        """
        param_info = {"type": "string"}  # Default type

        # Try to infer type from annotation
        if param.annotation != inspect.Parameter.empty:
            if param.annotation is int:
                param_info["type"] = "integer"
            elif param.annotation is float:
                param_info["type"] = "number"
            elif param.annotation is bool:
                param_info["type"] = "boolean"
            elif hasattr(param.annotation, "__origin__"):
                # Handle generic types like Optional[str], List[str], etc.
                origin = param.annotation.__origin__
                if origin is Union:
                    # Handle Optional types (Union[T, None])
                    args = param.annotation.__args__
                    if len(args) == 2 and type(None) in args:
                        non_none_type = next(arg for arg in args if arg is not type(None))
                        if non_none_type is int:
                            param_info["type"] = "integer"
                        elif non_none_type is float:
                            param_info["type"] = "number"
                        elif non_none_type is bool:
                            param_info["type"] = "boolean"
                elif origin in (list, List):
                    param_info["type"] = "array"

        return param_info
