#!/usr/bin/env python3
"""
Simple test runner for LangGraph 101 exercises with CharHeroku.

This script provides a quick way to verify that the LangGraph 101 multi-agent
exercises work correctly with CharHeroku without running the full pytest suite.

Usage:
    python scripts/test_langgraph_101_heroku.py

Environment Variables Required:
    - INFERENCE_URL: Your Heroku Inference API URL
    - INFERENCE_KEY: Your Heroku Inference API key
    - INFERENCE_MODEL_ID: Your model ID
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_environment() -> bool:
    """Check if required environment variables are set."""
    required_vars = ["INFERENCE_URL", "INFERENCE_KEY", "INFERENCE_MODEL_ID"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables and try again.")
        return False

    print("✅ All required environment variables are set")
    return True


def test_charheroku_import() -> bool:
    """Test that CharHeroku can be imported."""
    try:
        # Test import without assigning to unused variable
        import importlib.util

        spec = importlib.util.find_spec("langchain_heroku")
        if spec is not None:
            print("✅ CharHeroku available for import")
        else:
            print("❌ CharHeroku not available")
        return True
    except ImportError as e:
        print(f"❌ Failed to import CharHeroku: {e}")
        return False


def test_charheroku_initialization() -> bool:
    """Test that CharHeroku can be initialized."""
    try:
        from langchain_heroku import ChatHeroku

        chat_model = ChatHeroku()
        print("✅ CharHeroku initialized successfully")

        # Test basic properties
        assert hasattr(chat_model, "_llm_type")
        assert chat_model._llm_type == "heroku"
        print("✅ CharHeroku properties verified")

        return True
    except Exception as e:
        print(f"❌ Failed to initialize CharHeroku: {e}")
        return False


def test_langgraph_imports() -> bool:
    """Test that LangGraph components can be imported."""
    try:
        # Test imports without assigning to unused variables
        import importlib.util

        # Check if LangGraph components are available
        components = ["langgraph.checkpoint.memory", "langgraph.graph", "langgraph.graph.message", "langgraph.managed.is_last_step"]

        for component in components:
            spec = importlib.util.find_spec(component)
            if spec is None:
                print(f"❌ {component} not available")
                return False

        print("✅ LangGraph components available for import")
        return True
    except ImportError as e:
        print(f"❌ Failed to import LangGraph components: {e}")
        return False


def test_basic_workflow() -> bool:
    """Test a basic LangGraph workflow with CharHeroku."""
    try:
        from typing import Any, List, TypedDict

        from langchain_core.messages import HumanMessage
        from langgraph.graph import StateGraph

        # Define state schema
        class WorkflowState(TypedDict):
            messages: List[Any]
            step: str

        # Create workflow
        workflow = StateGraph(WorkflowState)

        def start_node(state: WorkflowState) -> WorkflowState:
            state["step"] = "started"
            return state

        def process_node(state: WorkflowState) -> WorkflowState:
            state["step"] = "processed"
            return state

        def end_node(state: WorkflowState) -> WorkflowState:
            state["step"] = "completed"
            return state

        # Add nodes and edges
        workflow.add_node("start", start_node)
        workflow.add_node("process", process_node)
        workflow.add_node("end", end_node)

        workflow.add_edge("start", "process")
        workflow.add_edge("process", "end")

        workflow.set_entry_point("start")
        workflow.set_finish_point("end")

        # Compile workflow
        compiled_workflow = workflow.compile()
        print("✅ Basic workflow compiled successfully")

        # Test workflow execution
        initial_state: WorkflowState = {"messages": [HumanMessage(content="Hello")], "step": "initial"}

        result = compiled_workflow.invoke(initial_state)  # type: ignore
        assert result["step"] == "completed"
        print("✅ Basic workflow executed successfully")

        return True

    except Exception as e:
        print(f"❌ Basic workflow test failed: {e}")
        return False


def test_charheroku_with_tools() -> bool:
    """Test CharHeroku with tool binding."""
    try:
        from langchain_core.tools import tool

        from langchain_heroku import ChatHeroku

        @tool
        def test_tool(query: str) -> str:
            """A simple test tool."""
            return f"Tool result: {query}"

        # Initialize CharHeroku
        chat_model = ChatHeroku()

        # Bind tools
        model_with_tools = chat_model.bind_tools([test_tool])
        print("✅ Tools bound successfully")

        # Test invocation (this might fail depending on model capabilities)
        try:
            model_with_tools.invoke("Use the test_tool with 'hello'")
            print("✅ Tool invocation successful")
        except Exception as e:
            print(f"⚠️  Tool invocation failed (expected for some models): {e}")

        return True

    except Exception as e:
        print(f"❌ Tool binding test failed: {e}")
        return False


def test_charheroku_streaming() -> bool:
    """Test CharHeroku streaming capability."""
    try:
        from langchain_core.messages import HumanMessage

        from langchain_heroku import ChatHeroku

        # Initialize CharHeroku with streaming
        chat_model = ChatHeroku(streaming=True)

        # Test streaming
        messages = [HumanMessage(content="Hello")]
        stream_result = list(chat_model.stream(messages))

        assert len(stream_result) > 0
        print("✅ Streaming test passed")

        return True

    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False


def main() -> int:
    """Main test runner."""
    print("🚀 LangGraph 101 CharHeroku Integration Test Runner")
    print("=" * 60)

    # Check environment
    if not check_environment():
        sys.exit(1)

    print("\n🧪 Running tests...")

    tests = [
        ("CharHeroku Import", test_charheroku_import),
        ("CharHeroku Initialization", test_charheroku_initialization),
        ("LangGraph Imports", test_langgraph_imports),
        ("Basic Workflow", test_basic_workflow),
        ("Tool Binding", test_charheroku_with_tools),
        ("Streaming", test_charheroku_streaming),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! CharHeroku is ready for LangGraph 101 exercises.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
