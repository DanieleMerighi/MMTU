#!/usr/bin/env python3
"""
Universal MCP Bridge for Local LLMs

This bridge provides MCP tool calling support for ANY local LLM, with automatic
detection of native function calling capability and fallback to prompt-based approach.

Supports:
1. Native function calling (Llama 3.1+, Qwen 2.5, Mistral, etc.)
2. Prompt-based tool calling (any model)
3. Automatic detection and fallback

Usage:
    bridge = MCPBridge(model, tokenizer, mcp_server)
    response = bridge.chat_with_tools(messages, max_tools_rounds=3)
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from table_mcp_server import TableMCPServer


class MCPBridge:
    """
    Universal bridge between LLMs and MCP servers

    Automatically detects if model supports native function calling,
    otherwise falls back to prompt-based approach.
    """

    def __init__(
        self,
        model,
        tokenizer,
        mcp_server: TableMCPServer,
        mode: str = "auto"
    ):
        """
        Initialize MCP bridge

        Args:
            model: HuggingFace model or pipeline
            tokenizer: HuggingFace tokenizer
            mcp_server: MCP server instance
            mode: "auto" (detect), "native" (force native), or "prompt" (force prompt-based)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.mcp_server = mcp_server
        self.mode = mode

        # Detect function calling support
        if mode == "auto":
            self.supports_native = self._detect_function_calling_support()
        elif mode == "native":
            self.supports_native = True
        else:
            self.supports_native = False

        print(f"ℹ️  MCP Bridge initialized in {'NATIVE' if self.supports_native else 'PROMPT-BASED'} mode")

    def _detect_function_calling_support(self) -> bool:
        """
        Detect if tokenizer supports native function calling

        Returns:
            True if native function calling supported
        """
        try:
            # Try to apply chat template with tools
            test_messages = [{"role": "user", "content": "test"}]
            test_tools = [{"type": "function", "function": {"name": "test", "description": "test", "parameters": {}}}]

            self.tokenizer.apply_chat_template(
                test_messages,
                tools=test_tools,
                tokenize=False
            )
            return True
        except (TypeError, AttributeError):
            return False

    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        max_tool_rounds: int = 3,
        max_new_tokens: int = 2048,
        temperature: float = 0.0
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Chat with model using MCP tools

        Args:
            messages: Chat messages in OpenAI format
            max_tool_rounds: Maximum number of tool calling rounds
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (final_response, tool_calls_history)
        """
        if self.supports_native:
            return self._chat_with_native_tools(
                messages, max_tool_rounds, max_new_tokens, temperature
            )
        else:
            return self._chat_with_prompt_tools(
                messages, max_tool_rounds, max_new_tokens, temperature
            )

    def _chat_with_native_tools(
        self,
        messages: List[Dict[str, str]],
        max_tool_rounds: int,
        max_new_tokens: int,
        temperature: float
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Chat using native function calling

        This method uses the model's built-in function calling capability
        """
        tools = self.mcp_server.get_tools_schema()
        tool_calls_history = []

        for round_idx in range(max_tool_rounds):
            # Apply chat template with tools
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate response
            # Check if model is a pipeline or raw model
            if hasattr(self.model, '__call__') and not hasattr(self.model, 'generate'):
                # It's a pipeline - call directly
                response = self.model(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    return_full_text=False
                )[0]['generated_text']
            else:
                # It's a raw model - use generate
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse for tool calls
            tool_call = self._parse_native_tool_call(response)

            if tool_call is None:
                # No tool call, return final response
                return response, tool_calls_history

            # Execute tool
            tool_result = self.mcp_server.execute_tool(
                tool_call["name"],
                tool_call["arguments"]
            )

            # Record tool call
            tool_calls_history.append({
                "round": round_idx + 1,
                "tool": tool_call["name"],
                "arguments": tool_call["arguments"],
                "result": tool_result
            })

            # Add tool result to conversation
            messages.append({
                "role": "assistant",
                "content": f"Tool call: {tool_call['name']}({tool_call['arguments']})"
            })
            messages.append({
                "role": "tool",
                "content": tool_result
            })

        # Max rounds reached
        return response, tool_calls_history

    def _chat_with_prompt_tools(
        self,
        messages: List[Dict[str, str]],
        max_tool_rounds: int,
        max_new_tokens: int,
        temperature: float
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Chat using prompt-based tool calling

        This method works with ANY model by teaching it to use tools via prompting
        """
        tools = self.mcp_server.get_tools_schema()
        tool_calls_history = []

        # Add system message with tool instructions
        system_message = self._create_tool_system_message(tools)
        messages_with_tools = [
            {"role": "system", "content": system_message}
        ] + messages

        for round_idx in range(max_tool_rounds):
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages_with_tools,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate response
            # Check if model is a pipeline or raw model
            if hasattr(self.model, '__call__') and not hasattr(self.model, 'generate'):
                # It's a pipeline - call directly
                response = self.model(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    return_full_text=False
                )[0]['generated_text']
            else:
                # It's a raw model - use generate
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0
                )
                # Skip prompt tokens in response
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Parse for tool calls using prompt-based markers
            tool_call = self._parse_prompt_tool_call(response)

            if tool_call is None:
                # No tool call, return final response
                # Clean up tool markers if present
                response = self._clean_tool_markers(response)
                return response, tool_calls_history

            # Execute tool
            tool_result = self.mcp_server.execute_tool(
                tool_call["name"],
                tool_call["arguments"]
            )

            # Record tool call
            tool_calls_history.append({
                "round": round_idx + 1,
                "tool": tool_call["name"],
                "arguments": tool_call["arguments"],
                "result": tool_result
            })

            # Add tool result to conversation
            messages_with_tools.append({
                "role": "assistant",
                "content": f"<tool_call>{tool_call['name']}({json.dumps(tool_call['arguments'])})</tool_call>"
            })
            messages_with_tools.append({
                "role": "user",
                "content": f"<tool_result>\n{tool_result}\n</tool_result>"
            })

        # Max rounds reached, return last response
        return self._clean_tool_markers(response), tool_calls_history

    def _create_tool_system_message(self, tools: List[Dict]) -> str:
        """
        Create system message that teaches model to use tools

        Args:
            tools: List of tool definitions

        Returns:
            System message with tool instructions
        """
        tool_descriptions = []
        for tool in tools:
            func = tool["function"]
            params = func.get("parameters", {}).get("properties", {})

            params_str = ", ".join([
                f"{name}: {info.get('type', 'any')}"
                for name, info in params.items()
            ])

            tool_descriptions.append(
                f"- {func['name']}({params_str}): {func['description']}"
            )

        return f"""You have access to the following tools to help answer questions about the database:

{chr(10).join(tool_descriptions)}

To use a tool, respond with:
<tool_call>tool_name({{"param": "value"}})</tool_call>

The tool result will be provided, and you can then use it to answer the question.
If you don't need any tools, just respond normally.

Important: Only use tools when needed. For simple questions, answer directly."""

    def _parse_native_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse tool call from native function calling format

        Args:
            response: Model response

        Returns:
            Tool call dict or None if no tool call
        """
        # This depends on model-specific format
        # Llama 3.1+ uses <|python_tag|> format
        # For now, return None (will be implemented based on actual model format)
        return None

    def _parse_prompt_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse tool call from prompt-based format

        Args:
            response: Model response

        Returns:
            Tool call dict or None if no tool call
        """
        # Look for <tool_call>name(args)</tool_call>
        match = re.search(r'<tool_call>(\w+)\((.*?)\)</tool_call>', response, re.DOTALL)

        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments (JSON or simple key=value)
        try:
            # Try JSON format
            arguments = json.loads(args_str) if args_str.strip() else {}
        except json.JSONDecodeError:
            # Try key=value format: table_name="conference", n=10
            arguments = {}
            for part in args_str.split(','):
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    # Try to convert to int if possible
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                    arguments[key] = value

        return {
            "name": tool_name,
            "arguments": arguments
        }

    def _clean_tool_markers(self, text: str) -> str:
        """Remove tool call markers from text"""
        # Remove <tool_call>...</tool_call>
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        # Remove <tool_result>...</tool_result>
        text = re.sub(r'<tool_result>.*?</tool_result>', '', text, flags=re.DOTALL)
        return text.strip()


def test_bridge():
    """Test the MCP bridge with a simple example"""
    from transformers import pipeline

    print("=" * 80)
    print("TESTING MCP BRIDGE")
    print("=" * 80)

    # Create simple test MCP server
    import os
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data_sqlite/spider/spider_data/test_database/conference/conference.sqlite"
    )

    mcp_server = TableMCPServer()
    mcp_server.connect(db_path)

    # For testing, we'll use a simple mock model
    class MockModel:
        def generate(self, **kwargs):
            # Mock response that calls get_schema
            import torch
            return torch.tensor([[1, 2, 3]])  # Dummy output

    class MockTokenizer:
        def apply_chat_template(self, messages, tools=None, **kwargs):
            if tools:
                raise TypeError("unexpected keyword argument 'tools'")  # Simulate no native support
            return "Test prompt"

        def __call__(self, text, **kwargs):
            import torch
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, tokens, **kwargs):
            return "<tool_call>get_schema({})</tool_call>"

    model = MockModel()
    tokenizer = MockTokenizer()

    # Create bridge
    bridge = MCPBridge(model, tokenizer, mcp_server, mode="prompt")

    print("\n✓ Bridge created successfully")
    print(f"  Mode: {'Native' if bridge.supports_native else 'Prompt-based'}")

    mcp_server.disconnect()


if __name__ == "__main__":
    test_bridge()
