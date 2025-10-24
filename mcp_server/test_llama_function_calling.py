#!/usr/bin/env python3
"""
Test if Llama 3.2 supports function calling

This script verifies if the loaded Llama 3.2 model can handle tool definitions
in its chat template.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_function_calling_support():
    """Test if Llama 3.2 tokenizer supports function calling"""

    print("=" * 80)
    print("TESTING LLAMA 3.2 FUNCTION CALLING SUPPORT")
    print("=" * 80)

    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    print(f"\nLoading tokenizer: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir="/llms"
        )
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False

    # Define a simple tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_schema",
                "description": "Get database schema",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]

    messages = [
        {"role": "user", "content": "Show me the database schema"}
    ]

    print("\nTesting chat template with tools...")
    try:
        # Try to apply chat template with tools
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,  # This is the key parameter
            tokenize=False,
            add_generation_prompt=True
        )

        print("✓ Chat template supports tools!")
        print("\nGenerated prompt with tools:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)

        return True

    except TypeError as e:
        if "unexpected keyword argument 'tools'" in str(e):
            print("✗ This tokenizer does NOT support function calling")
            print("   The 'tools' parameter is not accepted")

            # Try without tools
            print("\nFalling back to standard chat template (no tools):")
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print(prompt)

            return False
        else:
            print(f"✗ Unexpected error: {e}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    supports_function_calling = test_function_calling_support()

    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)

    if supports_function_calling:
        print("✓ Llama 3.2 SUPPORTS function calling!")
        print("  You can use MCP with native tool calling")
    else:
        print("✗ Llama 3.2 does NOT support function calling natively")
        print("  You'll need to use prompt-based tool calling (Approach 2)")
        print("\nRecommendation:")
        print("  Use prompt engineering to simulate tool calling:")
        print("  - Add tool descriptions to system prompt")
        print("  - Parse model output for tool calls")
        print("  - Execute tools and feed results back")

    return 0 if supports_function_calling else 1


if __name__ == "__main__":
    sys.exit(main())
