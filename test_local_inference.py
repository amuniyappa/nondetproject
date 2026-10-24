#!/usr/bin/env python3
"""
Verification Script for Agentless Qwen3-8B-AWQ Integration.

Tests:
1. Direct connection to the local OpenAI-compatible server.
2. Agentless model factory (make_model with backend="qwen" / "local").
3. Generation and thinking-token stripping.
"""

import logging
import os
import sys

# Add Agentless-main to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Agentless-main"))

from agentless.util.model import make_model
from agentless.util.api_requests import strip_thinking_tokens


def test_thinking_tag_stripping():
    sample_reasoning = "<think>\nLet's analyze the issue. The file path is app.py.\n</think>\n```python\nprint('hello')\n```"
    stripped = strip_thinking_tokens(sample_reasoning)
    expected = "```python\nprint('hello')\n```"
    assert stripped == expected, f"Expected {expected}, got {stripped}"
    print("✓ Thinking tag stripping test passed!")


def test_agentless_model_factory(base_url="http://localhost:8000/v1", model_name="qwen3.5-4b-super-coder"):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_qwen")

    # Ensure URL ends with /v1
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    print(f"\nTesting Agentless make_model(model='{model_name}', backend='qwen', base_url='{base_url}')...")
    model = make_model(
        model=model_name,
        backend="qwen",
        logger=logger,
        base_url=base_url,
        temperature=0.0,
        max_tokens=100,
    )

    test_prompt = "Return exactly the word 'SUCCESS' and nothing else."
    print(f"Sending prompt: {test_prompt}")
    try:
        trajs = model.codegen(test_prompt, num_samples=1)
        print("Response received from local server:")
        print("---")
        print(trajs[0]["response"])
        print("---")
        print(f"Token usage: {trajs[0]['usage']}")
        print("✓ Agentless model generation test passed!")
    except Exception as e:
        print(f"\n[!] Connection to {base_url} failed: {e}")
        print("Make sure your local server is running (e.g. `python serve_local.py`, LM Studio, or vLLM).")


if __name__ == "__main__":
    test_thinking_tag_stripping()
    base_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    model_name = sys.argv[2] if len(sys.argv) > 2 else os.getenv("MODEL_NAME", "qwen3.5-4b-super-coder")
    test_agentless_model_factory(base_url=base_url, model_name=model_name)
